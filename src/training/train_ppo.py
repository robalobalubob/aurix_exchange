"""Train PPO on the clean OU core — Phase 2, third policy-gradient rung.

The step up from A2C is sample reuse under a trust region: each batch of
episodes is consumed for several epochs of minibatched updates, with the
probability-ratio clip keeping the updated policy close to the one that
collected the data. Collector, network capacity, GAE machinery, CRN regret
harness, and save-best rule are all identical to the A2C rung — the ladder
isolates the algorithmic change (roadmap v2, Phase 2).

Advantages are normalized per minibatch (the standard PPO convention), and the
clip fraction — the share of samples where the ratio clip is active — is
logged as the trust-region diagnostic.

Usage:
    python -m src.training.train_ppo
    python -m src.training.train_ppo --updates 600 --seed 0
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from src.env.ou_core import OUCoreConfig, OUTradingEnv
from src.models.policy import ActorCriticNet
from src.solvers import dp, regret
from src.training.artifacts import create_run, file_sha256
from src.training.train_a2c import gae
from src.training.train_core import solve_or_load
from src.training.train_reinforce import (
    _save,
    collect_batch,
    greedy_policy,
    held_out_seed_intervals,
)


@dataclass(frozen=True)
class TrainPPOConfig:
    # Environment (canonical clean core by default).
    core: OUCoreConfig = field(default_factory=OUCoreConfig)

    # Schedule: `epochs` passes of minibatched updates per batch of episodes.
    total_updates: int = 300
    batch_episodes: int = 32
    epochs: int = 4
    minibatch_size: int = 1_600  # T * B = 6400 -> 4 minibatches per epoch

    # Optimisation
    lr: float = 3.0e-4
    clip_eps: float = 0.2
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 10.0

    # Periodic greedy seeded eval (regret-based save-best) + final report.
    eval_every: int = 25
    eval_episodes: int = 500
    final_eval_episodes: int = 5_000
    eval_seed0: int = 10_000
    test_seed0: int = 100_000
    # Keep training market paths disjoint from model-selection and test paths.
    # This field is serialized with every run, making the rule auditable.
    exclude_evaluation_seed_blocks_from_training: bool = True

    # Bookkeeping
    log_every: int = 5
    artifact_root: str = "exports/runs"
    run_id: Optional[str] = None
    dp_cache_dir: str = "exports"
    seed: Optional[int] = 0


# ---------------------------------------------------------------------------
# Pure pieces (module-level for testability)
# ---------------------------------------------------------------------------
def ppo_loss(
    net: ActorCriticNet,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_prob: torch.Tensor,
    adv: torch.Tensor,
    value_targets: torch.Tensor,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, float, float]:
    """Clipped-surrogate objective (Schulman et al. 2017):

        L = -E[min(r A, clip(r, 1-eps, 1+eps) A)]
            + value_coef * MSE(V, target) - entropy_coef * H,
        r = pi(a|s) / pi_old(a|s).

    Returns (loss, mean-entropy, clip-fraction). Clip fraction is the share of
    samples where the ratio clip binds, a trust-region health diagnostic.
    """
    logits, values = net(obs)
    dist = Categorical(logits=logits)
    ratio = torch.exp(dist.log_prob(actions) - old_log_prob)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
    value_loss = F.mse_loss(values, value_targets)
    entropy = dist.entropy().mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    clip_fraction = float((ratio != clipped).float().mean().item())
    return loss, float(entropy.item()), clip_fraction


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg: TrainPPOConfig) -> tuple[ActorCriticNet, regret.RegretReport]:
    core = cfg.core
    evaluation_seed_intervals = held_out_seed_intervals(
        cfg.eval_seed0,
        cfg.eval_episodes,
        cfg.test_seed0,
        cfg.final_eval_episodes,
    )
    reserved_training_seeds = (
        evaluation_seed_intervals
        if cfg.exclude_evaluation_seed_blocks_from_training
        else ()
    )
    torch.manual_seed(cfg.seed if cfg.seed is not None else 0)
    rng = np.random.default_rng(cfg.seed)

    # Ground truth plus pi* returns on the validation seeds (paired regret).
    res = solve_or_load(core, dp.DPGrid(), cfg.dp_cache_dir)
    run = create_run(
        "ppo_core",
        seed=cfg.seed,
        training_config=cfg,
        environment_config=core,
        root=cfg.artifact_root,
        run_id=cfg.run_id,
        metadata={"dp_config_hash": res.config_hash},
    )
    print(f"Run artifacts: {run.run_directory}")
    eval_seeds = regret.make_seed_block(cfg.eval_episodes, cfg.eval_seed0)
    pistar_eval_returns, _ = regret.rollout(
        regret.from_dp(res), core, eval_seeds
    )

    envs = [OUTradingEnv(config=core) for _ in range(cfg.batch_episodes)]
    net = ActorCriticNet(obs_dim=core.obs_dim, n_actions=core.n_actions)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    best_regret = float("inf")
    best_validation_step: int | None = None
    pbar = tqdm(range(1, cfg.total_updates + 1), desc="train_ppo",
                unit="update", dynamic_ncols=True, smoothing=0.05)
    for update in pbar:
        obs_b, act_b, rew_b = collect_batch(
            net,
            envs,
            rng,
            reserved_seed_intervals=reserved_training_seeds,
        )
        n_samples = rew_b.size
        obs_t = torch.from_numpy(obs_b.reshape(-1, core.obs_dim))
        act_t = torch.from_numpy(act_b.reshape(-1))

        # Behaviour-policy quantities, fixed for all epochs of this batch.
        with torch.no_grad():
            logits_t, values_t = net(obs_t)
            old_log_prob = Categorical(logits=logits_t).log_prob(act_t)
        values = values_t.numpy().astype(np.float64).reshape(rew_b.shape)
        adv, value_targets = gae(rew_b, values, core.gamma, cfg.gae_lambda)
        adv_t = torch.from_numpy(adv.reshape(-1).astype(np.float32))
        target_t = torch.from_numpy(
            value_targets.reshape(-1).astype(np.float32)
        )
        value_loss_before_update = float(
            F.mse_loss(values_t, target_t).item()
        )

        loss_total = 0.0
        entropy_total = 0.0
        clip_fraction_total = 0.0
        minibatch_count = 0
        for _ in range(cfg.epochs):
            perm = torch.from_numpy(rng.permutation(n_samples))
            for start in range(0, n_samples, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                mb_adv = adv_t[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                loss, entropy, clip_fraction = ppo_loss(
                    net, obs_t[idx], act_t[idx], old_log_prob[idx],
                    mb_adv, target_t[idx],
                    cfg.clip_eps, cfg.value_coef, cfg.entropy_coef,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                optimizer.step()
                loss_total += float(loss.item())
                entropy_total += entropy
                clip_fraction_total += clip_fraction
                minibatch_count += 1

        if minibatch_count:
            mean_loss = loss_total / minibatch_count
            entropy = entropy_total / minibatch_count
            clip_fraction = clip_fraction_total / minibatch_count
        else:
            mean_loss = 0.0
            entropy = 0.0
            clip_fraction = 0.0

        if update % cfg.eval_every == 0:
            rep = regret.evaluate_regret(
                greedy_policy(net), res, n_episodes=cfg.eval_episodes,
                seed0=cfg.eval_seed0, pistar_returns=pistar_eval_returns,
            )
            net.train()
            selected_as_best = rep.regret < best_regret
            validation_checkpoint = None
            if selected_as_best:
                best_regret = rep.regret
                best_validation_step = update
                _save(net, run.best_checkpoint_path)
                validation_checkpoint = {
                    "path": run.best_checkpoint_path.name,
                    "sha256": file_sha256(run.best_checkpoint_path),
                    "selected_validation_step": update,
                }
            validation_metrics = asdict(rep)
            validation_metrics["selected_as_best"] = selected_as_best
            run.append_metrics(
                step=update,
                split="validation",
                metrics=validation_metrics,
                metadata=(
                    {"selected_checkpoint": validation_checkpoint}
                    if validation_checkpoint is not None
                    else None
                ),
            )
            pbar.write(
                f"update {update:>5} | EVAL regret {rep.regret:7.4f} "
                f"(paired {rep.paired_regret:7.4f} +/- {rep.paired_ci:.4f}) | "
                f"agree {rep.agreement_rate:.1%} | "
                f"best_regret {best_regret:7.4f}"
            )

        if update % cfg.log_every == 0:
            run.append_metrics(
                step=update,
                split="train",
                metrics={
                    "loss": mean_loss,
                    "mean_episode_return": float(
                        rew_b.sum(axis=0).mean()
                    ),
                    "entropy": entropy,
                    "value_loss_before_update": value_loss_before_update,
                    "clip_fraction": clip_fraction,
                    "best_validation_regret": (
                        best_regret if np.isfinite(best_regret) else None
                    ),
                },
            )
            pbar.set_postfix(
                mean_return=f"{rew_b.sum(axis=0).mean():.3f}",
                entropy=f"{entropy:.3f}", clip=f"{clip_fraction:.2f}",
                best_regret=f"{best_regret:.3f}", refresh=False,
            )

    pbar.close()
    _save(net, run.last_checkpoint_path)
    if best_regret == float("inf"):
        rep = regret.evaluate_regret(
            greedy_policy(net),
            res,
            n_episodes=cfg.eval_episodes,
            seed0=cfg.eval_seed0,
            pistar_returns=pistar_eval_returns,
        )
        net.train()
        best_regret = rep.regret
        best_validation_step = cfg.total_updates
        _save(net, run.best_checkpoint_path)
        validation_metrics = asdict(rep)
        validation_metrics["selected_as_best"] = True
        run.append_metrics(
            step=cfg.total_updates,
            split="validation",
            metrics=validation_metrics,
            metadata={
                "selected_checkpoint": {
                    "path": run.best_checkpoint_path.name,
                    "sha256": file_sha256(run.best_checkpoint_path),
                    "selected_validation_step": cfg.total_updates,
                }
            },
        )

    assert best_validation_step is not None
    selected_checkpoint = {
        "path": run.best_checkpoint_path.name,
        "sha256": file_sha256(run.best_checkpoint_path),
        "selected_validation_step": best_validation_step,
    }

    # The validation block selected this checkpoint. Score it once on a
    # disjoint held-out seed block for an unbiased ladder report.
    best = ActorCriticNet(obs_dim=core.obs_dim, n_actions=core.n_actions)
    best.load_state_dict(
        torch.load(
            run.best_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    )
    final = regret.evaluate_regret(
        greedy_policy(best), res, n_episodes=cfg.final_eval_episodes,
        seed0=cfg.test_seed0,
    )
    run.append_metrics(
        step=cfg.total_updates,
        split="test",
        metrics=asdict(final),
        metadata={"selected_checkpoint": selected_checkpoint},
    )
    run.finish(
        summary={
            "best_validation_regret": (
                best_regret if np.isfinite(best_regret) else None
            ),
            "test": asdict(final),
            "dp_config_hash": res.config_hash,
            "selected_checkpoint": selected_checkpoint,
        },
    )
    print("\n=== PPO on clean core - Phase 2 ladder entry ===")
    print(f"config hash {res.config_hash}")
    print(final.render())
    return best, final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO on the clean OU core."
    )
    parser.add_argument(
        "--updates", type=int, default=TrainPPOConfig.total_updates
    )
    parser.add_argument("--batch-episodes", type=int,
                        default=TrainPPOConfig.batch_episodes)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(replace(
        TrainPPOConfig(),
        total_updates=args.updates,
        batch_episodes=args.batch_episodes,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
