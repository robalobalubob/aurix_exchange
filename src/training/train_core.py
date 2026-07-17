"""Train the DQN/PER/dueling stack on the clean OU trading core.

This learner-alignment run uses the game trainer's network and PER machinery on
``OUTradingEnv`` (obs ``[z, f, t/T]``, 3 actions). The canonical objective has
gamma=1, termination at T, unclipped reward, and seeded PER. Its output is the
learned agent's optimality gap against DP ground truth, with a common-random-
numbers confidence interval.

The save-best signal is regret itself (lowest is best), measured by periodic
greedy seeded validation. The DP optimum is solved once and cached by config
hash.

Usage:
    python -m src.training.train_core
    python -m src.training.train_core --steps 300000 --seed 0
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.env.ou_core import OUCoreConfig, OUTradingEnv
from src.models.dqn import DuelingDQN
from src.solvers import dp, regret
from src.training.artifacts import create_run, file_sha256
from src.training.replay_buffer import PrioritizedReplayBuffer
from src.training.train_dqn import _linear_anneal, compute_loss, select_action


@dataclass(frozen=True)
class TrainCoreConfig:
    # Environment (canonical clean core by default; ablations override here).
    core: OUCoreConfig = field(default_factory=OUCoreConfig)

    # Schedule
    total_steps: int = 150_000
    warmup_steps: int = 2_000
    train_every: int = 1
    target_sync: int = 1_000

    # learner_gamma defaults to the canonical env discount. The gamma=0.99
    # ablation sets this and disables core.time_feature.
    batch_size: int = 64
    lr: float = 5.0e-4
    learner_gamma: Optional[float] = None  # None -> use core.gamma
    grad_clip: float = 10.0

    # Replay / PER
    buffer_capacity: int = 100_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000

    # Periodic validation selects the checkpoint; final evaluation uses a
    # disjoint seed block and therefore remains held out.
    eval_every: int = 10_000
    eval_episodes: int = 1_000
    final_eval_episodes: int = 5_000
    eval_seed0: int = 10_000
    test_seed0: int = 100_000

    # Bookkeeping
    log_every: int = 2_000
    artifact_root: str = "exports/runs"
    run_id: Optional[str] = None
    dp_cache_dir: str = "exports"
    seed: Optional[int] = 0

    @property
    def gamma(self) -> float:
        return (
            self.core.gamma
            if self.learner_gamma is None
            else self.learner_gamma
        )


def solve_or_load(
    core: OUCoreConfig,
    grid: dp.DPGrid,
    cache_dir: str,
) -> dp.DPResult:
    """Solve the DP optimum, caching it on disk by (config, grid) hash."""
    h = dp.config_hash(core, grid)
    path = os.path.join(cache_dir, f"dp_{h}.npz")
    if os.path.exists(path):
        return dp.DPResult.load(path)
    res = dp.solve(core, grid)
    os.makedirs(cache_dir, exist_ok=True)
    res.save(path)
    return res


def net_policy(net: DuelingDQN) -> regret.Policy:
    """Wrap a network as a greedy clean-core policy."""
    net.eval()

    def policy(obs: np.ndarray, t: int) -> int:
        with torch.no_grad():
            q = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(q, dim=1).item())

    return policy


def _save(net: DuelingDQN, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(net.state_dict(), path)


def train(cfg: TrainCoreConfig) -> tuple[DuelingDQN, regret.RegretReport]:
    core = cfg.core
    obs_dim, n_actions = core.obs_dim, core.n_actions

    act_seed, buf_seed = np.random.SeedSequence(cfg.seed).spawn(2)
    rng = np.random.default_rng(act_seed)
    torch.manual_seed(cfg.seed if cfg.seed is not None else 0)

    # Ground truth plus pi* returns on validation seeds for paired regret.
    res = solve_or_load(core, dp.DPGrid(), cfg.dp_cache_dir)
    run = create_run(
        "dqn_core",
        seed=cfg.seed,
        training_config=cfg,
        environment_config=core,
        root=cfg.artifact_root,
        run_id=cfg.run_id,
        metadata={
            "dp_config_hash": res.config_hash,
            "environment": "OUTradingEnv",
            "selection_metric": "validation.regret",
        },
    )
    print(f"run artifacts -> {run.run_directory}")

    eval_seeds = regret.make_seed_block(cfg.eval_episodes, cfg.eval_seed0)
    pistar_eval_returns, _ = regret.rollout(
        regret.from_dp(res), core, eval_seeds
    )

    env = OUTradingEnv(config=core, seed=cfg.seed)
    online = DuelingDQN(obs_dim=obs_dim, n_actions=n_actions)
    target = DuelingDQN(obs_dim=obs_dim, n_actions=n_actions)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buffer = PrioritizedReplayBuffer(
        cfg.buffer_capacity, n_actions, alpha=cfg.per_alpha, obs_dim=obs_dim,
        rng=np.random.default_rng(buf_seed),
    )

    obs, info = env.reset()
    mask = info["action_mask"]
    running_loss, loss_count = 0.0, 0
    best_regret = float("inf")
    best_validation_step: int | None = None

    pbar = tqdm(range(1, cfg.total_steps + 1), desc="train_core", unit="step",
                dynamic_ncols=True, smoothing=0.05)
    for step in pbar:
        eps = _linear_anneal(
            cfg.eps_start,
            cfg.eps_end,
            step / cfg.eps_decay_steps,
        )
        action = select_action(online, obs, mask, eps, rng)

        next_obs, reward_t, terminated, truncated, info = env.step(action)
        next_mask = info["action_mask"]
        # Only `terminated` zeroes bootstrap; the core terminates at T.
        buffer.add(obs, action, reward_t, next_obs, terminated, next_mask)
        obs, mask = next_obs, next_mask

        if terminated or truncated:
            obs, info = env.reset()
            mask = info["action_mask"]

        if step >= cfg.warmup_steps and step % cfg.train_every == 0:
            beta = _linear_anneal(cfg.per_beta_start, cfg.per_beta_end,
                                  step / cfg.total_steps)
            batch = buffer.sample(cfg.batch_size, beta)
            loss, td_errors = compute_loss(online, target, batch, cfg.gamma)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
            optimizer.step()
            buffer.update_priorities(batch["indices"], td_errors)
            running_loss += float(loss.item())
            loss_count += 1

        if step % cfg.target_sync == 0:
            target.load_state_dict(online.state_dict())

        if step >= cfg.warmup_steps and step % cfg.eval_every == 0:
            rep = regret.evaluate_regret(
                net_policy(online), res, n_episodes=cfg.eval_episodes,
                seed0=cfg.eval_seed0, pistar_returns=pistar_eval_returns,
            )
            online.train()
            selected_as_best = rep.regret < best_regret
            validation_checkpoint = None
            if selected_as_best:
                best_regret = rep.regret
                best_validation_step = step
                _save(online, str(run.best_checkpoint_path))
                validation_checkpoint = {
                    "path": run.best_checkpoint_path.name,
                    "sha256": file_sha256(run.best_checkpoint_path),
                    "selected_validation_step": step,
                }
            validation_metrics = asdict(rep)
            validation_metrics["selected_as_best"] = selected_as_best
            run.append_metrics(
                step=step,
                split="validation",
                metrics=validation_metrics,
                metadata=(
                    {"selected_checkpoint": validation_checkpoint}
                    if validation_checkpoint is not None
                    else None
                ),
            )
            pbar.write(
                f"step {step:>7} | EVAL regret {rep.regret:7.4f} "
                f"(paired {rep.paired_regret:7.4f} +/- {rep.paired_ci:.4f}) | "
                f"agree {rep.agreement_rate:.1%} | "
                f"best_regret {best_regret:7.4f}"
            )

        if step % cfg.log_every == 0:
            avg_loss = running_loss / max(loss_count, 1)
            pbar.set_postfix(eps=f"{eps:.3f}", loss=f"{avg_loss:.4f}",
                             best_regret=f"{best_regret:.3f}", refresh=False)
            run.append_metrics(
                step=step,
                split="train",
                metrics={
                    "epsilon": eps,
                    "loss_mean": avg_loss,
                    "replay_size": len(buffer),
                    "best_validation_regret": (
                        best_regret
                        if best_regret != float("inf")
                        else None
                    ),
                },
            )
            running_loss, loss_count = 0.0, 0

    pbar.close()
    _save(online, str(run.last_checkpoint_path))
    if best_regret == float("inf"):
        rep = regret.evaluate_regret(
            net_policy(online),
            res,
            n_episodes=cfg.eval_episodes,
            seed0=cfg.eval_seed0,
            pistar_returns=pistar_eval_returns,
        )
        online.train()
        best_regret = rep.regret
        best_validation_step = cfg.total_steps
        _save(online, str(run.best_checkpoint_path))
        validation_metrics = asdict(rep)
        validation_metrics["selected_as_best"] = True
        run.append_metrics(
            step=cfg.total_steps,
            split="validation",
            metrics=validation_metrics,
            metadata={
                "selected_checkpoint": {
                    "path": run.best_checkpoint_path.name,
                    "sha256": file_sha256(run.best_checkpoint_path),
                    "selected_validation_step": cfg.total_steps,
                }
            },
        )

    assert best_validation_step is not None
    selected_checkpoint = {
        "path": run.best_checkpoint_path.name,
        "sha256": file_sha256(run.best_checkpoint_path),
        "selected_validation_step": best_validation_step,
    }

    # Evaluate the selected checkpoint once on a disjoint held-out seed block.
    best = DuelingDQN(obs_dim=obs_dim, n_actions=n_actions)
    best.load_state_dict(
        torch.load(
            run.best_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    )
    final = regret.evaluate_regret(
        net_policy(best),
        res,
        n_episodes=cfg.final_eval_episodes,
        seed0=cfg.test_seed0,
    )
    run.append_metrics(
        step=cfg.total_steps,
        split="test",
        metrics=asdict(final),
        metadata={"selected_checkpoint": selected_checkpoint},
    )
    run.finish(
        summary={
            "best_validation_regret": (
                best_regret if best_regret != float("inf") else None
            ),
            "test": asdict(final),
            "dp_config_hash": res.config_hash,
            "selected_checkpoint": selected_checkpoint,
        }
    )
    print("\n=== DQN on clean core - first regret number ===")
    print(f"config hash {res.config_hash}")
    print(final.render())
    print(f"completed run -> {run.run_directory}")
    return best, final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DQN on the clean OU core."
    )
    parser.add_argument(
        "--steps", type=int, default=TrainCoreConfig.total_steps
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(replace(TrainCoreConfig(), total_steps=args.steps, seed=args.seed))


if __name__ == "__main__":
    main()
