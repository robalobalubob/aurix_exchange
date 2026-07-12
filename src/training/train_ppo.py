"""Train PPO on the clean OU trading core — Phase 2, third policy-gradient rung.

The step up from A2C is sample reuse under a trust region: each batch of
episodes is consumed for several epochs of minibatched updates, with the
probability-ratio clip keeping the updated policy close to the one that
collected the data. Collector, network capacity, GAE machinery, CRN regret
harness, and save-best rule are all identical to the A2C rung — the ladder
isolates the algorithmic change (roadmap v2, Phase 2).

Advantages are normalized per minibatch (the standard PPO convention), and the
clip fraction — the share of samples where the ratio clip is active — is logged
as the trust-region diagnostic.

Usage:
    python -m src.training.train_ppo
    python -m src.training.train_ppo --updates 600 --seed 0
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
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
from src.training.train_a2c import gae
from src.training.train_core import solve_or_load
from src.training.train_reinforce import _save, collect_batch, greedy_policy


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

    # Bookkeeping
    log_every: int = 5
    checkpoint_path: str = "exports/ppo_last.pt"
    best_path: str = "exports/ppo_best.pt"
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

    Returns (loss, mean-entropy, clip-fraction) — clip fraction is the share of
    samples where the ratio clip binds, the trust-region health diagnostic.
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
    torch.manual_seed(cfg.seed if cfg.seed is not None else 0)
    rng = np.random.default_rng(cfg.seed)

    # Ground truth + precomputed pi* returns on the eval seed block (paired regret).
    res = solve_or_load(core, dp.DPGrid(), cfg.dp_cache_dir)
    eval_seeds = regret.make_seed_block(cfg.eval_episodes, cfg.eval_seed0)
    pistar_eval_returns, _ = regret.rollout(regret.from_dp(res), core, eval_seeds)

    envs = [OUTradingEnv(config=core) for _ in range(cfg.batch_episodes)]
    net = ActorCriticNet(obs_dim=core.obs_dim, n_actions=core.n_actions)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    best_regret = float("inf")
    pbar = tqdm(range(1, cfg.total_updates + 1), desc="train_ppo",
                unit="update", dynamic_ncols=True, smoothing=0.05)
    for update in pbar:
        obs_b, act_b, rew_b = collect_batch(net, envs, rng)
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
        target_t = torch.from_numpy(value_targets.reshape(-1).astype(np.float32))

        entropy, clip_fraction = 0.0, 0.0
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

        if update % cfg.eval_every == 0:
            rep = regret.evaluate_regret(
                greedy_policy(net), res, n_episodes=cfg.eval_episodes,
                seed0=cfg.eval_seed0, pistar_returns=pistar_eval_returns,
            )
            net.train()
            if rep.regret < best_regret:
                best_regret = rep.regret
                _save(net, cfg.best_path)
            pbar.write(
                f"update {update:>5} | EVAL regret {rep.regret:7.4f} "
                f"(paired {rep.paired_regret:7.4f} +/- {rep.paired_ci:.4f}) | "
                f"agree {rep.agreement_rate:.1%} | best_regret {best_regret:7.4f}"
            )

        if update % cfg.log_every == 0:
            pbar.set_postfix(
                mean_return=f"{rew_b.sum(axis=0).mean():.3f}",
                entropy=f"{entropy:.3f}", clip=f"{clip_fraction:.2f}",
                best_regret=f"{best_regret:.3f}", refresh=False,
            )

    pbar.close()
    _save(net, cfg.checkpoint_path)
    if best_regret == float("inf"):
        _save(net, cfg.best_path)

    # Ladder entry: evaluate the best checkpoint at full resolution.
    best = ActorCriticNet(obs_dim=core.obs_dim, n_actions=core.n_actions)
    best.load_state_dict(torch.load(cfg.best_path))
    final = regret.evaluate_regret(
        greedy_policy(best), res, n_episodes=cfg.final_eval_episodes,
        seed0=cfg.eval_seed0,
    )
    print("\n=== PPO on clean core - Phase 2 ladder entry ===")
    print(f"config hash {res.config_hash}")
    print(final.render())
    return best, final


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on the clean OU core.")
    parser.add_argument("--updates", type=int, default=TrainPPOConfig.total_updates)
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
