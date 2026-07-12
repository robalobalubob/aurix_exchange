"""Train A2C on the clean OU trading core — Phase 2, second policy-gradient rung.

The step up from REINFORCE is the learned, bootstrapped critic: advantages come
from Generalized Advantage Estimation (GAE) over temporal-difference residuals
instead of Monte-Carlo returns against a batch-mean baseline. Everything else is
held fixed on purpose — same collector, same network capacity, same CRN regret
harness and DP ground truth, same save-best rule — so the ladder isolates the
algorithmic change (roadmap v2, Phase 2).

Episodes run to the fixed horizon T where the core terminates (D2), so the
terminal bootstrap value is exactly zero and GAE needs no off-the-end value
estimate. With the canonical gamma=1, lambda is the only knob trading critic
bias against Monte-Carlo variance (lambda=1 recovers REINFORCE-with-critic-
baseline; lambda=0 is one-step TD).

Usage:
    python -m src.training.train_a2c
    python -m src.training.train_a2c --updates 600 --seed 0
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
from src.training.train_core import solve_or_load
from src.training.train_reinforce import _save, collect_batch, greedy_policy


@dataclass(frozen=True)
class TrainA2CConfig:
    # Environment (canonical clean core by default).
    core: OUCoreConfig = field(default_factory=OUCoreConfig)

    # Schedule: one gradient update per batch of complete episodes. Budget
    # matches the REINFORCE rung (1200 x 32 x T=200 env steps) for a fair ladder.
    total_updates: int = 1_200
    batch_episodes: int = 32

    # Optimisation
    lr: float = 3.0e-4
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 10.0

    # Periodic greedy seeded eval (regret-based save-best) + final report.
    eval_every: int = 50
    eval_episodes: int = 500
    final_eval_episodes: int = 5_000
    eval_seed0: int = 10_000

    # Bookkeeping
    log_every: int = 10
    checkpoint_path: str = "exports/a2c_last.pt"
    best_path: str = "exports/a2c_best.pt"
    dp_cache_dir: str = "exports"
    seed: Optional[int] = 0


# ---------------------------------------------------------------------------
# Pure pieces (module-level for testability)
# ---------------------------------------------------------------------------
def gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation over full fixed-horizon episodes.

    Args:
        rewards: (T, B) per-step rewards.
        values:  (T, B) critic estimates V(s_t) for the visited states.
        gamma:   discount (canonical core: 1.0).
        lam:     GAE lambda.

    Returns:
        (advantages (T, B), value_targets (T, B)) where the terminal bootstrap
        V(s_T) is exactly zero (the core *terminates* at T — D2) and value
        targets are the TD(lambda) returns ``advantages + values``.
    """
    t_max = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float64)
    next_value = np.zeros(rewards.shape[1:], dtype=np.float64)
    next_adv = np.zeros(rewards.shape[1:], dtype=np.float64)
    for t in range(t_max - 1, -1, -1):
        delta = rewards[t] + gamma * next_value - values[t]
        next_adv = delta + gamma * lam * next_adv
        adv[t] = next_adv
        next_value = values[t]
    return adv, adv + values


def normalize(adv: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-scale advantage normalization (step-size conditioning)."""
    return (adv - adv.mean()) / (adv.std() + 1e-8)


def a2c_loss(
    net: ActorCriticNet,
    obs: torch.Tensor,
    actions: torch.Tensor,
    adv: torch.Tensor,
    value_targets: torch.Tensor,
    value_coef: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, float, float]:
    """Joint actor-critic objective.

    loss = -E[log pi(a|s) A] + value_coef * MSE(V, target) - entropy_coef * H
    Returns (loss, mean-entropy, value-loss) — the extras are logged diagnostics.
    """
    logits, values = net(obs)
    dist = Categorical(logits=logits)
    policy_loss = -(dist.log_prob(actions) * adv).mean()
    value_loss = F.mse_loss(values, value_targets)
    entropy = dist.entropy().mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return loss, float(entropy.item()), float(value_loss.item())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg: TrainA2CConfig) -> tuple[ActorCriticNet, regret.RegretReport]:
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
    pbar = tqdm(range(1, cfg.total_updates + 1), desc="train_a2c",
                unit="update", dynamic_ncols=True, smoothing=0.05)
    for update in pbar:
        obs_b, act_b, rew_b = collect_batch(net, envs, rng)
        obs_t = torch.from_numpy(obs_b.reshape(-1, core.obs_dim))
        act_t = torch.from_numpy(act_b.reshape(-1))

        with torch.no_grad():
            _, values_t = net(obs_t)
        values = values_t.numpy().astype(np.float64).reshape(rew_b.shape)
        adv, value_targets = gae(rew_b, values, core.gamma, cfg.gae_lambda)

        loss, entropy, value_loss = a2c_loss(
            net, obs_t, act_t,
            torch.from_numpy(normalize(adv).reshape(-1).astype(np.float32)),
            torch.from_numpy(value_targets.reshape(-1).astype(np.float32)),
            cfg.value_coef, cfg.entropy_coef,
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
                entropy=f"{entropy:.3f}", v_loss=f"{value_loss:.3f}",
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
    print("\n=== A2C on clean core - Phase 2 ladder entry ===")
    print(f"config hash {res.config_hash}")
    print(final.render())
    return best, final


def main() -> None:
    parser = argparse.ArgumentParser(description="Train A2C on the clean OU core.")
    parser.add_argument("--updates", type=int, default=TrainA2CConfig.total_updates)
    parser.add_argument("--batch-episodes", type=int,
                        default=TrainA2CConfig.batch_episodes)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(replace(
        TrainA2CConfig(),
        total_updates=args.updates,
        batch_episodes=args.batch_episodes,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
