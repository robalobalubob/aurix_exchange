"""Train REINFORCE on the clean OU trading core — Phase 2, first policy-gradient rung.

The algorithm-ladder contract (roadmap v2, Phase 2): every learner runs on the
*identical* clean core and is scored by the same CRN regret harness against the same
DP optimum. This trainer therefore reuses ``solve_or_load`` (DP cache), the
``regret`` harness for periodic greedy seeded evaluation, and regret-based
save-best — exactly the machinery that produced the DQN's first regret number.

Algorithm: batch-episode Monte-Carlo policy gradient (REINFORCE) with two standard
critic-free variance reducers:

- a per-timestep baseline ``b_t = mean_i G_{i,t}`` over the episode batch (all
  episodes share the fixed horizon T, so this is exact, not padded); the ~1/B
  self-inclusion bias is second-order and conventional;
- global standard-deviation normalization of the advantages, which decouples the
  step size from the reward scale.

No bootstrapping and no learned critic — that is the point of this rung; the
bootstrapped-critic version is the next rung (A2C). The canonical objective is
finite-horizon undiscounted (gamma=1), so returns-to-go are plain reverse cumsums.

Usage:
    python -m src.training.train_reinforce
    python -m src.training.train_reinforce --updates 600 --seed 0
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

from src.env.ou_core import OUCoreConfig, OUTradingEnv
from src.models.policy import PolicyNet
from src.solvers import dp, regret
from src.training.train_core import solve_or_load


@dataclass(frozen=True)
class TrainReinforceConfig:
    # Environment (canonical clean core by default).
    core: OUCoreConfig = field(default_factory=OUCoreConfig)

    # Schedule: one gradient update per batch of complete episodes. 1200 updates
    # x 32 episodes x T=200 is the budget of the reported ladder entry.
    total_updates: int = 1_200
    batch_episodes: int = 32

    # Optimisation
    lr: float = 3.0e-4
    entropy_coef: float = 0.01
    grad_clip: float = 10.0

    # Periodic greedy seeded eval (regret-based save-best) + final report.
    eval_every: int = 50
    eval_episodes: int = 500
    final_eval_episodes: int = 5_000
    eval_seed0: int = 10_000

    # Bookkeeping
    log_every: int = 10
    checkpoint_path: str = "exports/reinforce_last.pt"
    best_path: str = "exports/reinforce_best.pt"
    dp_cache_dir: str = "exports"
    seed: Optional[int] = 0


# ---------------------------------------------------------------------------
# Pure pieces (module-level for testability)
# ---------------------------------------------------------------------------
def returns_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted returns-to-go along axis 0 (time): G_t = r_t + gamma * G_{t+1}.

    Works on (T,) or (T, B) arrays; gamma=1 (canonical) reduces to a reverse cumsum.
    """
    out = np.empty(rewards.shape, dtype=np.float64)
    g = np.zeros(rewards.shape[1:], dtype=np.float64)
    for t in range(rewards.shape[0] - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def advantages(returns: np.ndarray) -> np.ndarray:
    """Baselined, scale-normalized advantages from a (T, B) returns-to-go matrix.

    Per-timestep batch-mean baseline, then global std normalization (see module
    docstring for why both are bias-benign).
    """
    adv = returns - returns.mean(axis=1, keepdims=True)
    return adv / (adv.std() + 1e-8)


def reinforce_loss(
    net: PolicyNet,
    obs: torch.Tensor,
    actions: torch.Tensor,
    adv: torch.Tensor,
    entropy_coef: float,
) -> tuple[torch.Tensor, float]:
    """Policy-gradient surrogate: -E[log pi(a|s) * A] - entropy_coef * H(pi).

    Returns (loss, mean-entropy) — the entropy is logged as a collapse diagnostic.
    """
    dist = Categorical(logits=net(obs))
    log_prob = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    loss = -(log_prob * adv).mean() - entropy_coef * entropy
    return loss, float(entropy.item())


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------
def collect_batch(
    net: nn.Module,
    envs: list[OUTradingEnv],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll one complete episode in every env, sampling actions from the policy.

    ``net`` is any module exposing ``action_logits`` (PolicyNet, ActorCriticNet) —
    the whole ladder shares this one collector. The clean core has a fixed
    horizon, so the B envs run in lockstep and each timestep costs a single
    batched forward pass. Returns
    (obs (T, B, obs_dim) float32, actions (T, B) int64, rewards (T, B) float64).
    """
    cfg = envs[0].cfg
    t_max, n = cfg.t_max, len(envs)

    obs_buf = np.empty((t_max, n, cfg.obs_dim), dtype=np.float32)
    act_buf = np.empty((t_max, n), dtype=np.int64)
    rew_buf = np.empty((t_max, n), dtype=np.float64)

    obs = np.stack(
        [env.reset(seed=int(rng.integers(2**31 - 1)))[0] for env in envs]
    )
    net.eval()
    with torch.no_grad():
        for t in range(t_max):
            obs_buf[t] = obs
            logits = net.action_logits(torch.from_numpy(obs))
            actions = Categorical(logits=logits).sample().numpy()
            act_buf[t] = actions
            for i, env in enumerate(envs):
                obs_i, r, terminated, truncated, _ = env.step(int(actions[i]))
                obs[i] = obs_i
                rew_buf[t, i] = r
            # Fixed horizon: every env must terminate exactly at t_max (D2).
            assert (terminated and not truncated) == (t == t_max - 1)
    net.train()
    return obs_buf, act_buf, rew_buf


def greedy_policy(net: nn.Module) -> regret.Policy:
    """Deterministic mode-action closure for the regret harness (greedy seeded
    evaluation, matching how the DQN rung was scored). Works for any module
    exposing ``action_logits``."""
    net.eval()

    def policy(obs: np.ndarray, t: int) -> int:
        with torch.no_grad():
            logits = net.action_logits(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            )
            return int(torch.argmax(logits, dim=1).item())

    return policy


def _save(net: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(net.state_dict(), path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg: TrainReinforceConfig) -> tuple[PolicyNet, regret.RegretReport]:
    core = cfg.core
    torch.manual_seed(cfg.seed if cfg.seed is not None else 0)
    rng = np.random.default_rng(cfg.seed)

    # Ground truth + precomputed pi* returns on the eval seed block (paired regret).
    res = solve_or_load(core, dp.DPGrid(), cfg.dp_cache_dir)
    eval_seeds = regret.make_seed_block(cfg.eval_episodes, cfg.eval_seed0)
    pistar_eval_returns, _ = regret.rollout(regret.from_dp(res), core, eval_seeds)

    envs = [OUTradingEnv(config=core) for _ in range(cfg.batch_episodes)]
    net = PolicyNet(obs_dim=core.obs_dim, n_actions=core.n_actions)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    best_regret = float("inf")
    pbar = tqdm(range(1, cfg.total_updates + 1), desc="train_reinforce",
                unit="update", dynamic_ncols=True, smoothing=0.05)
    for update in pbar:
        obs_b, act_b, rew_b = collect_batch(net, envs, rng)
        adv = advantages(returns_to_go(rew_b, core.gamma))

        obs_t = torch.from_numpy(obs_b.reshape(-1, core.obs_dim))
        act_t = torch.from_numpy(act_b.reshape(-1))
        adv_t = torch.from_numpy(adv.reshape(-1).astype(np.float32))

        loss, entropy = reinforce_loss(net, obs_t, act_t, adv_t, cfg.entropy_coef)
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
                entropy=f"{entropy:.3f}", best_regret=f"{best_regret:.3f}",
                refresh=False,
            )

    pbar.close()
    _save(net, cfg.checkpoint_path)
    if best_regret == float("inf"):
        _save(net, cfg.best_path)

    # Ladder entry: evaluate the best checkpoint at full resolution.
    best = PolicyNet(obs_dim=core.obs_dim, n_actions=core.n_actions)
    best.load_state_dict(torch.load(cfg.best_path))
    final = regret.evaluate_regret(
        greedy_policy(best), res, n_episodes=cfg.final_eval_episodes,
        seed0=cfg.eval_seed0,
    )
    print("\n=== REINFORCE on clean core - Phase 2 ladder entry ===")
    print(f"config hash {res.config_hash}")
    print(final.render())
    return best, final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train REINFORCE on the clean OU core."
    )
    parser.add_argument("--updates", type=int,
                        default=TrainReinforceConfig.total_updates)
    parser.add_argument("--batch-episodes", type=int,
                        default=TrainReinforceConfig.batch_episodes)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(replace(
        TrainReinforceConfig(),
        total_updates=args.updates,
        batch_episodes=args.batch_episodes,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
