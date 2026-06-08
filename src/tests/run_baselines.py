"""Heuristic baselines & Phase A acceptance harness for Aurix Exchange.

Validates the environment design before any network weights are exported, against
the acceptance criteria in docs/rebalancing.md §4:

  * Random-policy survival rate >= 90% over 500 seeded episodes. A random policy
    must navigate the environment without going bankrupt, so the network has a
    stable space to optimise within.
  * A rule-based mean-reversion heuristic establishes the median terminal net
    worth the trained Double DQN must later beat by >= 1.8x.

This module only measures the *environment* (random + heuristic policies); it does
not require a trained checkpoint. The 1.8x DQN target is reported as the bar a
future trained policy must clear.

Usage:
    python -m src.tests.run_baselines
    python -m src.tests.run_baselines --episodes 500
"""
from __future__ import annotations

import argparse

import numpy as np

from src.env.aurix_env import (
    AurixExchangeEnv,
    EnvConfig,
    HOLD,
    BUY_BYRINIUM,
    SELL_BYRINIUM,
    BUY_HERBS,
    SELL_HERBS,
    BUY_TOOLS,
    SELL_TOOLS,
    IDX_B,
    IDX_H,
    IDX_T,
)

# Acceptance targets (docs/rebalancing.md §4).
_SURVIVAL_TARGET = 0.90
_DQN_OUTPERFORMANCE = 1.8
# Terminal net worth at or below this counts as ruin.
_RUIN_THRESHOLD = 1.0

# Map each commodity to its (buy, sell) action so the heuristic can trade any asset.
_TRADE_ACTIONS = {
    IDX_B: (BUY_BYRINIUM, SELL_BYRINIUM),
    IDX_H: (BUY_HERBS, SELL_HERBS),
    IDX_T: (BUY_TOOLS, SELL_TOOLS),
}


def _random_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    """Uniformly pick a *valid* action (the env coerces invalid ones to HOLD anyway)."""
    return int(rng.choice(np.flatnonzero(mask)))


def _mean_reversion_action(env: AurixExchangeEnv, mask: np.ndarray) -> int:
    """Rule-based mean-reversion trader over the normalized price z-scores.

    Observation indices 4-6 are (X_i - mu_i) / sigma_stat_i, i.e. how many stationary
    stds each log-price sits from its reversion target. The heuristic buys the most
    underpriced commodity and sells the most overpriced one, falling back to HOLD.
    """
    obs = env._normalize_obs()
    z = obs[4:7]  # [B, H, T] price z-scores

    cheapest = int(np.argmin(z))
    dearest = int(np.argmax(z))

    # Sell an overpriced holding first (lock in reversion gains), else buy a dip.
    buy_d, sell_d = _TRADE_ACTIONS[dearest]
    if z[dearest] > 0.5 and mask[sell_d]:
        return sell_d
    buy_c, _ = _TRADE_ACTIONS[cheapest]
    if z[cheapest] < -0.5 and mask[buy_c]:
        return buy_c
    return HOLD


def _run_policy(policy: str, n_episodes: int, seed0: int, cfg: EnvConfig) -> dict:
    """Roll out ``policy`` over ``n_episodes`` seeded episodes; collect terminal stats."""
    rng = np.random.default_rng(seed0)
    final_net_worth = np.empty(n_episodes, dtype=np.float64)

    for ep in range(n_episodes):
        seed = seed0 + ep
        env = AurixExchangeEnv(config=cfg, seed=seed)
        _, info = env.reset(seed=seed)
        mask = info["action_mask"]
        while True:
            if policy == "random":
                action = _random_action(mask, rng)
            else:
                action = _mean_reversion_action(env, mask)
            _, _, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            if terminated or truncated:
                break
        final_net_worth[ep] = info["net_worth"]

    return {
        "policy": policy,
        "n_episodes": n_episodes,
        "nw_mean": float(np.mean(final_net_worth)),
        "nw_median": float(np.median(final_net_worth)),
        "survival_rate": float(np.mean(final_net_worth >= _RUIN_THRESHOLD)),
    }


def run(n_episodes: int = 500, seed0: int = 10_000, cfg: EnvConfig | None = None) -> dict:
    cfg = cfg or EnvConfig()
    random_stats = _run_policy("random", n_episodes, seed0, cfg)
    heuristic_stats = _run_policy("mean_reversion", n_episodes, seed0, cfg)
    return {"random": random_stats, "heuristic": heuristic_stats}


def _print_report(results: dict) -> bool:
    """Render results and return True iff all hard acceptance criteria pass."""
    r = results["random"]
    h = results["heuristic"]

    print("=" * 64)
    print("Aurix Exchange - Phase A baselines (rebalancing.md sec 4)")
    print("=" * 64)
    print(f"random policy   : episodes {r['n_episodes']}")
    print(f"  survival rate          {r['survival_rate']:>8.1%}  (target >= {_SURVIVAL_TARGET:.0%})")
    print(f"  median net worth       {r['nw_median']:>12,.0f}")
    print(f"heuristic (mean-revert):")
    print(f"  survival rate          {h['survival_rate']:>8.1%}")
    print(f"  median net worth       {h['nw_median']:>12,.0f}")
    print("-" * 64)

    survival_ok = r["survival_rate"] >= _SURVIVAL_TARGET
    dqn_bar = h["nw_median"] * _DQN_OUTPERFORMANCE
    print(f"[{'PASS' if survival_ok else 'FAIL'}] random survival >= {_SURVIVAL_TARGET:.0%}")
    print(
        f"[ -- ] future DQN target: median net worth >= {dqn_bar:,.0f} "
        f"({_DQN_OUTPERFORMANCE}x heuristic) - requires a trained checkpoint"
    )
    print("=" * 64)
    return survival_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Aurix Exchange Phase A baselines.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed0", type=int, default=10_000)
    args = parser.parse_args()

    results = run(n_episodes=args.episodes, seed0=args.seed0)
    passed = _print_report(results)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
