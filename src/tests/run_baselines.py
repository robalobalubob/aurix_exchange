"""Heuristic baselines & Phase A acceptance harness for Aurix Exchange.

Validates the environment design before any network weights are exported,
against the acceptance criteria in docs/rebalancing.md §4:

  * Random-policy survival rate >= 90% over 500 seeded episodes. A random
    policy must navigate the environment without going bankrupt, so the
    network has a stable space to optimise within.
  * A rule-based mean-reversion heuristic establishes the median terminal net
    worth the trained Double DQN must later beat by >= 1.8x.

This module only measures the *environment* (random + heuristic policies); it
does not require a trained checkpoint. The 1.8x DQN target is reported as the
bar a future trained policy must clear.

Usage:
    python -m src.tests.run_baselines
    python -m src.tests.run_baselines --episodes 500
    python -m src.tests.run_baselines --json exports/baselines/m2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

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
    LAUNCH_EXPEDITION,
    N_ACTIONS,
    IDX_B,
    IDX_H,
    IDX_T,
)

# Acceptance targets (docs/rebalancing.md §4).
_SURVIVAL_TARGET = 0.90
_DQN_OUTPERFORMANCE = 1.8
# Terminal net worth at or below this counts as ruin.
_RUIN_THRESHOLD = 1.0
_DEVELOPMENT_SEED_START = 10_000
_DEVELOPMENT_SEED_STOP = 10_500

# Map each commodity to its (buy, sell) action so the heuristic can trade any
# asset.
_TRADE_ACTIONS = {
    IDX_B: (BUY_BYRINIUM, SELL_BYRINIUM),
    IDX_H: (BUY_HERBS, SELL_HERBS),
    IDX_T: (BUY_TOOLS, SELL_TOOLS),
}

# Tie every readable name to its exported action constant. Building the ordered
# tuple from those indices prevents diagnostics from being silently mislabeled
# if the discrete interface ever changes.
_ACTION_NAME_BY_INDEX = {
    HOLD: "HOLD",
    BUY_BYRINIUM: "BUY_BYRINIUM",
    SELL_BYRINIUM: "SELL_BYRINIUM",
    BUY_HERBS: "BUY_HERBS",
    SELL_HERBS: "SELL_HERBS",
    BUY_TOOLS: "BUY_TOOLS",
    SELL_TOOLS: "SELL_TOOLS",
    LAUNCH_EXPEDITION: "LAUNCH_EXPEDITION",
}
_ACTION_NAMES = tuple(
    _ACTION_NAME_BY_INDEX[action] for action in range(N_ACTIONS)
)


def _random_action(mask: np.ndarray, rng: np.random.Generator) -> int:
    """Uniformly pick a valid action from the environment mask."""
    return int(rng.choice(np.flatnonzero(mask)))


def _mean_reversion_action(env: AurixExchangeEnv, mask: np.ndarray) -> int:
    """Rule-based mean-reversion trader over the normalized price z-scores.

    Observation indices 4-6 are (X_i - mu_i) / sigma_stat_i, i.e. how many
    stationary stds each log-price sits from its reversion target. The
    heuristic buys the most underpriced commodity and sells the most
    overpriced one, falling back to HOLD.
    """
    obs = env._normalize_obs()
    z = obs[4:7]  # [B, H, T] price z-scores

    cheapest = int(np.argmin(z))
    dearest = int(np.argmax(z))

    # Sell an overpriced holding first (lock in gains), else buy a dip.
    buy_d, sell_d = _TRADE_ACTIONS[dearest]
    if z[dearest] > 0.5 and mask[sell_d]:
        return sell_d
    buy_c, _ = _TRADE_ACTIONS[cheapest]
    if z[cheapest] < -0.5 and mask[buy_c]:
        return buy_c
    return HOLD


def _rested_expedition_action(env: AurixExchangeEnv, mask: np.ndarray) -> int:
    """Launch only while nearly rested; otherwise recover fatigue with HOLD.

    One step's fatigue decay is the threshold. This comparator deliberately
    never trades, so terminal wealth isolates the economic contribution of
    expedition sourcing from open-market mean reversion.
    """
    is_rested = env._fatigue <= env.cfg.fatigue_decay_rate
    if is_rested and mask[LAUNCH_EXPEDITION]:
        return LAUNCH_EXPEDITION
    return HOLD


def _run_policy(
    policy: str,
    n_episodes: int,
    seed0: int,
    cfg: EnvConfig,
) -> dict:
    """Roll out one policy and collect reproducible economic diagnostics."""
    if policy not in {"hold", "random", "mean_reversion", "rested_expedition"}:
        raise ValueError(f"unknown baseline policy: {policy}")
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")

    rng = np.random.default_rng(seed0)
    final_net_worth = np.empty(n_episodes, dtype=np.float64)
    action_counts = np.zeros(len(_ACTION_NAMES), dtype=np.int64)
    terminated_count = 0
    truncated_count = 0

    for ep in range(n_episodes):
        seed = seed0 + ep
        env = AurixExchangeEnv(config=cfg, seed=seed)
        _, info = env.reset(seed=seed)
        mask = info["action_mask"]
        while True:
            if policy == "hold":
                action = HOLD
            elif policy == "random":
                action = _random_action(mask, rng)
            elif policy == "mean_reversion":
                action = _mean_reversion_action(env, mask)
            else:
                action = _rested_expedition_action(env, mask)
            action_counts[action] += 1
            _, _, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            if terminated or truncated:
                terminated_count += int(terminated)
                truncated_count += int(truncated)
                break
        final_net_worth[ep] = info["net_worth"]

    total_actions = int(np.sum(action_counts))
    counts = {
        name: int(count) for name, count in zip(_ACTION_NAMES, action_counts)
    }
    frequencies = {
        name: float(count / total_actions)
        for name, count in zip(_ACTION_NAMES, action_counts)
    }
    survival_count = int(np.count_nonzero(final_net_worth >= _RUIN_THRESHOLD))
    ruin_count = n_episodes - survival_count
    loss_count = int(np.count_nonzero(final_net_worth < cfg.initial_cash))
    expedition_attempts = counts["LAUNCH_EXPEDITION"]

    return {
        "policy": policy,
        "n_episodes": n_episodes,
        "seed0": seed0,
        "seed_last": seed0 + n_episodes - 1,
        "policy_rng_seed": seed0 if policy == "random" else None,
        "total_actions": total_actions,
        "action_counts": counts,
        "action_frequencies": frequencies,
        "expedition_attempts": expedition_attempts,
        "expedition_attempt_rate": float(expedition_attempts / total_actions),
        "nw_mean": float(np.mean(final_net_worth)),
        "nw_median": float(np.median(final_net_worth)),
        "nw_p10": float(np.percentile(final_net_worth, 10)),
        "nw_p90": float(np.percentile(final_net_worth, 90)),
        "survival_count": survival_count,
        "survival_rate": float(survival_count / n_episodes),
        "ruin_count": ruin_count,
        "ruin_rate": float(ruin_count / n_episodes),
        "loss_count": loss_count,
        "loss_rate": float(loss_count / n_episodes),
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
    }


def run(
    n_episodes: int = 500,
    seed0: int = 10_000,
    cfg: EnvConfig | None = None,
) -> dict:
    """Evaluate both development baselines under one identified contract."""
    cfg = cfg or EnvConfig()
    canonical_config_text = json.dumps(
        asdict(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    config = json.loads(canonical_config_text)
    canonical_config = canonical_config_text.encode("utf-8")
    config_sha256 = hashlib.sha256(canonical_config).hexdigest()

    hold_stats = _run_policy("hold", n_episodes, seed0, cfg)
    random_stats = _run_policy("random", n_episodes, seed0, cfg)
    heuristic_stats = _run_policy("mean_reversion", n_episodes, seed0, cfg)
    expedition_stats = _run_policy("rested_expedition", n_episodes, seed0, cfg)
    dqn_bar = heuristic_stats["nw_median"] * _DQN_OUTPERFORMANCE
    seed_last = seed0 + n_episodes - 1
    is_development_block = (
        seed0 >= _DEVELOPMENT_SEED_START
        and seed_last < _DEVELOPMENT_SEED_STOP
    )
    return {
        "schema_version": 1,
        "environment": {
            "name": "AurixExchangeEnv",
            "stationary_contract_version": cfg.stationary_contract_version,
            "config_sha256": config_sha256,
            "config": config,
        },
        "seed_block": {
            "purpose": (
                "development_baseline" if is_development_block else "custom"
            ),
            "seed0": seed0,
            "seed_last": seed_last,
            "n_episodes": n_episodes,
        },
        "diagnostic_thresholds": {
            "ruin_net_worth": _RUIN_THRESHOLD,
            "loss_net_worth": cfg.initial_cash,
        },
        "acceptance": {
            "random_survival_target": _SURVIVAL_TARGET,
            "random_survival_passed": (
                random_stats["survival_rate"] >= _SURVIVAL_TARGET
            ),
            "dqn_outperformance_multiple": _DQN_OUTPERFORMANCE,
            "future_dqn_median_target": dqn_bar,
        },
        "hold": hold_stats,
        "random": random_stats,
        "heuristic": heuristic_stats,
        "rested_expedition": expedition_stats,
    }


def _write_json_atomic(results: dict, output_path: str | Path) -> None:
    """Atomically replace a path with deterministic, formatted JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                results,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _print_policy_stats(label: str, stats: dict) -> None:
    """Print one compact but complete policy diagnostic block."""
    print(label)
    print(
        f"  episodes / actions      {stats['n_episodes']:>8} / "
        f"{stats['total_actions']:,}"
    )
    print(
        "  terminal net worth      "
        f"mean {stats['nw_mean']:,.0f} | median {stats['nw_median']:,.0f} | "
        f"p10 {stats['nw_p10']:,.0f} | p90 {stats['nw_p90']:,.0f}"
    )
    print(
        f"  survival / ruin         {stats['survival_rate']:>8.1%} / "
        f"{stats['ruin_rate']:.1%}"
    )
    print(f"  finish below start      {stats['loss_rate']:>8.1%}")
    print(
        f"  expedition attempts     {stats['expedition_attempts']:>8,} "
        f"({stats['expedition_attempt_rate']:.1%} of actions)"
    )
    action_mix = " | ".join(
        f"{name} {stats['action_counts'][name]:,} "
        f"({stats['action_frequencies'][name]:.1%})"
        for name in _ACTION_NAMES
    )
    print(f"  action mix               {action_mix}")


def _print_report(results: dict) -> bool:
    """Render results and return True iff all hard acceptance criteria pass."""
    r = results["random"]
    h = results["heuristic"]

    print("=" * 64)
    print("Aurix Exchange - Phase A baselines (rebalancing.md sec 4)")
    print("=" * 64)
    environment = results["environment"]
    seed_block = results["seed_block"]
    print(
        f"contract: {environment['stationary_contract_version']} | "
        f"config sha256: {environment['config_sha256'][:12]}..."
    )
    print(
        f"seeds: {seed_block['seed0']}..{seed_block['seed_last']} "
        f"({seed_block['n_episodes']} paths; {seed_block['purpose']})"
    )
    print("-" * 64)
    _print_policy_stats("hold-only control:", results["hold"])
    _print_policy_stats("random policy:", r)
    _print_policy_stats("heuristic (mean-revert):", h)
    _print_policy_stats(
        "rested-expedition control:", results["rested_expedition"]
    )
    print("-" * 64)

    # Derive the exit gate from the measured random result, preserving the
    # original CLI acceptance behavior even if a loaded record has stale
    # convenience metadata.
    survival_ok = r["survival_rate"] >= _SURVIVAL_TARGET
    dqn_bar = h["nw_median"] * _DQN_OUTPERFORMANCE
    status = "PASS" if survival_ok else "FAIL"
    print(f"[{status}] random survival >= {_SURVIVAL_TARGET:.0%}")
    print(
        f"[ -- ] future DQN target: median net worth >= {dqn_bar:,.0f} "
        f"({_DQN_OUTPERFORMANCE}x heuristic) - requires a trained checkpoint"
    )
    print("=" * 64)
    return survival_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Aurix Exchange Phase A baselines."
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed0", type=int, default=10_000)
    parser.add_argument(
        "--json",
        "--json-output",
        dest="json_output",
        type=Path,
        help="optionally write the complete diagnostic record as JSON",
    )
    args = parser.parse_args()

    results = run(n_episodes=args.episodes, seed0=args.seed0)
    if args.json_output is not None:
        _write_json_atomic(results, args.json_output)
    passed = _print_report(results)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
