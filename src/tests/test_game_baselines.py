"""Focused tests for the stationary game baseline diagnostics."""
from __future__ import annotations

import json

import pytest

from src.env.aurix_env import EnvConfig
from src.tests.run_baselines import _print_report, _write_json_atomic, run


def test_tiny_baseline_run_is_reproducible_and_complete() -> None:
    """A tiny development run should produce stable diagnostic evidence."""
    cfg = EnvConfig(t_max=8)

    first = run(n_episodes=3, seed0=10_000, cfg=cfg)
    second = run(n_episodes=3, seed0=10_000, cfg=cfg)

    assert first == second
    assert (
        first["environment"]["stationary_contract_version"]
        == "m2_stationary_v1"
    )
    assert len(first["environment"]["config_sha256"]) == 64
    assert first["seed_block"] == {
        "purpose": "development_baseline",
        "seed0": 10_000,
        "seed_last": 10_002,
        "n_episodes": 3,
    }
    assert first["diagnostic_thresholds"] == {
        "ruin_net_worth": 1.0,
        "loss_net_worth": cfg.initial_cash,
    }

    for stats in (
        first["hold"],
        first["random"],
        first["heuristic"],
        first["rested_expedition"],
    ):
        assert stats["total_actions"] == 3 * cfg.t_max
        assert sum(stats["action_counts"].values()) == stats["total_actions"]
        assert sum(stats["action_frequencies"].values()) == pytest.approx(1.0)
        assert stats["expedition_attempts"] == stats["action_counts"][
            "LAUNCH_EXPEDITION"
        ]
        assert stats["nw_p10"] <= stats["nw_median"] <= stats["nw_p90"]
        assert stats["survival_count"] + stats["ruin_count"] == 3
        assert 0.0 <= stats["loss_rate"] <= 1.0
        assert stats["terminated_count"] == 0
        assert stats["truncated_count"] == 3

    assert first["hold"]["action_counts"]["HOLD"] == 3 * cfg.t_max
    assert first["hold"]["expedition_attempts"] == 0
    expedition_counts = first["rested_expedition"]["action_counts"]
    assert expedition_counts["LAUNCH_EXPEDITION"] > 0
    assert sum(
        count
        for action, count in expedition_counts.items()
        if action not in {"HOLD", "LAUNCH_EXPEDITION"}
    ) == 0


def test_json_output_is_atomic_and_reproducible(tmp_path) -> None:
    """JSON export should round-trip and leave no temporary files behind."""
    results = run(n_episodes=2, seed0=10_000, cfg=EnvConfig(t_max=4))
    output_path = tmp_path / "nested" / "baseline.json"

    _write_json_atomic(results, output_path)
    first_bytes = output_path.read_bytes()
    _write_json_atomic(results, output_path)

    assert output_path.read_bytes() == first_bytes
    assert json.loads(output_path.read_text(encoding="utf-8")) == results
    assert list(output_path.parent.glob(".*.tmp")) == []


def test_report_acceptance_remains_random_survival_only(capsys) -> None:
    """Comparator outcomes must not change the original 90% acceptance gate."""
    results = run(n_episodes=1, seed0=10_000, cfg=EnvConfig(t_max=2))
    results["random"]["survival_rate"] = 0.89
    results["acceptance"]["random_survival_passed"] = True

    assert _print_report(results) is False
    assert "[FAIL] random survival >= 90%" in capsys.readouterr().out
