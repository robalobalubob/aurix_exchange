"""Smoke-test that a trainer produces one complete, discoverable run."""
from __future__ import annotations

import json

import pytest

from src.training.train_a2c import TrainA2CConfig
from src.training.artifacts import create_run
from src.training.train_core import TrainCoreConfig
from src.training.train_dqn import TrainConfig, train
from src.training.train_ppo import TrainPPOConfig
from src.training.train_reinforce import TrainReinforceConfig


def _read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


@pytest.mark.correctness
@pytest.mark.parametrize(
    "cfg",
    (
        TrainConfig(),
        TrainCoreConfig(),
        TrainReinforceConfig(),
        TrainA2CConfig(),
        TrainPPOConfig(),
    ),
)
def test_default_validation_and_test_seed_blocks_are_disjoint(cfg) -> None:
    """Model-selection paths and final-report paths must never overlap."""
    test_episodes = (
        cfg.test_episodes
        if hasattr(cfg, "test_episodes")
        else cfg.final_eval_episodes
    )
    validation_start = cfg.eval_seed0
    validation_end = validation_start + cfg.eval_episodes
    test_start = cfg.test_seed0
    test_end = test_start + test_episodes

    assert validation_end <= test_start or test_end <= validation_start


@pytest.mark.correctness
@pytest.mark.parametrize(
    ("algorithm", "cfg"),
    (
        ("reinforce_core", TrainReinforceConfig()),
        ("a2c_core", TrainA2CConfig()),
        ("ppo_core", TrainPPOConfig()),
    ),
)
def test_pg_training_seed_exclusion_is_recorded(
    tmp_path, algorithm, cfg
) -> None:
    """The train/evaluation separation rule must be visible in provenance."""
    assert cfg.exclude_evaluation_seed_blocks_from_training is True
    run = create_run(
        algorithm,
        seed=cfg.seed,
        training_config=cfg,
        environment_config=cfg.core,
        root=tmp_path,
        run_id="seed-policy",
    )

    config = _read_json(run.config_path)
    assert config["training"][
        "exclude_evaluation_seed_blocks_from_training"
    ] is True


@pytest.mark.correctness
def test_game_dqn_writes_and_finalizes_run_artifacts(tmp_path) -> None:
    """Even a no-learning run records its experiment contract."""
    cfg = TrainConfig(
        total_steps=1,
        warmup_steps=2,
        target_sync=10,
        log_every=1,
        eval_every=10,
        test_episodes=2,
        artifact_root=str(tmp_path),
        run_id="smoke",
        seed=7,
    )

    train(cfg)

    run_directory = tmp_path / "dqn_game" / "seed_007" / "smoke"
    assert (run_directory / "best.pt").is_file()
    assert (run_directory / "last.pt").is_file()
    assert (run_directory / "aurix_config.json").is_file()

    config = _read_json(run_directory / "config.json")
    manifest = _read_json(run_directory / "manifest.json")
    assert config["training"]["test_seed0"] == 100_000
    assert config["environment"]["t_max"] == 200
    assert manifest["status"] == "completed"
    assert manifest["summary"]["environment_sidecar"] == (
        "aurix_config.json"
    )
    assert manifest["summary"]["best_validation_nw_median"] is None

    records = [
        json.loads(line)
        for line in (run_directory / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["split"] for record in records] == ["train", "test"]
    action_counts = records[-1]["metrics"]["action_counts"]
    assert all(isinstance(action, str) for action in action_counts)
