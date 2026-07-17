"""Focused tests for reproducible experiment artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest

from src.training.artifacts import create_run, file_sha256, stable_hash


@dataclass(frozen=True)
class _EnvironmentConfig:
    horizon: int = 200
    fee: float = 0.01


@dataclass(frozen=True)
class _TrainingConfig:
    total_steps: int = 1_000
    learning_rate: float = 5.0e-4


def _read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


@pytest.mark.correctness
def test_create_run_snapshots_provenance_and_expected_paths(tmp_path):
    run = create_run(
        "dqn_core",
        seed=7,
        training_config=_TrainingConfig(),
        environment_config=_EnvironmentConfig(),
        root=tmp_path,
        run_id="trial_001",
        dependency_names=("pytest", "package-that-does-not-exist-aurix"),
        metadata={"observation_schema": "ou_core_v1"},
    )

    assert run.run_directory == (
        tmp_path / "dqn_core" / "seed_007" / "trial_001"
    )
    assert run.config_path.is_file()
    assert run.metrics_path.read_text(encoding="utf-8") == ""
    assert run.best_checkpoint_path.name == "best.pt"
    assert run.last_checkpoint_path.name == "last.pt"

    config = _read_json(run.config_path)
    manifest = run.read_manifest()
    assert config["training"]["total_steps"] == 1_000
    assert config["environment"] == {"fee": 0.01, "horizon": 200}
    assert manifest["schema_version"] == 1
    assert manifest["algorithm"] == "dqn_core"
    assert manifest["seed"] == 7
    assert manifest["status"] == "running"
    assert manifest["configuration"]["path"] == "config.json"
    assert manifest["configuration"]["sha256"] == stable_hash(config)
    assert manifest["configuration"]["environment_sha256"] == stable_hash(
        config["environment"]
    )
    assert manifest["artifacts"] == {
        "best_checkpoint": "best.pt",
        "last_checkpoint": "last.pt",
        "metrics": "metrics.jsonl",
    }
    assert manifest["metadata"]["observation_schema"] == "ou_core_v1"
    assert manifest["runtime"]["dependencies"]["pytest"] is not None
    assert (
        manifest["runtime"]["dependencies"][
            "package-that-does-not-exist-aurix"
        ]
        is None
    )
    assert "git_commit" in manifest["source"]
    assert "git_dirty" in manifest["source"]


@pytest.mark.correctness
def test_explicit_run_id_is_exclusive_and_does_not_reuse_directory(tmp_path):
    arguments = {
        "seed": 0,
        "training_config": {},
        "environment_config": {},
        "root": tmp_path,
        "run_id": "same_run",
        "dependency_names": (),
    }
    first = create_run("ppo", **arguments)
    first.metrics_path.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_run("ppo", **arguments)

    assert first.metrics_path.read_text(encoding="utf-8") == "sentinel\n"


@pytest.mark.regression
def test_automatic_run_ids_are_unique_within_seed(tmp_path):
    arguments = {
        "seed": 3,
        "training_config": {},
        "environment_config": {},
        "root": tmp_path,
        "dependency_names": (),
    }
    first = create_run("reinforce", **arguments)
    second = create_run("reinforce", **arguments)

    assert first.run_directory != second.run_directory
    assert first.run_directory.parent == second.run_directory.parent


@pytest.mark.correctness
def test_metrics_are_append_only_valid_json_lines(tmp_path):
    run = create_run(
        "a2c",
        seed=1,
        training_config={},
        environment_config={},
        root=tmp_path,
        run_id="metrics",
        dependency_names=(),
    )
    timestamp = datetime(2026, 7, 16, 12, 30, tzinfo=timezone.utc)

    run.append_metrics(
        step=10,
        split="train",
        metrics={"loss": np.float32(0.25), "updates": np.int64(4)},
        metadata={"checkpoint": "best.pt"},
        timestamp=timestamp,
    )
    run.append_metrics(
        step=10,
        split="validation",
        metrics={"regret": 0.1},
        timestamp=timestamp,
    )

    records = [
        json.loads(line)
        for line in run.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["split"] for record in records] == ["train", "validation"]
    assert records[0]["metrics"] == {"loss": 0.25, "updates": 4}
    assert records[0]["metadata"] == {"checkpoint": "best.pt"}
    assert records[0]["timestamp_utc"] == "2026-07-16T12:30:00Z"
    assert records[1]["metrics"]["regret"] == pytest.approx(0.1)


@pytest.mark.correctness
def test_invalid_metric_is_rejected_before_any_line_is_written(tmp_path):
    run = create_run(
        "dqn",
        seed=None,
        training_config={},
        environment_config={},
        root=tmp_path,
        run_id="invalid_metric",
        dependency_names=(),
    )

    with pytest.raises(ValueError, match="NaN or infinity"):
        run.append_metrics(
            step=1,
            split="train",
            metrics={"loss": float("nan")},
        )

    assert run.metrics_path.read_text(encoding="utf-8") == ""


@pytest.mark.correctness
def test_finish_atomically_records_status_and_summary(tmp_path):
    run = create_run(
        "ppo",
        seed=11,
        training_config={},
        environment_config={},
        root=tmp_path,
        run_id="finished",
        dependency_names=(),
    )
    timestamp = datetime(2026, 7, 16, 13, 0, tzinfo=timezone.utc)

    run.finish(
        status="completed",
        summary={"best_regret": np.float64(0.078)},
        timestamp=timestamp,
    )

    manifest = run.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["summary"]["best_regret"] == pytest.approx(0.078)
    assert manifest["timestamps"]["finished_at_utc"] == "2026-07-16T13:00:00Z"
    assert not list(run.run_directory.glob("*.tmp"))
    with pytest.raises(RuntimeError, match="only a running experiment"):
        run.finish()


@pytest.mark.correctness
def test_hash_is_stable_across_mapping_order_and_rejects_unordered_data():
    assert stable_hash({"alpha": 1, "beta": [2, 3]}) == stable_hash(
        {"beta": [2, 3], "alpha": 1}
    )
    with pytest.raises(
        TypeError, match="unsupported experiment artifact value"
    ):
        stable_hash({"seeds": {1, 2, 3}})


@pytest.mark.correctness
def test_file_sha256_tracks_exact_bytes(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"first")
    first = file_sha256(checkpoint)
    assert len(first) == 64

    checkpoint.write_bytes(b"second")
    assert file_sha256(checkpoint) != first


@pytest.mark.correctness
@pytest.mark.parametrize("algorithm", ("../escape", "has spaces", ""))
def test_algorithm_cannot_escape_artifact_root(tmp_path, algorithm):
    with pytest.raises(ValueError, match="algorithm"):
        create_run(
            algorithm,
            seed=0,
            training_config={},
            environment_config={},
            root=tmp_path,
            dependency_names=(),
        )
