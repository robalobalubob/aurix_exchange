"""Tests for deterministic, protocol-aware multi-seed aggregation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.aggregate_runs import (
    aggregate_experiment,
    summarize_values,
    write_results,
)
from src.training.artifacts import create_run, file_sha256, stable_hash


ENVIRONMENT = {
    "horizon": 200,
    "gamma": 1.0,
    "obs_dim": 3,
}
BASE_TRAINING = {
    "total_updates": 20,
    "learning_rate": 0.001,
    "eval_seed0": 10_000,
    "eval_episodes": 50,
    "test_seed0": 100_000,
    "final_eval_episodes": 100,
}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_run(
    root: Path,
    *,
    algorithm: str,
    seed: int,
    regret: float,
    environment=None,
    training_updates=None,
    dp_hash: str = "dp-canonical",
    metric_n_episodes: int | None = None,
    selected_validation_step: int | None = 10,
):
    artifact_root = (root / "runs").resolve()
    run_id = "run"
    training = {
        **BASE_TRAINING,
        "seed": seed,
        "run_id": run_id,
        "artifact_root": str(artifact_root),
        "dp_cache_dir": str(root / "cache"),
    }
    if training_updates:
        training.update(training_updates)
    run = create_run(
        algorithm,
        seed=seed,
        training_config=training,
        environment_config=environment or ENVIRONMENT,
        root=artifact_root,
        run_id=run_id,
        dependency_names=(),
        metadata={"dp_config_hash": dp_hash},
    )
    metrics = {
        "n_episodes": (
            training["final_eval_episodes"]
            if metric_n_episodes is None
            else metric_n_episodes
        ),
        "return_mean": 1.0 - regret,
        "return_ci": 0.02,
        "regret": regret,
        "paired_regret": regret + 0.01,
        "paired_ci": 0.01,
        "agreement_rate": 0.75,
    }
    run.best_checkpoint_path.write_bytes(
        f"best:{algorithm}:{seed}".encode("utf-8")
    )
    run.last_checkpoint_path.write_bytes(
        f"last:{algorithm}:{seed}".encode("utf-8")
    )
    selected_checkpoint = {
        "path": run.best_checkpoint_path.name,
        "sha256": file_sha256(run.best_checkpoint_path),
        "selected_validation_step": selected_validation_step,
    }
    if selected_validation_step is not None:
        run.append_metrics(
            step=selected_validation_step,
            split="validation",
            metrics={
                "regret": regret + 0.1,
                "selected_as_best": True,
            },
            metadata={"selected_checkpoint": selected_checkpoint},
        )
    run.append_metrics(
        step=20,
        split="test",
        metrics=metrics,
        metadata={"selected_checkpoint": selected_checkpoint},
    )
    run.finish(
        summary={
            "test": metrics,
            "dp_config_hash": dp_hash,
            "selected_checkpoint": selected_checkpoint,
        }
    )
    return {
        "attempt": 1,
        "run_id": run_id,
        "status": "completed",
        "run_directory": str(run.run_directory.resolve()),
        "manifest_path": str(run.manifest_path.resolve()),
        "error": None,
    }


def _experiment(*, algorithms, seeds, trials):
    comparison_config = dict(BASE_TRAINING)
    return {
        "schema_version": 1,
        "experiment_id": "m1-test",
        "milestone": "M1",
        "track": "exact_ou_benchmark",
        "profile": "smoke",
        "canonical_protocol": False,
        "scientific_protocol": False,
        "source": {
            "scientific_clean": False,
            "content_sha256": "test-source-hash",
            "git_commit": None,
        },
        "algorithms": algorithms,
        "seeds": seeds,
        "protocol": {
            "algorithms": {
                algorithm: {
                    "comparison_config": comparison_config,
                    "comparison_config_sha256": stable_hash(
                        comparison_config
                    ),
                    "environment_sha256": stable_hash(ENVIRONMENT),
                }
                for algorithm in algorithms
            }
        },
        "trials": trials,
    }


def _completed_trial(algorithm, seed, attempt):
    return {
        "algorithm": algorithm,
        "seed": seed,
        "status": "completed",
        "attempts": [attempt],
    }


@pytest.mark.correctness
def test_summarize_values_uses_sample_statistics() -> None:
    summary = summarize_values([0.4, 0.1, 0.2])

    assert summary == {
        "n": 3,
        "median": 0.2,
        "min": 0.1,
        "max": 0.4,
        "mean": pytest.approx(0.23333333333333334),
        "sample_sd": pytest.approx(0.1527525231651947),
        "raw_values": [0.1, 0.2, 0.4],
    }


@pytest.mark.correctness
def test_aggregation_is_deterministic_under_input_permutation(
    tmp_path,
) -> None:
    attempts = {
        seed: _make_run(
            tmp_path,
            algorithm="dqn_core",
            seed=seed,
            regret=regret,
        )
        for seed, regret in ((0, 0.1), (1, 0.2), (2, 0.4))
    }
    trials = [
        _completed_trial("dqn_core", seed, attempts[seed])
        for seed in (0, 1, 2)
    ]
    first_path = tmp_path / "experiment.json"
    _write_json(
        first_path,
        _experiment(
            algorithms=["dqn_core"], seeds=[0, 1, 2], trials=trials
        ),
    )
    first = aggregate_experiment(first_path)

    _write_json(
        first_path,
        _experiment(
            algorithms=["dqn_core"],
            seeds=[2, 0, 1],
            trials=list(reversed(trials)),
        ),
    )
    second = aggregate_experiment(first_path)

    assert first == second
    headline = first["algorithms"][0]["regret_across_training_seeds"]
    assert headline["median"] == pytest.approx(0.2)
    assert headline["min"] == pytest.approx(0.1)
    assert headline["max"] == pytest.approx(0.4)
    assert headline["sample_sd"] == pytest.approx(0.1527525231651947)
    assert [
        row["seed"] for row in first["algorithms"][0]["raw_seed_results"]
    ] == [0, 1, 2]


@pytest.mark.correctness
def test_duplicate_algorithm_seed_trial_is_rejected(tmp_path) -> None:
    attempt = _make_run(
        tmp_path, algorithm="dqn_core", seed=0, regret=0.1
    )
    trial = _completed_trial("dqn_core", 0, attempt)
    path = tmp_path / "experiment.json"
    _write_json(
        path,
        _experiment(
            algorithms=["dqn_core"], seeds=[0], trials=[trial, trial]
        ),
    )

    with pytest.raises(ValueError, match="duplicate trial"):
        aggregate_experiment(path)


@pytest.mark.correctness
def test_failed_and_missing_trials_remain_visible_but_are_excluded(
    tmp_path,
) -> None:
    completed = _make_run(
        tmp_path, algorithm="dqn_core", seed=0, regret=0.1
    )
    failed = {
        "algorithm": "dqn_core",
        "seed": 1,
        "status": "failed",
        # An incomplete trial is deliberately not dereferenced.
        "attempts": [
            {
                "status": "failed",
                "manifest_path": "does-not-exist/manifest.json",
                "run_directory": "does-not-exist",
                "error": "simulated worker failure",
            }
        ],
    }
    path = tmp_path / "experiment.json"
    _write_json(
        path,
        _experiment(
            algorithms=["dqn_core"],
            seeds=[0, 1, 2],
            trials=[_completed_trial("dqn_core", 0, completed), failed],
        ),
    )

    results = aggregate_experiment(path)

    assert results["trial_counts"] == {
        "planned": 3,
        "requested": 3,
        "completed": 1,
        "failed": 1,
        "incomplete": 1,
        "included": 1,
        "excluded": 2,
    }
    assert [trial["status"] for trial in results["trials"]] == [
        "completed",
        "failed",
        "missing",
    ]
    assert results["algorithms"][0]["excluded_seeds"] == [1, 2]
    assert results["trials"][1]["last_attempt_error"] == (
        "simulated worker failure"
    )
    assert results["trials"][2]["last_attempt_error"] is None


@pytest.mark.correctness
@pytest.mark.parametrize(
    ("second_run_changes", "error"),
    (
        (
            {"environment": {**ENVIRONMENT, "horizon": 201}},
            "environment hash mismatch",
        ),
        (
            {"training_updates": {"learning_rate": 0.002}},
            "normalized training config mismatch",
        ),
        ({"dp_hash": "different-dp"}, "DP hash mismatch"),
        (
            {"training_updates": {"eval_seed0": 12_000}},
            "normalized training config mismatch",
        ),
        (
            {"training_updates": {"test_seed0": 200_000}},
            "normalized training config mismatch",
        ),
    ),
)
def test_mismatched_completed_seeds_are_rejected(
    tmp_path, second_run_changes, error
) -> None:
    first = _make_run(
        tmp_path, algorithm="dqn_core", seed=0, regret=0.1
    )
    second = _make_run(
        tmp_path,
        algorithm="dqn_core",
        seed=1,
        regret=0.2,
        **second_run_changes,
    )
    path = tmp_path / "experiment.json"
    _write_json(
        path,
        _experiment(
            algorithms=["dqn_core"],
            seeds=[0, 1],
            trials=[
                _completed_trial("dqn_core", 0, first),
                _completed_trial("dqn_core", 1, second),
            ],
        ),
    )

    with pytest.raises(ValueError, match=error):
        aggregate_experiment(path)


@pytest.mark.correctness
def test_test_jsonl_must_match_manifest_summary(tmp_path) -> None:
    attempt = _make_run(
        tmp_path, algorithm="dqn_core", seed=0, regret=0.1
    )
    manifest_path = Path(attempt["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary"]["test"]["regret"] = 99.0
    _write_json(manifest_path, manifest)
    path = tmp_path / "experiment.json"
    _write_json(
        path,
        _experiment(
            algorithms=["dqn_core"],
            seeds=[0],
            trials=[_completed_trial("dqn_core", 0, attempt)],
        ),
    )

    with pytest.raises(ValueError, match="do not match"):
        aggregate_experiment(path)


@pytest.mark.correctness
def test_selected_checkpoint_bytes_are_bound_to_test_record(tmp_path) -> None:
    attempt = _make_run(
        tmp_path, algorithm="dqn_core", seed=0, regret=0.1
    )
    best_path = Path(attempt["run_directory"]) / "best.pt"
    best_path.write_bytes(b"different checkpoint bytes")
    path = tmp_path / "experiment.json"
    _write_json(
        path,
        _experiment(
            algorithms=["dqn_core"],
            seeds=[0],
            trials=[_completed_trial("dqn_core", 0, attempt)],
        ),
    )

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        aggregate_experiment(path)


@pytest.mark.correctness
def test_selected_validation_record_must_precede_test(tmp_path) -> None:
    attempt = _make_run(
        tmp_path, algorithm="dqn_core", seed=0, regret=0.1
    )
    metrics_path = Path(attempt["run_directory"]) / "metrics.jsonl"
    records = metrics_path.read_text(encoding="utf-8").splitlines()
    metrics_path.write_text(
        "\n".join(reversed(records)) + "\n",
        encoding="utf-8",
    )
    path = tmp_path / "experiment.json"
    _write_json(
        path,
        _experiment(
            algorithms=["dqn_core"],
            seeds=[0],
            trials=[_completed_trial("dqn_core", 0, attempt)],
        ),
    )

    with pytest.raises(ValueError, match="must precede"):
        aggregate_experiment(path)


@pytest.mark.correctness
def test_validation_and_test_checkpoint_references_must_match(
    tmp_path,
) -> None:
    attempt = _make_run(
        tmp_path, algorithm="dqn_core", seed=0, regret=0.1
    )
    metrics_path = Path(attempt["run_directory"]) / "metrics.jsonl"
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["metadata"]["selected_checkpoint"]["sha256"] = "0" * 64
    metrics_path.write_text(
        "\n".join(
            json.dumps(record, allow_nan=False, sort_keys=True)
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )
    path = tmp_path / "experiment.json"
    _write_json(
        path,
        _experiment(
            algorithms=["dqn_core"],
            seeds=[0],
            trials=[_completed_trial("dqn_core", 0, attempt)],
        ),
    )

    with pytest.raises(ValueError, match="validation checkpoint and held-out"):
        aggregate_experiment(path)


@pytest.mark.correctness
def test_game_algorithm_is_rejected(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    _write_json(
        path,
        _experiment(algorithms=["dqn_game"], seeds=[0], trials=[]),
    )

    with pytest.raises(ValueError, match="unsupported exact-OU algorithm"):
        aggregate_experiment(path)


@pytest.mark.correctness
def test_write_results_refreshes_deterministic_derived_views(tmp_path) -> None:
    attempt = _make_run(
        tmp_path, algorithm="dqn_core", seed=0, regret=0.1
    )
    path = tmp_path / "experiment.json"
    _write_json(
        path,
        _experiment(
            algorithms=["dqn_core"],
            seeds=[0],
            trials=[_completed_trial("dqn_core", 0, attempt)],
        ),
    )

    paths = write_results(path)
    before = {name: output.read_bytes() for name, output in paths.items()}
    refreshed = write_results(path)

    assert before == {
        name: output.read_bytes() for name, output in refreshed.items()
    }
    assert "return_ci95_half_width_rollouts" in paths["csv"].read_text(
        encoding="utf-8"
    )
    report = paths["report"].read_text(encoding="utf-8")
    assert "No population confidence interval or p-value" in report
    assert "rollouts" in report
    assert attempt["run_directory"] in report
    assert "| 10 |" in report


@pytest.mark.correctness
@pytest.mark.parametrize(
    ("training_updates", "attempt_update", "error"),
    (
        ({"seed": 99}, {}, "configured training seed"),
        ({"run_id": "wrong"}, {}, "configured training run_id"),
        (
            {"artifact_root": "wrong-root"},
            {},
            "configured artifact_root",
        ),
        ({}, {"run_id": "wrong"}, "run-directory name"),
    ),
)
def test_child_run_identity_must_match_trial_and_configuration(
    tmp_path, training_updates, attempt_update, error
) -> None:
    attempt = _make_run(
        tmp_path,
        algorithm="dqn_core",
        seed=0,
        regret=0.1,
        training_updates=training_updates,
    )
    attempt.update(attempt_update)
    path = tmp_path / "experiment.json"
    _write_json(
        path,
        _experiment(
            algorithms=["dqn_core"],
            seeds=[0],
            trials=[_completed_trial("dqn_core", 0, attempt)],
        ),
    )

    with pytest.raises(ValueError, match=error):
        aggregate_experiment(path)


@pytest.mark.correctness
def test_test_episode_count_must_match_declared_protocol(tmp_path) -> None:
    attempt = _make_run(
        tmp_path,
        algorithm="dqn_core",
        seed=0,
        regret=0.1,
        metric_n_episodes=99,
    )
    path = tmp_path / "experiment.json"
    _write_json(
        path,
        _experiment(
            algorithms=["dqn_core"],
            seeds=[0],
            trials=[_completed_trial("dqn_core", 0, attempt)],
        ),
    )

    with pytest.raises(ValueError, match="n_episodes"):
        aggregate_experiment(path)


@pytest.mark.correctness
def test_validation_and_test_seed_intervals_must_be_disjoint(tmp_path) -> None:
    attempt = _make_run(
        tmp_path,
        algorithm="dqn_core",
        seed=0,
        regret=0.1,
        training_updates={
            "eval_seed0": 100,
            "eval_episodes": 50,
            "test_seed0": 120,
            "final_eval_episodes": 100,
        },
    )
    path = tmp_path / "experiment.json"
    _write_json(
        path,
        _experiment(
            algorithms=["dqn_core"],
            seeds=[0],
            trials=[_completed_trial("dqn_core", 0, attempt)],
        ),
    )

    with pytest.raises(ValueError, match="intervals must be disjoint"):
        aggregate_experiment(path)


@pytest.mark.correctness
@pytest.mark.parametrize("identity", ("commit", "dirty", "runtime"))
def test_completed_runs_must_share_source_and_runtime_identity(
    tmp_path, identity
) -> None:
    first = _make_run(
        tmp_path, algorithm="dqn_core", seed=0, regret=0.1
    )
    second = _make_run(
        tmp_path, algorithm="dqn_core", seed=1, regret=0.2
    )
    second_manifest_path = Path(second["manifest_path"])
    manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
    if identity == "commit":
        manifest["source"]["git_commit"] = "different-commit"
    elif identity == "dirty":
        manifest["source"]["git_dirty"] = not bool(
            manifest["source"]["git_dirty"]
        )
    else:
        manifest["runtime"]["dependencies"]["torch"] = "different-version"
    _write_json(second_manifest_path, manifest)
    path = tmp_path / "experiment.json"
    _write_json(
        path,
        _experiment(
            algorithms=["dqn_core"],
            seeds=[0, 1],
            trials=[
                _completed_trial("dqn_core", 0, first),
                _completed_trial("dqn_core", 1, second),
            ],
        ),
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        aggregate_experiment(path)


@pytest.mark.correctness
def test_experiment_commit_must_match_child_and_identity_is_reported(
    tmp_path,
) -> None:
    attempt = _make_run(
        tmp_path, algorithm="dqn_core", seed=0, regret=0.1
    )
    manifest = json.loads(
        Path(attempt["manifest_path"]).read_text(encoding="utf-8")
    )
    path = tmp_path / "experiment.json"
    experiment = _experiment(
        algorithms=["dqn_core"],
        seeds=[0],
        trials=[_completed_trial("dqn_core", 0, attempt)],
    )
    experiment["source"]["git_commit"] = manifest["source"]["git_commit"]
    _write_json(path, experiment)

    results = aggregate_experiment(path)

    assert results["protocol"]["source"] == manifest["source"]
    assert results["protocol"]["runtime"]["details"] == manifest["runtime"]

    experiment["source"]["git_commit"] = "not-the-child-commit"
    _write_json(path, experiment)
    with pytest.raises(ValueError, match="does not match experiment source"):
        aggregate_experiment(path)


@pytest.mark.correctness
@pytest.mark.parametrize(
    ("profile", "algorithms", "seeds", "error"),
    (
        (
            "smoke",
            ["dqn_core", "reinforce_core", "a2c_core", "ppo_core"],
            [0, 1, 2, 3, 4],
            "full profile",
        ),
        ("full", ["dqn_core"], [0, 1, 2, 3, 4], "all four canonical"),
        (
            "full",
            ["dqn_core", "reinforce_core", "a2c_core", "ppo_core"],
            [0, 1, 2],
            "canonical seeds",
        ),
    ),
)
def test_scientific_protocol_requires_canonical_full_cohort(
    tmp_path, profile, algorithms, seeds, error
) -> None:
    experiment = _experiment(
        algorithms=algorithms,
        seeds=seeds,
        trials=[],
    )
    experiment["profile"] = profile
    experiment["scientific_protocol"] = True
    path = tmp_path / "experiment.json"
    _write_json(path, experiment)

    with pytest.raises(ValueError, match=error):
        aggregate_experiment(path)


@pytest.mark.correctness
@pytest.mark.parametrize(
    ("planned_key", "planned_value", "error"),
    (
        (
            "comparison_config_sha256",
            "wrong-training-hash",
            "planned protocol",
        ),
        ("environment_sha256", "wrong-environment-hash", "planned protocol"),
    ),
)
def test_completed_run_must_match_planned_protocol_hashes(
    tmp_path, planned_key, planned_value, error
) -> None:
    attempt = _make_run(
        tmp_path, algorithm="dqn_core", seed=0, regret=0.1
    )
    path = tmp_path / "experiment.json"
    experiment = _experiment(
        algorithms=["dqn_core"],
        seeds=[0],
        trials=[_completed_trial("dqn_core", 0, attempt)],
    )
    experiment["protocol"] = {
        "algorithms": {"dqn_core": {planned_key: planned_value}}
    }
    _write_json(path, experiment)

    with pytest.raises(ValueError, match=error):
        aggregate_experiment(path)


@pytest.mark.correctness
def test_matching_planned_protocol_hashes_are_accepted(tmp_path) -> None:
    attempt = _make_run(
        tmp_path, algorithm="dqn_core", seed=0, regret=0.1
    )
    manifest = json.loads(
        Path(attempt["manifest_path"]).read_text(encoding="utf-8")
    )
    config_path = Path(attempt["run_directory"]) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalized_training = {
        key: value
        for key, value in config["training"].items()
        if key not in {"seed", "run_id", "artifact_root", "dp_cache_dir"}
    }
    experiment = _experiment(
        algorithms=["dqn_core"],
        seeds=[0],
        trials=[_completed_trial("dqn_core", 0, attempt)],
    )
    experiment["protocol"] = {
        "algorithms": {
            "dqn_core": {
                "comparison_config_sha256": stable_hash(
                    normalized_training
                ),
                "environment_sha256": manifest["configuration"][
                    "environment_sha256"
                ],
            }
        }
    }
    path = tmp_path / "experiment.json"
    _write_json(path, experiment)

    results = aggregate_experiment(path)

    assert results["trial_counts"]["included"] == 1


@pytest.mark.correctness
def test_experiment_schema_and_protocol_are_required(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    experiment = _experiment(
        algorithms=["dqn_core"], seeds=[0], trials=[]
    )

    experiment.pop("schema_version")
    _write_json(path, experiment)
    with pytest.raises(ValueError, match="schema version"):
        aggregate_experiment(path)

    experiment["schema_version"] = 1
    experiment.pop("protocol")
    _write_json(path, experiment)
    with pytest.raises(ValueError, match="experiment protocol"):
        aggregate_experiment(path)


@pytest.mark.correctness
def test_planned_protocol_fails_closed_on_missing_or_tampered_data(
    tmp_path,
) -> None:
    path = tmp_path / "experiment.json"
    experiment = _experiment(
        algorithms=["dqn_core"], seeds=[0], trials=[]
    )

    experiment["protocol"]["algorithms"] = {}
    _write_json(path, experiment)
    with pytest.raises(ValueError, match="exactly match"):
        aggregate_experiment(path)

    experiment = _experiment(
        algorithms=["dqn_core"], seeds=[0], trials=[]
    )
    planned = experiment["protocol"]["algorithms"]["dqn_core"]
    planned.pop("environment_sha256")
    _write_json(path, experiment)
    with pytest.raises(ValueError, match="environment_sha256"):
        aggregate_experiment(path)

    experiment = _experiment(
        algorithms=["dqn_core"], seeds=[0], trials=[]
    )
    planned = experiment["protocol"]["algorithms"]["dqn_core"]
    planned["comparison_config"]["learning_rate"] = 999.0
    _write_json(path, experiment)
    with pytest.raises(ValueError, match="self-consistent"):
        aggregate_experiment(path)


@pytest.mark.correctness
def test_canonical_and_scientific_labels_are_validated(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    experiment = _experiment(
        algorithms=["dqn_core"], seeds=[0], trials=[]
    )
    experiment["canonical_protocol"] = True
    _write_json(path, experiment)
    with pytest.raises(ValueError, match="canonical_protocol"):
        aggregate_experiment(path)

    algorithms = [
        "dqn_core",
        "reinforce_core",
        "a2c_core",
        "ppo_core",
    ]
    experiment = _experiment(
        algorithms=algorithms,
        seeds=[0, 1, 2, 3, 4],
        trials=[],
    )
    experiment["profile"] = "full"
    experiment["canonical_protocol"] = True
    experiment["scientific_protocol"] = True
    _write_json(path, experiment)
    with pytest.raises(ValueError, match="clean recorded source"):
        aggregate_experiment(path)

    experiment["source"]["scientific_clean"] = True
    experiment["source"]["git_commit"] = "test-commit"
    _write_json(path, experiment)
    results = aggregate_experiment(path)
    assert results["scientific_protocol"] is True
    assert results["trial_counts"]["incomplete"] == 20


@pytest.mark.correctness
def test_scientific_child_requires_clean_source_attestations(tmp_path) -> None:
    attempt = _make_run(
        tmp_path, algorithm="dqn_core", seed=0, regret=0.1
    )
    manifest = json.loads(
        Path(attempt["manifest_path"]).read_text(encoding="utf-8")
    )
    algorithms = [
        "dqn_core",
        "reinforce_core",
        "a2c_core",
        "ppo_core",
    ]
    experiment = _experiment(
        algorithms=algorithms,
        seeds=[0, 1, 2, 3, 4],
        trials=[_completed_trial("dqn_core", 0, attempt)],
    )
    experiment["profile"] = "full"
    experiment["canonical_protocol"] = True
    experiment["scientific_protocol"] = True
    experiment["source"] = {
        "scientific_clean": True,
        "content_sha256": "planned-source-content",
        "git_commit": manifest["source"]["git_commit"],
    }
    source_attestation = {
        "scientific_clean": True,
        "content_sha256": "planned-source-content",
        "git_commit": manifest["source"]["git_commit"],
    }
    attempt["source_before"] = dict(source_attestation)
    attempt["source_after"] = dict(source_attestation)
    path = tmp_path / "experiment.json"
    _write_json(path, experiment)

    results = aggregate_experiment(path)

    assert results["scientific_protocol"] is True
    assert results["trial_counts"]["completed"] == 1

    attempt["source_after"]["scientific_clean"] = False
    _write_json(path, experiment)
    with pytest.raises(ValueError, match="must attest clean"):
        aggregate_experiment(path)
