"""Tests for M1 study planning, isolation, and restart semantics."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from src.training.multiseed import (
    DEFAULT_ALGORITHMS,
    DEFAULT_SEEDS,
    _execution_lock_path,
    _run_logged_subprocess,
    build_experiment,
    create_experiment,
    estimate_environment_steps,
    execute_experiment,
    make_training_config,
    render_plan,
    source_fingerprint,
    source_state,
)


def _make_project(root: Path) -> Path:
    source = root / "src"
    source.mkdir(parents=True)
    (source / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    return root


def _completed_child(
    experiment: dict,
    trial: dict,
    attempt: dict,
) -> int:
    run_directory = Path(attempt["run_directory"])
    run_directory.mkdir(parents=True)
    manifest = {
        "status": "completed",
        "algorithm": trial["algorithm"],
        "seed": trial["seed"],
    }
    Path(attempt["manifest_path"]).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return 0


def _no_results(experiment_path: Path) -> None:
    return None


@pytest.mark.correctness
def test_default_plan_is_predeclared_twenty_run_protocol(tmp_path) -> None:
    source = {
        "scientific_clean": True,
        "content_sha256": "source-hash",
        "git_commit": "commit-hash",
    }
    experiment = build_experiment(
        "m1-core-v1",
        artifact_root=tmp_path / "runs",
        project_root=tmp_path,
        source=source,
    )

    assert experiment["algorithms"] == list(DEFAULT_ALGORITHMS)
    assert experiment["seeds"] == list(DEFAULT_SEEDS)
    assert len(experiment["trials"]) == 20
    assert experiment["canonical_protocol"] is True
    assert experiment["scientific_protocol"] is True
    assert experiment["protocol"]["estimated_environment_steps"] == (
        174_650_000
    )

    per_run = {
        name: experiment["protocol"]["algorithms"][name][
            "environment_steps_per_run"
        ]["total"]
        for name in DEFAULT_ALGORITHMS
    }
    assert per_run == {
        "dqn_core": 5_350_000,
        "reinforce_core": 12_180_000,
        "a2c_core": 12_180_000,
        "ppo_core": 5_220_000,
    }
    protocol = experiment["protocol"]["algorithms"]
    environment_hashes = {
        entry["environment_sha256"] for entry in protocol.values()
    }
    assert len(environment_hashes) == 1
    for entry in protocol.values():
        comparison = entry["comparison_config"]
        assert not {
            "seed",
            "run_id",
            "artifact_root",
            "dp_cache_dir",
        } & comparison.keys()
        assert len(entry["comparison_config_sha256"]) == 64
    assert "174,650,000" in render_plan(experiment)


@pytest.mark.correctness
def test_plan_rejects_invalid_algorithms_and_seeds(tmp_path) -> None:
    arguments = {
        "artifact_root": tmp_path / "runs",
        "project_root": tmp_path,
        "source": {"scientific_clean": False},
    }
    with pytest.raises(ValueError, match="exact-OU algorithms only"):
        build_experiment(
            "bad-game",
            algorithms=("dqn_game",),
            **arguments,
        )
    with pytest.raises(ValueError, match="algorithm names must be unique"):
        build_experiment(
            "duplicate-algorithm",
            algorithms=("ppo_core", "ppo_core"),
            **arguments,
        )
    with pytest.raises(ValueError, match="training seeds must be unique"):
        build_experiment(
            "duplicate-seed",
            seeds=(0, 0),
            **arguments,
        )
    with pytest.raises(ValueError, match="non-negative"):
        build_experiment("negative-seed", seeds=(-1,), **arguments)


@pytest.mark.correctness
@pytest.mark.parametrize("algorithm", DEFAULT_ALGORITHMS)
def test_smoke_configs_preserve_identity_but_reduce_work(
    tmp_path,
    algorithm,
) -> None:
    cfg = make_training_config(
        algorithm,
        seed=7,
        artifact_root=tmp_path / "runs",
        run_id="smoke.a01",
        profile="smoke",
    )

    assert cfg.seed == 7
    assert cfg.run_id == "smoke.a01"
    assert cfg.core.t_max == 8
    assert cfg.eval_seed0 == 10_000
    assert cfg.test_seed0 == 100_000
    assert cfg.eval_episodes == 2
    assert cfg.final_eval_episodes == 2
    assert estimate_environment_steps(algorithm, cfg)["total"] < 1_000


@pytest.mark.correctness
def test_source_fingerprint_is_stable_and_content_sensitive(tmp_path) -> None:
    project = _make_project(tmp_path)
    first = source_fingerprint(project)
    second = source_fingerprint(project)
    assert first == second

    (project / "src" / "example.py").write_text(
        "VALUE = 2\n",
        encoding="utf-8",
    )
    assert source_fingerprint(project) != first


@pytest.mark.correctness
def test_logged_subprocess_combines_and_streams_output(
    tmp_path,
    capsys,
) -> None:
    log_path = tmp_path / "nested" / "attempt.log"
    command = [
        sys.executable,
        "-u",
        "-c",
        (
            "import sys; "
            "print('child stdout', flush=True); "
            "print('child stderr', file=sys.stderr, flush=True)"
        ),
    ]

    return_code = _run_logged_subprocess(
        command,
        cwd=tmp_path,
        log_path=log_path,
    )

    assert return_code == 0
    persisted = log_path.read_text(encoding="utf-8")
    streamed = capsys.readouterr().out
    assert "child stdout" in persisted
    assert "child stderr" in persisted
    assert "child stdout" in streamed
    assert "child stderr" in streamed


@pytest.mark.correctness
def test_execution_lock_blocks_concurrent_resume_and_releases(
    tmp_path,
) -> None:
    project = _make_project(tmp_path / "project")
    experiment_path = create_experiment(
        "locked-study",
        algorithms=("dqn_core",),
        seeds=(0,),
        profile="smoke",
        artifact_root=tmp_path / "runs",
        study_root=tmp_path / "studies",
        project_root=project,
        source=source_state(project),
    )
    lock_path = _execution_lock_path(experiment_path)

    def inspect_lock_and_complete(experiment, trial, attempt):
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid()
        assert owner["experiment_path"] == str(experiment_path.resolve())
        assert owner["hostname"]
        assert owner["started_at_utc"].endswith("Z")
        with pytest.raises(RuntimeError) as collision:
            execute_experiment(
                experiment_path,
                resume=True,
                trial_executor=_completed_child,
                result_writer=_no_results,
            )
        message = str(collision.value)
        assert "already being executed" in message
        assert f"pid={os.getpid()}" in message
        assert f"lock={lock_path}" in message
        return _completed_child(experiment, trial, attempt)

    completed = execute_experiment(
        experiment_path,
        trial_executor=inspect_lock_and_complete,
        result_writer=_no_results,
    )

    assert completed["status"] == "completed"
    assert not lock_path.exists()
    attempt = completed["trials"][0]["attempts"][0]
    assert Path(attempt["log_path"]) == (
        experiment_path.parent
        / "logs"
        / "dqn_core"
        / "seed_000"
        / "attempt_01.log"
    ).resolve()


@pytest.mark.correctness
def test_existing_lock_owner_is_preserved(tmp_path) -> None:
    project = _make_project(tmp_path / "project")
    experiment_path = create_experiment(
        "stale-lock-study",
        algorithms=("dqn_core",),
        seeds=(0,),
        profile="smoke",
        artifact_root=tmp_path / "runs",
        study_root=tmp_path / "studies",
        project_root=project,
        source=source_state(project),
    )
    lock_path = _execution_lock_path(experiment_path)
    existing_owner = {
        "owner_token": "someone-else",
        "pid": 4242,
        "hostname": "other-host",
        "started_at_utc": "2026-07-16T12:00:00Z",
        "command": ["python", "runner.py"],
    }
    lock_path.write_text(
        json.dumps(existing_owner),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as collision:
        execute_experiment(
            experiment_path,
            trial_executor=_completed_child,
            result_writer=_no_results,
        )

    message = str(collision.value)
    assert "pid=4242" in message
    assert "host=other-host" in message
    assert "verifying that its recorded owner" in message
    assert json.loads(lock_path.read_text(encoding="utf-8")) == existing_owner


@pytest.mark.correctness
def test_execute_records_completed_trials_and_resume_skips(tmp_path) -> None:
    project = _make_project(tmp_path / "project")
    experiment_path = create_experiment(
        "smoke-study",
        algorithms=("dqn_core",),
        seeds=(0, 1),
        profile="smoke",
        artifact_root=tmp_path / "runs",
        study_root=tmp_path / "studies",
        project_root=project,
        source=source_state(project),
    )

    completed = execute_experiment(
        experiment_path,
        trial_executor=_completed_child,
        result_writer=_no_results,
    )
    assert completed["status"] == "completed"
    assert [trial["status"] for trial in completed["trials"]] == [
        "completed",
        "completed",
    ]
    assert all(
        trial["attempts"][0]["run_id"] == "smoke-study.a01"
        for trial in completed["trials"]
    )
    for trial in completed["trials"]:
        attempt = trial["attempts"][0]
        assert attempt["source_before"] == attempt["source_after"]
        assert set(attempt["source_before"]) == {
            "content_sha256",
            "git_commit",
            "scientific_clean",
        }

    def fail_if_called(experiment, trial, attempt):
        raise AssertionError("completed trials must not be rerun")

    resumed = execute_experiment(
        experiment_path,
        resume=True,
        trial_executor=fail_if_called,
        result_writer=_no_results,
    )
    assert resumed["status"] == "completed"
    assert all(len(trial["attempts"]) == 1 for trial in resumed["trials"])


@pytest.mark.correctness
def test_failure_remains_visible_and_resume_uses_new_attempt(tmp_path) -> None:
    project = _make_project(tmp_path / "project")
    experiment_path = create_experiment(
        "retry-study",
        algorithms=("ppo_core",),
        seeds=(3,),
        profile="smoke",
        artifact_root=tmp_path / "runs",
        study_root=tmp_path / "studies",
        project_root=project,
        source=source_state(project),
    )

    failed = execute_experiment(
        experiment_path,
        trial_executor=lambda experiment, trial, attempt: 17,
        result_writer=_no_results,
    )
    trial = failed["trials"][0]
    assert failed["status"] == "incomplete"
    assert trial["status"] == "failed"
    assert trial["attempts"][0]["run_id"] == "retry-study.a01"

    resumed = execute_experiment(
        experiment_path,
        resume=True,
        trial_executor=_completed_child,
        result_writer=_no_results,
    )
    trial = resumed["trials"][0]
    assert resumed["status"] == "completed"
    assert [attempt["status"] for attempt in trial["attempts"]] == [
        "failed",
        "completed",
    ]
    assert trial["attempts"][1]["run_id"] == "retry-study.a02"
    assert all(
        attempt["source_before"] == attempt["source_after"]
        for attempt in trial["attempts"]
    )


@pytest.mark.correctness
def test_resume_recovers_completed_child(tmp_path) -> None:
    project = _make_project(tmp_path / "project")
    runs = tmp_path / "runs"
    experiment_path = create_experiment(
        "recover-study",
        algorithms=("dqn_core",),
        seeds=(0,),
        profile="smoke",
        artifact_root=runs,
        study_root=tmp_path / "studies",
        project_root=project,
        source=source_state(project),
    )
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    run_directory = (
        runs
        / "dqn_core"
        / "seed_000"
        / "recover-study.a01"
    ).resolve()
    attempt = {
        "attempt": 1,
        "run_id": "recover-study.a01",
        "status": "running",
        "started_at_utc": experiment["created_at_utc"],
        "finished_at_utc": None,
        "run_directory": str(run_directory),
        "manifest_path": str(run_directory / "manifest.json"),
        "return_code": None,
        "error": None,
    }
    experiment["status"] = "running"
    experiment["trials"][0]["status"] = "running"
    experiment["trials"][0]["attempts"] = [attempt]
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    _completed_child(experiment, experiment["trials"][0], attempt)

    def fail_if_called(experiment, trial, attempt):
        raise AssertionError("completed child should be recovered")

    recovered = execute_experiment(
        experiment_path,
        resume=True,
        trial_executor=fail_if_called,
        result_writer=_no_results,
    )

    assert recovered["status"] == "completed"
    trial = recovered["trials"][0]
    assert trial["status"] == "completed"
    assert len(trial["attempts"]) == 1
    assert trial["attempts"][0]["status"] == "completed"
    assert trial["attempts"][0]["source_after"] == {
        "content_sha256": source_fingerprint(project),
        "git_commit": None,
        "scientific_clean": False,
    }


@pytest.mark.correctness
def test_attempt_attests_source_before_and_after_child(tmp_path) -> None:
    project = _make_project(tmp_path / "project")
    experiment_path = create_experiment(
        "attestation-study",
        algorithms=("dqn_core",),
        seeds=(0,),
        profile="smoke",
        artifact_root=tmp_path / "runs",
        study_root=tmp_path / "studies",
        project_root=project,
        source=source_state(project),
    )

    def mutate_source(experiment, trial, attempt):
        return_code = _completed_child(experiment, trial, attempt)
        (project / "src" / "example.py").write_text(
            "VALUE = 200\n",
            encoding="utf-8",
        )
        return return_code

    changed = execute_experiment(
        experiment_path,
        trial_executor=mutate_source,
        result_writer=_no_results,
    )

    assert changed["status"] == "source_changed"
    attempt = changed["trials"][0]["attempts"][0]
    assert attempt["source_before"]["content_sha256"] != (
        attempt["source_after"]["content_sha256"]
    )
    assert attempt["source_before"]["scientific_clean"] is False
    assert attempt["source_after"]["scientific_clean"] is False


@pytest.mark.correctness
def test_aggregation_failure_stops_and_remains_recoverable(tmp_path) -> None:
    project = _make_project(tmp_path / "project")
    experiment_path = create_experiment(
        "aggregation-failure",
        algorithms=("dqn_core",),
        seeds=(0,),
        profile="smoke",
        artifact_root=tmp_path / "runs",
        study_root=tmp_path / "studies",
        project_root=project,
        source=source_state(project),
    )
    writer_calls = 0

    def fail_after_child(experiment_path):
        nonlocal writer_calls
        writer_calls += 1
        if writer_calls == 2:
            raise ValueError("simulated integrity failure")

    with pytest.raises(RuntimeError, match="evidence could not be validated"):
        execute_experiment(
            experiment_path,
            trial_executor=_completed_child,
            result_writer=fail_after_child,
        )

    failed = json.loads(experiment_path.read_text(encoding="utf-8"))
    assert failed["status"] == "aggregation_failed"
    assert failed["trials"][0]["status"] == "completed"
    assert failed["aggregation_error"]["error_type"] == "ValueError"
    assert failed["aggregation_error"]["message"] == (
        "simulated integrity failure"
    )
    assert len(failed["aggregation_failures"]) == 1

    def fail_if_called(experiment, trial, attempt):
        raise AssertionError("validated completed trial must not be rerun")

    recovered = execute_experiment(
        experiment_path,
        resume=True,
        trial_executor=fail_if_called,
        result_writer=_no_results,
    )
    assert recovered["status"] == "completed"
    assert recovered["aggregation_error"] is None
    assert len(recovered["aggregation_failures"]) == 1


@pytest.mark.correctness
def test_execution_refuses_source_drift_and_dirty_full_study(tmp_path) -> None:
    project = _make_project(tmp_path / "project")
    source = source_state(project)
    smoke_path = create_experiment(
        "source-drift",
        algorithms=("a2c_core",),
        seeds=(0,),
        profile="smoke",
        artifact_root=tmp_path / "runs",
        study_root=tmp_path / "studies",
        project_root=project,
        source=source,
    )
    (project / "src" / "example.py").write_text(
        "VALUE = 99\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="source changed"):
        execute_experiment(
            smoke_path,
            trial_executor=_completed_child,
            result_writer=_no_results,
        )

    full_path = create_experiment(
        "dirty-full",
        algorithms=("dqn_core",),
        seeds=(0,),
        profile="full",
        artifact_root=tmp_path / "runs",
        study_root=tmp_path / "studies",
        project_root=project,
        source=source_state(project),
    )
    with pytest.raises(RuntimeError, match="requires committed"):
        execute_experiment(
            full_path,
            trial_executor=_completed_child,
            result_writer=_no_results,
        )
