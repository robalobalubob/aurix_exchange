"""Validate and aggregate multi-seed exact-OU experiments.

An M1 experiment is an index over immutable M0 run directories.  This module
never trains a model and never edits a child run.  It reads completed child
manifests, verifies that the runs are scientifically comparable, and builds
deterministic JSON, CSV, and Markdown views of the experiment.

There are two different sources of uncertainty in the output:

* ``return_ci`` and ``paired_ci`` are 95% confidence-interval half-widths from
  many evaluation rollouts of one trained policy.
* ``sample_sd`` describes variation across independently trained seeds.

Those quantities answer different questions and must not be conflated.  The
aggregator deliberately computes no population confidence interval or p-value
from the small collection of training seeds.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.training.artifacts import file_sha256


RESULTS_SCHEMA_VERSION = 1
ALLOWED_ALGORITHMS = frozenset(
    {"dqn_core", "reinforce_core", "a2c_core", "ppo_core"}
)
_IGNORED_TRAINING_KEYS = frozenset(
    {"seed", "run_id", "artifact_root", "dp_cache_dir"}
)
_PROTOCOL_KEYS = (
    "eval_seed0",
    "eval_episodes",
    "test_seed0",
    "final_eval_episodes",
)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_seed(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _unique(values: Sequence[Any], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates")


def _resolve_path(value: Any, base: Path, label: str) -> Path:
    text = _require_string(value, label)
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _child_path(run_directory: Path, value: Any, label: str) -> Path:
    text = _require_string(value, label)
    relative = Path(text)
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative to the child run")
    resolved = (run_directory / relative).resolve()
    try:
        resolved.relative_to(run_directory)
    except ValueError as error:
        raise ValueError(f"{label} escapes the child run directory") from error
    return resolved


def _load_run_metrics(
    metrics_path: Path,
) -> tuple[dict[str, Any], int, dict[str, Any], dict[str, Any]]:
    test_records: list[tuple[int, dict[str, Any]]] = []
    selected_validation_step: int | None = None
    selected_validation_line: int | None = None
    selected_validation_checkpoint: dict[str, Any] | None = None
    try:
        with metrics_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON in {metrics_path} line {line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"metric record in {metrics_path} line "
                        f"{line_number} is not an object"
                    )
                if record.get("split") == "test":
                    if record.get("schema_version") != 1:
                        raise ValueError(
                            "unsupported test metric schema version in "
                            f"{metrics_path} line {line_number}"
                        )
                    test_records.append((line_number, record))
                if record.get("split") == "validation":
                    validation_metrics = record.get("metrics")
                    if (
                        isinstance(validation_metrics, dict)
                        and validation_metrics.get("selected_as_best") is True
                    ):
                        step = record.get("step")
                        if (
                            isinstance(step, bool)
                            or not isinstance(step, int)
                            or step < 0
                        ):
                            raise ValueError(
                                "selected validation record has an invalid "
                                f"step in {metrics_path} line {line_number}"
                            )
                        selected_validation_step = step
                        selected_validation_line = line_number
                        validation_metadata = _require_mapping(
                            record.get("metadata"),
                            "selected validation metric metadata in "
                            f"{metrics_path} line {line_number}",
                        )
                        selected_validation_checkpoint = _require_mapping(
                            validation_metadata.get("selected_checkpoint"),
                            "selected validation checkpoint in "
                            f"{metrics_path} line {line_number}",
                        )
    except OSError as error:
        raise ValueError(
            f"cannot read child metrics at {metrics_path}"
        ) from error

    if len(test_records) != 1:
        raise ValueError(
            f"expected exactly one split='test' record in {metrics_path}; "
            f"found {len(test_records)}"
        )
    if (
        selected_validation_step is None
        or selected_validation_line is None
        or selected_validation_checkpoint is None
    ):
        raise ValueError(
            "completed run has no validation record selected as best in "
            f"{metrics_path}"
        )
    test_line, test_record = test_records[0]
    if selected_validation_line >= test_line:
        raise ValueError(
            "selected validation record must precede the held-out test "
            f"record in {metrics_path}"
        )
    test_metadata = _require_mapping(
        test_record.get("metadata"),
        f"test metric metadata in {metrics_path}",
    )
    checkpoint = _require_mapping(
        test_metadata.get("selected_checkpoint"),
        f"selected checkpoint in {metrics_path}",
    )
    metrics = test_record.get("metrics")
    return (
        _validate_test_metrics(metrics, str(metrics_path)),
        selected_validation_step,
        checkpoint,
        selected_validation_checkpoint,
    )


def _validate_test_metrics(value: Any, label: str) -> dict[str, int | float]:
    metrics = _require_mapping(value, f"test metrics in {label}")
    if "regret" not in metrics:
        raise ValueError(f"test metrics in {label} must include regret")

    normalized: dict[str, int | float] = {}
    for key, metric in metrics.items():
        _require_string(key, f"test metric name in {label}")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError(
                f"test metric {key!r} in {label} must be numeric"
            )
        if not math.isfinite(metric):
            raise ValueError(
                f"test metric {key!r} in {label} must be finite"
            )
        normalized[key] = metric
    return normalized


def _validate_checkpoint_reference(
    value: Any,
    *,
    label: str,
    selected_validation_step: int,
) -> dict[str, Any]:
    reference = _require_mapping(value, label)
    path = _require_string(reference.get("path"), f"{label}.path")
    sha256 = _require_string(
        reference.get("sha256"), f"{label}.sha256"
    )
    if len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256.lower()
    ):
        raise ValueError(f"{label}.sha256 must be a full SHA-256 digest")
    step = reference.get("selected_validation_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(
            f"{label}.selected_validation_step must be non-negative"
        )
    if step != selected_validation_step:
        raise ValueError(
            f"{label}.selected_validation_step does not match metrics"
        )
    return {
        "path": path,
        "sha256": sha256.lower(),
        "selected_validation_step": step,
    }


def _normalized_training_config(training: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in training.items()
        if key not in _IGNORED_TRAINING_KEYS
    }


def _protocol_from_training(
    training: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, int]:
    protocol: dict[str, int] = {}
    for key in _PROTOCOL_KEYS:
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label}.{key} must be an integer")
        if value < 0:
            raise ValueError(f"{label}.{key} must be non-negative")
        protocol[key] = value
    if protocol["eval_episodes"] == 0:
        raise ValueError(f"{label}.eval_episodes must be positive")
    if protocol["final_eval_episodes"] == 0:
        raise ValueError(f"{label}.final_eval_episodes must be positive")
    validation_start = protocol["eval_seed0"]
    validation_end = validation_start + protocol["eval_episodes"]
    test_start = protocol["test_seed0"]
    test_end = test_start + protocol["final_eval_episodes"]
    if not (validation_end <= test_start or test_end <= validation_start):
        raise ValueError(
            f"{label} validation and test seed intervals must be disjoint"
        )
    return protocol


def _load_completed_trial(
    *,
    algorithm: str,
    seed: int,
    attempt: Mapping[str, Any],
    experiment_directory: Path,
) -> dict[str, Any]:
    run_directory = _resolve_path(
        attempt.get("run_directory"),
        experiment_directory,
        "completed attempt run_directory",
    )
    manifest_path = _resolve_path(
        attempt.get("manifest_path"),
        experiment_directory,
        "completed attempt manifest_path",
    )
    expected_manifest_path = (run_directory / "manifest.json").resolve()
    if manifest_path != expected_manifest_path:
        raise ValueError(
            "completed attempt manifest_path must identify manifest.json "
            "inside its run_directory"
        )

    attempt_run_id = _require_string(
        attempt.get("run_id"), "completed attempt run_id"
    )
    if run_directory.name != attempt_run_id:
        raise ValueError(
            "completed attempt run_id must equal its run-directory name"
        )

    manifest = _read_json_object(manifest_path, label="child manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"unsupported child manifest schema version: {manifest_path}"
        )
    if manifest.get("status") != "completed":
        raise ValueError(f"child manifest is not completed: {manifest_path}")
    if manifest.get("run_id") != attempt_run_id:
        raise ValueError(
            f"child manifest run_id does not match attempt: {manifest_path}"
        )
    if manifest.get("algorithm") != algorithm:
        raise ValueError(
            f"child manifest algorithm does not match trial {algorithm!r}: "
            f"{manifest_path}"
        )
    if manifest.get("seed") != seed:
        raise ValueError(
            f"child manifest seed does not match trial seed {seed}: "
            f"{manifest_path}"
        )

    configuration = _require_mapping(
        manifest.get("configuration"),
        f"child manifest configuration in {manifest_path}",
    )
    config_path = _child_path(
        run_directory,
        configuration.get("path"),
        "child configuration path",
    )
    config = _read_json_object(config_path, label="child configuration")
    if config.get("schema_version") != 1:
        raise ValueError(
            f"unsupported child configuration schema version: {config_path}"
        )
    training = _require_mapping(
        config.get("training"), f"training configuration in {config_path}"
    )
    environment = _require_mapping(
        config.get("environment"),
        f"environment configuration in {config_path}",
    )
    configured_seed = _require_seed(
        training.get("seed"), f"training seed in {config_path}"
    )
    if configured_seed != seed:
        raise ValueError(
            "configured training seed does not match trial seed in "
            f"{config_path}"
        )
    configured_run_id = _require_string(
        training.get("run_id"), f"training run_id in {config_path}"
    )
    if configured_run_id != attempt_run_id:
        raise ValueError(
            "configured training run_id does not match child run in "
            f"{config_path}"
        )
    configured_artifact_root = _resolve_path(
        training.get("artifact_root"),
        experiment_directory,
        f"training artifact_root in {config_path}",
    )
    if len(run_directory.parents) < 3:
        raise ValueError(
            f"child run directory is too shallow: {run_directory}"
        )
    actual_artifact_root = run_directory.parents[2]
    if configured_artifact_root != actual_artifact_root:
        raise ValueError(
            "configured artifact_root does not match child run in "
            f"{config_path}"
        )

    hash_checks = {
        "sha256": _stable_hash(config),
        "training_sha256": _stable_hash(training),
        "environment_sha256": _stable_hash(environment),
    }
    for key, actual_hash in hash_checks.items():
        declared_hash = configuration.get(key)
        if declared_hash != actual_hash:
            raise ValueError(
                f"child {key} does not match {config_path}: "
                f"declared {declared_hash!r}, calculated {actual_hash!r}"
            )

    artifacts = _require_mapping(
        manifest.get("artifacts"), f"child artifacts in {manifest_path}"
    )
    metrics_path = _child_path(
        run_directory,
        artifacts.get("metrics"),
        "child metrics path",
    )
    (
        metrics,
        selected_validation_step,
        test_checkpoint,
        validation_checkpoint,
    ) = _load_run_metrics(metrics_path)

    summary = _require_mapping(
        manifest.get("summary"), f"child summary in {manifest_path}"
    )
    summary_test = _validate_test_metrics(
        summary.get("test"), f"manifest summary in {manifest_path}"
    )
    if metrics != summary_test:
        raise ValueError(
            "split='test' metrics do not match manifest.summary.test in "
            f"{manifest_path}"
        )
    test_checkpoint = _validate_checkpoint_reference(
        test_checkpoint,
        label=f"test selected checkpoint in {metrics_path}",
        selected_validation_step=selected_validation_step,
    )
    validation_checkpoint = _validate_checkpoint_reference(
        validation_checkpoint,
        label=f"selected validation checkpoint in {metrics_path}",
        selected_validation_step=selected_validation_step,
    )
    if test_checkpoint != validation_checkpoint:
        raise ValueError(
            "selected validation checkpoint and held-out test checkpoint "
            f"disagree in {metrics_path}"
        )
    summary_checkpoint = _validate_checkpoint_reference(
        summary.get("selected_checkpoint"),
        label=f"manifest selected checkpoint in {manifest_path}",
        selected_validation_step=selected_validation_step,
    )
    if test_checkpoint != summary_checkpoint:
        raise ValueError(
            "test record and manifest selected checkpoint disagree in "
            f"{manifest_path}"
        )
    best_checkpoint_path = _child_path(
        run_directory,
        artifacts.get("best_checkpoint"),
        "child best checkpoint path",
    )
    selected_checkpoint_path = _child_path(
        run_directory,
        summary_checkpoint["path"],
        "selected checkpoint path",
    )
    if selected_checkpoint_path != best_checkpoint_path:
        raise ValueError(
            "selected checkpoint must identify the manifest best checkpoint "
            f"in {manifest_path}"
        )
    if not best_checkpoint_path.is_file():
        raise ValueError(
            f"selected checkpoint does not exist: {best_checkpoint_path}"
        )
    actual_checkpoint_sha256 = file_sha256(best_checkpoint_path)
    if actual_checkpoint_sha256 != summary_checkpoint["sha256"]:
        raise ValueError(
            "selected checkpoint SHA-256 does not match the recorded test "
            f"checkpoint in {manifest_path}"
        )

    metadata = _require_mapping(
        manifest.get("metadata"), f"child metadata in {manifest_path}"
    )
    metadata_dp_hash = _require_string(
        metadata.get("dp_config_hash"),
        f"metadata.dp_config_hash in {manifest_path}",
    )
    summary_dp_hash = _require_string(
        summary.get("dp_config_hash"),
        f"summary.dp_config_hash in {manifest_path}",
    )
    if metadata_dp_hash != summary_dp_hash:
        raise ValueError(
            f"DP hashes disagree inside child manifest {manifest_path}"
        )

    normalized_training = _normalized_training_config(training)
    protocol = _protocol_from_training(
        training, label=f"training configuration in {config_path}"
    )
    n_episodes = metrics.get("n_episodes")
    if (
        isinstance(n_episodes, bool)
        or not isinstance(n_episodes, int)
        or n_episodes != protocol["final_eval_episodes"]
    ):
        raise ValueError(
            "test metrics n_episodes must equal configured "
            f"final_eval_episodes in {manifest_path}"
        )

    source = _require_mapping(
        manifest.get("source"), f"child source in {manifest_path}"
    )
    git_commit = source.get("git_commit")
    if git_commit is not None:
        git_commit = _require_string(
            git_commit, f"source.git_commit in {manifest_path}"
        )
    git_dirty = source.get("git_dirty")
    if git_dirty is not None and not isinstance(git_dirty, bool):
        raise ValueError(
            f"source.git_dirty must be boolean or null in {manifest_path}"
        )
    runtime = _require_mapping(
        manifest.get("runtime"), f"child runtime in {manifest_path}"
    )
    dependencies = _require_mapping(
        runtime.get("dependencies"),
        f"child runtime dependencies in {manifest_path}",
    )
    return {
        "algorithm": algorithm,
        "seed": seed,
        "run_directory": str(attempt["run_directory"]),
        "manifest_path": str(attempt["manifest_path"]),
        "environment_hash": configuration["environment_sha256"],
        "dp_hash": metadata_dp_hash,
        "training_config_hash": _stable_hash(normalized_training),
        "protocol": protocol,
        "test_metrics": metrics,
        "selected_validation_step": selected_validation_step,
        "selected_checkpoint": summary_checkpoint,
        "source": {
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
        "runtime": runtime,
        "runtime_hash": _stable_hash(runtime),
        "dependency_hash": _stable_hash(dependencies),
    }


def summarize_values(values: Iterable[int | float]) -> dict[str, Any]:
    """Summarize variation across independent training seeds.

    Values are sorted so the result is independent of trial ordering.  The
    standard deviation is the sample statistic (``n - 1`` denominator); it is
    undefined for one seed and represented as ``None``.  No population CI or
    hypothesis test is inferred from this small sample.
    """
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("summary values must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("summary values must be finite")
        normalized.append(number)
    if not normalized:
        raise ValueError("at least one value is required")

    normalized.sort()
    return {
        "n": len(normalized),
        "median": statistics.median(normalized),
        "min": normalized[0],
        "max": normalized[-1],
        "mean": statistics.fmean(normalized),
        "sample_sd": (
            statistics.stdev(normalized) if len(normalized) >= 2 else None
        ),
        "raw_values": normalized,
    }


def _completed_attempt(
    trial: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    attempts = _require_sequence(trial.get("attempts"), f"{label}.attempts")
    completed = []
    for index, raw_attempt in enumerate(attempts):
        attempt = _require_mapping(raw_attempt, f"{label}.attempts[{index}]")
        if attempt.get("status") == "completed":
            completed.append(attempt)
    if len(completed) != 1:
        raise ValueError(
            f"completed {label} must have exactly one completed attempt; "
            f"found {len(completed)}"
        )
    return completed[0]


def _last_attempt_error(
    attempts: Sequence[Any], label: str
) -> str | None:
    if not attempts:
        return None
    last = _require_mapping(attempts[-1], f"{label} last attempt")
    error = last.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError(
            f"{label} last attempt error must be a string or null"
        )
    return error


def _last_attempt_log_path(
    attempts: Sequence[Any], label: str
) -> str | None:
    if not attempts:
        return None
    last = _require_mapping(attempts[-1], f"{label} last attempt")
    log_path = last.get("log_path")
    if log_path is None:
        return None
    return _require_string(log_path, f"{label} last attempt log_path")


def _validate_scientific_attempt_source(
    attempt: Mapping[str, Any],
    experiment_source: Mapping[str, Any],
) -> None:
    expected_content = _require_string(
        experiment_source.get("content_sha256"),
        "scientific experiment source content_sha256",
    )
    expected_commit = _require_string(
        experiment_source.get("git_commit"),
        "scientific experiment source git_commit",
    )
    for phase in ("source_before", "source_after"):
        attestation = _require_mapping(
            attempt.get(phase), f"scientific completed attempt {phase}"
        )
        if attestation.get("content_sha256") != expected_content:
            raise ValueError(
                f"scientific completed attempt {phase} content hash does "
                "not match the experiment source"
            )
        if attestation.get("git_commit") != expected_commit:
            raise ValueError(
                f"scientific completed attempt {phase} commit does not "
                "match the experiment source"
            )
        if attestation.get("scientific_clean") is not True:
            raise ValueError(
                f"scientific completed attempt {phase} must attest clean "
                "relevant source"
            )


def _check_planned_protocol(
    completed: Sequence[Mapping[str, Any]],
    experiment: Mapping[str, Any],
) -> None:
    protocol = _require_mapping(
        experiment.get("protocol"), "experiment protocol"
    )
    planned_algorithms = _require_mapping(
        protocol.get("algorithms"), "experiment protocol algorithms"
    )
    requested_algorithms = set(
        _require_sequence(experiment.get("algorithms"), "algorithms")
    )
    if set(planned_algorithms) != requested_algorithms:
        raise ValueError(
            "experiment protocol algorithms must exactly match requested "
            "algorithms"
        )
    for algorithm, raw_planned in planned_algorithms.items():
        planned = _require_mapping(
            raw_planned, f"planned protocol for {algorithm}"
        )
        expected_training = _require_string(
            planned.get("comparison_config_sha256"),
            f"planned protocol for {algorithm} comparison_config_sha256",
        )
        _require_string(
            planned.get("environment_sha256"),
            f"planned protocol for {algorithm} environment_sha256",
        )
        comparison_config = planned.get("comparison_config")
        if comparison_config is not None:
            comparison_config = _require_mapping(
                comparison_config,
                f"planned protocol for {algorithm} comparison_config",
            )
            if _stable_hash(comparison_config) != expected_training:
                raise ValueError(
                    "planned protocol comparison_config hash is not "
                    f"self-consistent for {algorithm}"
                )

    for child in completed:
        planned = _require_mapping(
            planned_algorithms[child["algorithm"]],
            f"planned protocol for {child['algorithm']}",
        )
        expected_training = planned["comparison_config_sha256"]
        if expected_training != child["training_config_hash"]:
            raise ValueError(
                "completed child normalized training config does not match "
                f"the planned protocol for {child['algorithm']}"
            )
        expected_environment = planned["environment_sha256"]
        if expected_environment != child["environment_hash"]:
            raise ValueError(
                "completed child environment does not match the planned "
                f"protocol for {child['algorithm']}"
            )


def _check_comparability(
    completed: Sequence[Mapping[str, Any]],
    experiment: Mapping[str, Any],
) -> None:
    by_algorithm: dict[str, list[Mapping[str, Any]]] = {}
    for child in completed:
        by_algorithm.setdefault(str(child["algorithm"]), []).append(child)

    for algorithm, children in sorted(by_algorithm.items()):
        for key, description in (
            ("environment_hash", "environment hash"),
            ("dp_hash", "DP hash"),
            ("training_config_hash", "normalized training config"),
        ):
            values = {child[key] for child in children}
            if len(values) != 1:
                raise ValueError(
                    f"{description} mismatch across {algorithm} seeds"
                )
        validation_protocols = {
            (
                child["protocol"]["eval_seed0"],
                child["protocol"]["eval_episodes"],
            )
            for child in children
        }
        if len(validation_protocols) != 1:
            raise ValueError(
                f"validation seed settings mismatch across {algorithm} seeds"
            )

    for key, description in (
        ("source", "source identity"),
        ("runtime_hash", "runtime identity"),
        ("dependency_hash", "dependency identity"),
    ):
        values = {_stable_hash(child[key]) for child in completed}
        if len(values) > 1:
            raise ValueError(f"{description} mismatch across completed runs")

    # Every algorithm is scored on the same held-out rollout block.  Validation
    # episode counts may be algorithm-specific, so those are checked only among
    # seeds of one algorithm above.
    test_protocols = {
        (
            child["protocol"]["test_seed0"],
            child["protocol"]["final_eval_episodes"],
        )
        for child in completed
    }
    if len(test_protocols) > 1:
        raise ValueError(
            "held-out test seed settings mismatch across completed runs"
        )

    environment_hashes = {child["environment_hash"] for child in completed}
    if len(environment_hashes) > 1:
        raise ValueError(
            "environment hash mismatch across exact-OU algorithms"
        )
    dp_hashes = {child["dp_hash"] for child in completed}
    if len(dp_hashes) > 1:
        raise ValueError("DP hash mismatch across exact-OU algorithms")

    experiment_source = experiment.get("source")
    if experiment_source is not None:
        experiment_source = _require_mapping(
            experiment_source, "experiment source"
        )
        planned_commit = experiment_source.get("git_commit")
        if planned_commit is not None:
            planned_commit = _require_string(
                planned_commit, "experiment source git_commit"
            )
            for child in completed:
                if child["source"]["git_commit"] != planned_commit:
                    raise ValueError(
                        "child git commit does not match experiment source"
                    )

    _check_planned_protocol(completed, experiment)


def aggregate_experiment(experiment_path: str | Path) -> dict[str, Any]:
    """Validate an experiment and return deterministic derived results."""
    source_path = Path(experiment_path).resolve()
    experiment = _read_json_object(source_path, label="experiment")
    if experiment.get("schema_version") != 1:
        raise ValueError("unsupported M1 experiment schema version")
    if experiment.get("milestone") != "M1":
        raise ValueError("milestone must be 'M1'")
    experiment_id = _require_string(
        experiment.get("experiment_id"), "experiment_id"
    )
    if experiment.get("track") != "exact_ou_benchmark":
        raise ValueError("track must be 'exact_ou_benchmark'")
    profile = _require_string(experiment.get("profile"), "profile")
    if profile not in {"full", "smoke"}:
        raise ValueError("profile must be 'full' or 'smoke'")
    scientific_protocol = experiment.get("scientific_protocol")
    if not isinstance(scientific_protocol, bool):
        raise ValueError("scientific_protocol must be a boolean")
    canonical_protocol = experiment.get("canonical_protocol")
    if not isinstance(canonical_protocol, bool):
        raise ValueError("canonical_protocol must be a boolean")
    experiment_source = _require_mapping(
        experiment.get("source"), "experiment source"
    )
    scientific_clean = experiment_source.get("scientific_clean")
    if not isinstance(scientific_clean, bool):
        raise ValueError("experiment source scientific_clean must be boolean")
    _require_string(
        experiment_source.get("content_sha256"),
        "experiment source content_sha256",
    )

    algorithms = [
        _require_string(value, "algorithm")
        for value in _require_sequence(
            experiment.get("algorithms"), "algorithms"
        )
    ]
    if not algorithms:
        raise ValueError("algorithms must not be empty")
    _unique(algorithms, "algorithms")
    unsupported = sorted(set(algorithms) - ALLOWED_ALGORITHMS)
    if unsupported:
        raise ValueError(
            "unsupported exact-OU algorithm(s): " + ", ".join(unsupported)
        )

    seeds = [
        _require_seed(value, "seed")
        for value in _require_sequence(experiment.get("seeds"), "seeds")
    ]
    if not seeds:
        raise ValueError("seeds must not be empty")
    _unique(seeds, "seeds")
    canonical_shape = (
        profile == "full"
        and set(algorithms) == ALLOWED_ALGORITHMS
        and sorted(seeds) == list(range(5))
    )
    if canonical_protocol != canonical_shape:
        raise ValueError(
            "canonical_protocol does not match the declared profile, "
            "algorithms, and seeds"
        )
    if scientific_protocol:
        if profile != "full":
            raise ValueError(
                "scientific_protocol=true requires the full profile"
            )
        if set(algorithms) != ALLOWED_ALGORITHMS:
            raise ValueError(
                "scientific_protocol=true requires all four canonical "
                "exact-OU algorithms"
            )
        if sorted(seeds) != list(range(5)):
            raise ValueError(
                "scientific_protocol=true requires canonical seeds 0..4"
            )
    if scientific_protocol and not (
        canonical_protocol and scientific_clean
    ):
        raise ValueError(
            "scientific_protocol=true requires canonical_protocol and clean "
            "recorded source"
        )
    if scientific_protocol:
        _require_string(
            experiment_source.get("git_commit"),
            "scientific experiment source git_commit",
        )

    _check_planned_protocol((), experiment)

    raw_trials = _require_sequence(experiment.get("trials"), "trials")
    indexed_trials: dict[tuple[str, int], dict[str, Any]] = {}
    for index, raw_trial in enumerate(raw_trials):
        label = f"trials[{index}]"
        trial = _require_mapping(raw_trial, label)
        algorithm = _require_string(
            trial.get("algorithm"), f"{label}.algorithm"
        )
        seed = _require_seed(trial.get("seed"), f"{label}.seed")
        if algorithm not in ALLOWED_ALGORITHMS:
            raise ValueError(f"unsupported exact-OU algorithm: {algorithm}")
        if algorithm not in algorithms or seed not in seeds:
            raise ValueError(
                f"{label} ({algorithm}, {seed}) is outside algorithms x seeds"
            )
        key = (algorithm, seed)
        if key in indexed_trials:
            raise ValueError(
                f"duplicate trial for algorithm {algorithm!r}, seed {seed}"
            )
        indexed_trials[key] = trial

    completed_children: list[dict[str, Any]] = []
    result_trials: list[dict[str, Any]] = []
    for algorithm in sorted(algorithms):
        for seed in sorted(seeds):
            key = (algorithm, seed)
            trial = indexed_trials.get(key)
            if trial is None:
                result_trials.append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "status": "missing",
                        "attempt_count": 0,
                        "included": False,
                        "exclusion_reason": "trial is absent from experiment",
                        "last_attempt_error": None,
                    }
                )
                continue

            status = _require_string(
                trial.get("status"), f"trial {algorithm}/{seed} status"
            )
            attempts = _require_sequence(
                trial.get("attempts"),
                f"trial {algorithm}/{seed} attempts",
            )
            base_result = {
                "algorithm": algorithm,
                "seed": seed,
                "status": status,
                "attempt_count": len(attempts),
                "last_attempt_error": _last_attempt_error(
                    attempts, f"trial {algorithm}/{seed}"
                ),
                "last_attempt_log_path": _last_attempt_log_path(
                    attempts, f"trial {algorithm}/{seed}"
                ),
            }
            if status != "completed":
                result_trials.append(
                    {
                        **base_result,
                        "included": False,
                        "exclusion_reason": f"trial status is {status}",
                    }
                )
                continue

            attempt = _completed_attempt(
                trial, f"trial {algorithm}/{seed}"
            )
            if scientific_protocol:
                _validate_scientific_attempt_source(
                    attempt, experiment_source
                )
            child = _load_completed_trial(
                algorithm=algorithm,
                seed=seed,
                attempt=attempt,
                experiment_directory=source_path.parent,
            )
            completed_children.append(child)
            result_trials.append(
                {
                    **base_result,
                    "included": True,
                    "exclusion_reason": None,
                    "run_directory": child["run_directory"],
                    "manifest_path": child["manifest_path"],
                    "test_metrics": child["test_metrics"],
                    "selected_validation_step": child[
                        "selected_validation_step"
                    ],
                    "selected_checkpoint": child["selected_checkpoint"],
                }
            )

    _check_comparability(completed_children, experiment)

    algorithm_results: list[dict[str, Any]] = []
    for algorithm in sorted(algorithms):
        raw_results = [
            {
                "seed": child["seed"],
                "test_metrics": child["test_metrics"],
                "run_directory": child["run_directory"],
                "manifest_path": child["manifest_path"],
                "selected_validation_step": child[
                    "selected_validation_step"
                ],
                "selected_checkpoint": child["selected_checkpoint"],
            }
            for child in sorted(
                (
                    item
                    for item in completed_children
                    if item["algorithm"] == algorithm
                ),
                key=lambda item: item["seed"],
            )
        ]
        completed_seeds = [item["seed"] for item in raw_results]
        regret_summary = (
            summarize_values(
                item["test_metrics"]["regret"] for item in raw_results
            )
            if raw_results
            else None
        )
        algorithm_results.append(
            {
                "algorithm": algorithm,
                "requested_seeds": sorted(seeds),
                "completed_seeds": completed_seeds,
                "excluded_seeds": sorted(set(seeds) - set(completed_seeds)),
                "regret_across_training_seeds": regret_summary,
                "raw_seed_results": raw_results,
            }
        )

    common_test_protocol = None
    common_environment_hash = None
    common_dp_hash = None
    common_source = None
    common_runtime = None
    if completed_children:
        first = completed_children[0]
        common_test_protocol = {
            "test_seed0": first["protocol"]["test_seed0"],
            "final_eval_episodes": first["protocol"][
                "final_eval_episodes"
            ],
        }
        common_environment_hash = first["environment_hash"]
        common_dp_hash = first["dp_hash"]
        common_source = first["source"]
        common_runtime = {
            "sha256": first["runtime_hash"],
            "dependency_sha256": first["dependency_hash"],
            "details": first["runtime"],
        }

    completed_count = len(completed_children)
    failed_count = sum(
        trial["status"] == "failed" for trial in result_trials
    )
    incomplete_count = (
        len(algorithms) * len(seeds) - completed_count - failed_count
    )

    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "track": "exact_ou_benchmark",
        "profile": profile,
        "scientific_protocol": scientific_protocol,
        "requested_algorithms": sorted(algorithms),
        "requested_seeds": sorted(seeds),
        "protocol": {
            "common_test_seed_block": common_test_protocol,
            "environment_hash": common_environment_hash,
            "dp_hash": common_dp_hash,
            "source": common_source,
            "runtime": common_runtime,
        },
        "uncertainty_labels": {
            "return_ci": "95% CI half-width across evaluation rollouts",
            "paired_ci": (
                "95% CI half-width across paired evaluation rollouts"
            ),
            "sample_sd": (
                "sample standard deviation across independent training seeds "
                "(n - 1 denominator)"
            ),
            "population_inference": (
                "No population confidence interval or p-value is computed"
            ),
        },
        "trial_counts": {
            "planned": len(algorithms) * len(seeds),
            "requested": len(algorithms) * len(seeds),
            "completed": completed_count,
            "failed": failed_count,
            "incomplete": incomplete_count,
            "included": completed_count,
            "excluded": len(algorithms) * len(seeds) - completed_count,
        },
        "trials": result_trials,
        "algorithms": algorithm_results,
    }


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _results_csv(results: Mapping[str, Any]) -> str:
    metric_names = sorted(
        {
            name
            for trial in results["trials"]
            for name in trial.get("test_metrics", {})
        }
    )
    renamed = {
        "return_ci": "return_ci95_half_width_rollouts",
        "paired_ci": "paired_ci95_half_width_rollouts",
    }
    base_fields = [
        "experiment_id",
        "algorithm",
        "seed",
        "trial_status",
        "included",
        "exclusion_reason",
        "last_attempt_error",
        "last_attempt_log_path",
        "attempt_count",
        "selected_validation_step",
        "selected_checkpoint_path",
        "selected_checkpoint_sha256",
        "manifest_path",
        "run_directory",
    ]
    fieldnames = base_fields + [
        renamed.get(name, name) for name in metric_names
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for trial in results["trials"]:
        row = {
            "experiment_id": results["experiment_id"],
            "algorithm": trial["algorithm"],
            "seed": trial["seed"],
            "trial_status": trial["status"],
            "included": str(trial["included"]).lower(),
            "exclusion_reason": trial["exclusion_reason"] or "",
            "last_attempt_error": trial["last_attempt_error"] or "",
            "last_attempt_log_path": (
                trial.get("last_attempt_log_path") or ""
            ),
            "attempt_count": trial["attempt_count"],
            "selected_validation_step": _format_number(
                trial.get("selected_validation_step")
            ),
            "selected_checkpoint_path": trial.get(
                "selected_checkpoint", {}
            ).get("path", ""),
            "selected_checkpoint_sha256": trial.get(
                "selected_checkpoint", {}
            ).get("sha256", ""),
            "manifest_path": trial.get("manifest_path", ""),
            "run_directory": trial.get("run_directory", ""),
        }
        for metric_name in metric_names:
            column = renamed.get(metric_name, metric_name)
            row[column] = _format_number(
                trial.get("test_metrics", {}).get(metric_name)
            )
        writer.writerow(row)
    return stream.getvalue()


def _markdown_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def _markdown_text(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value).replace("\r", " ").replace("\n", " ").replace(
        "|", "\\|"
    )


def _results_markdown(results: Mapping[str, Any]) -> str:
    lines = [
        f"# Experiment {results['experiment_id']}",
        "",
        f"- Track: `{results['track']}`",
        f"- Profile: `{results['profile']}`",
        "- Scientific protocol: "
        + ("yes" if results["scientific_protocol"] else "no"),
        (
            "- Trials: "
            f"planned {results['trial_counts']['planned']}; "
            f"completed {results['trial_counts']['completed']}; "
            f"failed {results['trial_counts']['failed']}; "
            f"incomplete {results['trial_counts']['incomplete']}"
        ),
        (
            "- Included in statistics: "
            f"{results['trial_counts']['included']} / "
            f"{results['trial_counts']['requested']}"
        ),
        "",
        "## Regret across independent training seeds",
        "",
        "| Algorithm | n | Median [min, max] | Mean | Sample SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for algorithm in results["algorithms"]:
        summary = algorithm["regret_across_training_seeds"]
        if summary is None:
            cells = [algorithm["algorithm"], "0", "n/a", "n/a", "n/a"]
        else:
            interval = (
                f"{_markdown_number(summary['median'])} "
                f"[{_markdown_number(summary['min'])}, "
                f"{_markdown_number(summary['max'])}]"
            )
            cells = [
                algorithm["algorithm"],
                str(summary["n"]),
                interval,
                _markdown_number(summary["mean"]),
                _markdown_number(summary["sample_sd"]),
            ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            (
                "The sample SD uses the `n - 1` denominator. No population "
                "confidence interval or p-value is inferred from these "
                "training seeds."
            ),
            "",
            "## Raw held-out results",
            "",
            (
                "`return_ci` and `paired_ci` below are 95% half-widths across "
                "evaluation rollouts for one trained seed; they are not "
                "uncertainty across training seeds."
            ),
            "",
            (
                "| Algorithm | Seed | Status | Regret | Paired regret | "
                "Return CI (rollout 95% half-width) | "
                "Paired CI (rollout 95% half-width) | Selected step | "
                "Checkpoint SHA-256 | Run directory | Attempt log | "
                "Last attempt error |"
            ),
            "|---|---:|---|---:|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for trial in results["trials"]:
        metrics = trial.get("test_metrics", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    trial["algorithm"],
                    str(trial["seed"]),
                    trial["status"],
                    _markdown_number(metrics.get("regret")),
                    _markdown_number(metrics.get("paired_regret")),
                    _markdown_number(metrics.get("return_ci")),
                    _markdown_number(metrics.get("paired_ci")),
                    _markdown_number(
                        trial.get("selected_validation_step")
                    ),
                    _markdown_text(
                        trial.get("selected_checkpoint", {}).get("sha256")
                    ),
                    _markdown_text(trial.get("run_directory")),
                    _markdown_text(
                        trial.get("last_attempt_log_path")
                    ),
                    _markdown_text(trial.get("last_attempt_error")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(path)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        if not overwrite and path.exists():
            raise FileExistsError(path)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_results(
    experiment_path: str | Path,
    *,
    overwrite: bool = True,
) -> dict[str, Path]:
    """Refresh deterministic derived files next to ``experiment.json``.

    Child run directories are read-only inputs.  Overwriting is enabled for
    these three derived views because ``experiment.json`` remains the source of
    truth and aggregation must be refreshable after a retry finishes.
    """
    source_path = Path(experiment_path).resolve()
    results = aggregate_experiment(source_path)
    paths = {
        "json": source_path.parent / "results.json",
        "csv": source_path.parent / "results.csv",
        "report": source_path.parent / "report.md",
    }
    if not overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(existing[0])
    json_text = json.dumps(
        results,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    _atomic_write_text(paths["json"], json_text, overwrite=overwrite)
    _atomic_write_text(
        paths["csv"], _results_csv(results), overwrite=overwrite
    )
    _atomic_write_text(
        paths["report"], _results_markdown(results), overwrite=overwrite
    )
    return paths
