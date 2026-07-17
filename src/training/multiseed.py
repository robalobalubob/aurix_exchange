"""Plan and execute M1 training-seed replication studies.

M0 made one training run reproducible. M1 groups independently trained runs
into a predeclared study, keeps failures visible, and summarizes run-level test
results. The exact OU benchmark is the only valid M1 track because all four
learners share its DP-regret endpoint.

The public ``run`` command is intentionally sequential. Parallel CPU learners
would compete for cores and memory, making an already large study less
predictable. Each trial still runs in a fresh Python process so PyTorch state
and failures cannot leak into the next training seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from src.env.ou_core import OUCoreConfig
from src.training.artifacts import stable_hash
from src.training.train_a2c import TrainA2CConfig, train as train_a2c
from src.training.train_core import TrainCoreConfig, train as train_dqn_core
from src.training.train_ppo import TrainPPOConfig, train as train_ppo
from src.training.train_reinforce import (
    TrainReinforceConfig,
    train as train_reinforce,
)


EXPERIMENT_SCHEMA_VERSION = 1
TRACK = "exact_ou_benchmark"
FULL_PROFILE = "full"
SMOKE_PROFILE = "smoke"
PROFILES = (FULL_PROFILE, SMOKE_PROFILE)
DEFAULT_ALGORITHMS = (
    "dqn_core",
    "reinforce_core",
    "a2c_core",
    "ppo_core",
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

_SAFE_EXPERIMENT_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,87}$"
)
_OPERATIONAL_CONFIG_FIELDS = frozenset(
    {"seed", "run_id", "artifact_root", "dp_cache_dir"}
)
_EXECUTION_LOCK_NAME = "execution.lock.json"


@dataclass(frozen=True)
class AlgorithmSpec:
    """One exact-benchmark learner exposed through a common M1 interface."""

    name: str
    config_factory: Callable[[], Any]
    trainer: Callable[[Any], Any]


ALGORITHM_SPECS = {
    "dqn_core": AlgorithmSpec(
        "dqn_core",
        TrainCoreConfig,
        train_dqn_core,
    ),
    "reinforce_core": AlgorithmSpec(
        "reinforce_core",
        TrainReinforceConfig,
        train_reinforce,
    ),
    "a2c_core": AlgorithmSpec(
        "a2c_core",
        TrainA2CConfig,
        train_a2c,
    ),
    "ppo_core": AlgorithmSpec(
        "ppo_core",
        TrainPPOConfig,
        train_ppo,
    ),
}


TrialExecutor = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], int]
ResultWriter = Callable[[Path], Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("experiment timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace a study document derived from in-memory state."""
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _execution_lock_path(experiment_path: str | Path) -> Path:
    """Return the on-disk lock shared by every runner for one study."""
    return Path(experiment_path).resolve().parent / _EXECUTION_LOCK_NAME


def _lock_owner_summary(owner: dict[str, Any] | None) -> str:
    if owner is None:
        return "owner metadata is unreadable or still being written"
    command = owner.get("command")
    if isinstance(command, list):
        command_text = " ".join(str(part) for part in command)
    else:
        command_text = str(command or "unknown")
    return (
        f"pid={owner.get('pid', 'unknown')}, "
        f"host={owner.get('hostname', 'unknown')}, "
        f"started={owner.get('started_at_utc', 'unknown')}, "
        f"command={command_text}"
    )


def _read_lock_owner(lock_path: Path) -> dict[str, Any] | None:
    try:
        return _read_json(lock_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _release_execution_lock(lock_path: Path, owner_token: str) -> bool:
    """Remove only the lock created by this runner instance."""
    owner = _read_lock_owner(lock_path)
    if owner is None or owner.get("owner_token") != owner_token:
        return False
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return True
    return True


@contextmanager
def _study_execution_lock(
    experiment_path: str | Path,
) -> Iterator[Path]:
    """Acquire an atomic, inspectable per-study execution lock."""
    resolved_experiment_path = Path(experiment_path).resolve()
    lock_path = _execution_lock_path(resolved_experiment_path)
    owner_token = uuid.uuid4().hex
    owner = {
        "schema_version": 1,
        "owner_token": owner_token,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at_utc": _format_utc(_utc_now()),
        "experiment_path": str(resolved_experiment_path),
        "working_directory": str(Path.cwd().resolve()),
        "command": list(sys.argv),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as error:
        existing_owner = _read_lock_owner(lock_path)
        raise RuntimeError(
            "study is already being executed; "
            f"lock={lock_path}; {_lock_owner_summary(existing_owner)}. "
            "A lock left by a crashed process must be removed only after "
            "verifying that its recorded owner is no longer running."
        ) from error

    try:
        with os.fdopen(
            descriptor,
            mode="w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            json.dump(
                owner,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        lock_path.unlink(missing_ok=True)
        raise

    try:
        yield lock_path
    finally:
        if not _release_execution_lock(lock_path, owner_token):
            print(
                "warning: execution lock ownership changed; refusing to "
                f"remove {lock_path}",
                file=sys.stderr,
                flush=True,
            )


def _validate_experiment_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or _SAFE_EXPERIMENT_ID.fullmatch(value) is None
    ):
        raise ValueError(
            "experiment_id must start with an alphanumeric character, be at "
            "most 88 characters, and contain only letters, digits, periods, "
            "underscores, or hyphens"
        )
    return value


def _normalize_algorithms(values: Sequence[str]) -> tuple[str, ...]:
    if not values:
        raise ValueError("at least one algorithm is required")
    if len(set(values)) != len(values):
        raise ValueError("algorithm names must be unique")
    unknown = sorted(set(values) - set(ALGORITHM_SPECS))
    if unknown:
        raise ValueError(
            "M1 accepts exact-OU algorithms only; unsupported: "
            + ", ".join(unknown)
        )
    selected = set(values)
    return tuple(name for name in DEFAULT_ALGORITHMS if name in selected)


def _normalize_seeds(values: Sequence[int]) -> tuple[int, ...]:
    if not values:
        raise ValueError("at least one training seed is required")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int)
        for seed in values
    ):
        raise TypeError("training seeds must be integers")
    if any(seed < 0 for seed in values):
        raise ValueError("training seeds must be non-negative")
    if len(set(values)) != len(values):
        raise ValueError("training seeds must be unique")
    return tuple(sorted(values))


def _git_output(project_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def source_fingerprint(project_root: str | Path) -> str:
    """Hash executable project inputs in a path- and order-stable way."""
    root = Path(project_root).resolve()
    candidates = list((root / "src").rglob("*.py"))
    candidates.extend(
        root / name for name in ("requirements.txt", "pytest.ini")
    )
    files = sorted(
        (path for path in candidates if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"no source files found under {root}")

    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def source_state(project_root: str | Path) -> dict[str, Any]:
    """Describe the committed revision and relevant source cleanliness."""
    root = Path(project_root).resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    tracked = _git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    untracked = _git_output(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "src",
        "requirements.txt",
        "pytest.ini",
    )
    tracked_changes = [] if not tracked else tracked.splitlines()
    untracked_source = [] if not untracked else untracked.splitlines()
    clean = (
        commit is not None
        and tracked is not None
        and untracked is not None
        and not tracked_changes
        and not untracked_source
    )
    return {
        "project_root": str(root),
        "git_commit": commit,
        "scientific_clean": clean,
        "tracked_changes": tracked_changes,
        "untracked_source": untracked_source,
        "content_sha256": source_fingerprint(root),
    }


def make_training_config(
    algorithm: str,
    *,
    seed: int,
    artifact_root: str | Path,
    run_id: str,
    profile: str = FULL_PROFILE,
) -> Any:
    """Build one frozen trainer config without changing learning defaults."""
    if algorithm not in ALGORITHM_SPECS:
        raise ValueError(f"unsupported M1 algorithm: {algorithm}")
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(PROFILES)}")

    root = str(Path(artifact_root).resolve())
    cfg = replace(
        ALGORITHM_SPECS[algorithm].config_factory(),
        seed=seed,
        artifact_root=root,
        run_id=run_id,
    )
    if profile == FULL_PROFILE:
        return cfg

    smoke_core = replace(OUCoreConfig(), t_max=8)
    cache_dir = str(Path(root).parent / "smoke_dp_cache")
    common = {
        "core": smoke_core,
        "eval_episodes": 2,
        "final_eval_episodes": 2,
        "log_every": 1,
        "dp_cache_dir": cache_dir,
    }
    if algorithm == "dqn_core":
        return replace(
            cfg,
            **common,
            total_steps=4,
            warmup_steps=4,
            target_sync=4,
            batch_size=2,
            buffer_capacity=16,
            eps_decay_steps=4,
            eval_every=4,
        )
    if algorithm in {"reinforce_core", "a2c_core"}:
        return replace(
            cfg,
            **common,
            total_updates=1,
            batch_episodes=2,
            eval_every=1,
        )
    return replace(
        cfg,
        **common,
        total_updates=1,
        batch_episodes=2,
        epochs=1,
        minibatch_size=16,
        eval_every=1,
    )


def _periodic_evaluations(
    total: int,
    every: int,
    first_allowed: int,
) -> int:
    if total < 0 or every <= 0 or first_allowed < 0:
        raise ValueError("invalid evaluation schedule")
    return sum(
        step >= first_allowed
        for step in range(every, total + 1, every)
    )


def estimate_environment_steps(algorithm: str, cfg: Any) -> dict[str, int]:
    """Estimate environment ``step()`` calls, including CRN evaluation."""
    horizon = cfg.core.t_max
    if algorithm == "dqn_core":
        training = cfg.total_steps
        evaluations = _periodic_evaluations(
            cfg.total_steps,
            cfg.eval_every,
            cfg.warmup_steps,
        )
    else:
        training = cfg.total_updates * cfg.batch_episodes * horizon
        evaluations = _periodic_evaluations(
            cfg.total_updates,
            cfg.eval_every,
            0,
        )

    validation_oracle = cfg.eval_episodes * horizon
    validation_policy = evaluations * cfg.eval_episodes * horizon
    test_policy = cfg.final_eval_episodes * horizon
    test_oracle = cfg.final_eval_episodes * horizon
    total = (
        training
        + validation_oracle
        + validation_policy
        + test_policy
        + test_oracle
    )
    return {
        "training": training,
        "validation_policy": validation_policy,
        "validation_oracle": validation_oracle,
        "test_policy": test_policy,
        "test_oracle": test_oracle,
        "total": total,
    }


def _protocol_for_algorithm(
    algorithm: str,
    *,
    profile: str,
    artifact_root: Path,
) -> dict[str, Any]:
    cfg = make_training_config(
        algorithm,
        seed=0,
        artifact_root=artifact_root,
        run_id="protocol",
        profile=profile,
    )
    training_config = asdict(cfg)
    comparison_config = {
        key: value
        for key, value in training_config.items()
        if key not in _OPERATIONAL_CONFIG_FIELDS
    }
    environment_config = asdict(cfg.core)
    return {
        "comparison_config": comparison_config,
        "comparison_config_sha256": stable_hash(comparison_config),
        "environment_sha256": stable_hash(environment_config),
        "validation": {
            "seed0": cfg.eval_seed0,
            "n_episodes": cfg.eval_episodes,
            "used_for_selection": True,
        },
        "test": {
            "seed0": cfg.test_seed0,
            "n_episodes": cfg.final_eval_episodes,
            "shared_across_runs": True,
            "used_for_selection": False,
        },
        "environment_steps_per_run": estimate_environment_steps(
            algorithm, cfg
        ),
    }


def build_experiment(
    experiment_id: str,
    *,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    profile: str = FULL_PROFILE,
    artifact_root: str | Path = Path("exports") / "runs",
    project_root: str | Path | None = None,
    source: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Create a deterministic in-memory M1 study declaration."""
    identifier = _validate_experiment_id(experiment_id)
    selected_algorithms = _normalize_algorithms(algorithms)
    selected_seeds = _normalize_seeds(seeds)
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(PROFILES)}")

    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    artifact_path = Path(artifact_root).resolve()
    source_document = source or source_state(root)
    now = _format_utc(timestamp or _utc_now())
    protocol_algorithms = {
        name: _protocol_for_algorithm(
            name,
            profile=profile,
            artifact_root=artifact_path,
        )
        for name in selected_algorithms
    }
    estimated_total = sum(
        protocol_algorithms[name]["environment_steps_per_run"]["total"]
        for name in selected_algorithms
        for _ in selected_seeds
    )
    canonical = (
        profile == FULL_PROFILE
        and selected_algorithms == DEFAULT_ALGORITHMS
        and selected_seeds == DEFAULT_SEEDS
    )

    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": identifier,
        "milestone": "M1",
        "track": TRACK,
        "purpose": "descriptive_training_seed_replication",
        "profile": profile,
        "canonical_protocol": canonical,
        "scientific_protocol": (
            canonical and bool(source_document.get("scientific_clean"))
        ),
        "status": "planned",
        "created_at_utc": now,
        "updated_at_utc": now,
        "project_root": str(root),
        "artifact_root": str(artifact_path),
        "algorithms": list(selected_algorithms),
        "seeds": list(selected_seeds),
        "unit_of_replication": (
            "one independently trained validation-selected checkpoint"
        ),
        "primary_metric": "test.regret",
        "source": source_document,
        "protocol": {
            "algorithms": protocol_algorithms,
            "estimated_environment_steps": estimated_total,
            "interpretation": (
                "Across-seed variation is computed over trained policies, not "
                "over evaluation episodes."
            ),
        },
        "trials": [
            {
                "algorithm": algorithm,
                "seed": seed,
                "status": "pending",
                "attempts": [],
            }
            for algorithm in selected_algorithms
            for seed in selected_seeds
        ],
    }


def create_experiment(
    experiment_id: str,
    *,
    study_root: str | Path = Path("exports") / "studies",
    **build_arguments: Any,
) -> Path:
    """Create a non-overwriting study directory and experiment manifest."""
    experiment = build_experiment(experiment_id, **build_arguments)
    study_directory = Path(study_root).resolve() / experiment_id
    study_directory.mkdir(parents=True, exist_ok=False)
    experiment_path = study_directory / "experiment.json"
    _atomic_write_json(experiment_path, experiment)
    return experiment_path


def _seed_directory(seed: int) -> str:
    return f"seed_{seed:03d}"


def _attempt_paths(
    experiment: dict[str, Any],
    trial: dict[str, Any],
    attempt_number: int,
) -> tuple[str, Path, Path]:
    run_id = f"{experiment['experiment_id']}.a{attempt_number:02d}"
    run_directory = (
        Path(experiment["artifact_root"])
        / trial["algorithm"]
        / _seed_directory(trial["seed"])
        / run_id
    ).resolve()
    return run_id, run_directory, run_directory / "manifest.json"


def _attempt_log_path(
    experiment_path: Path,
    trial: dict[str, Any],
    attempt_number: int,
) -> Path:
    """Place launcher logs outside trainer-owned run directories."""
    return (
        experiment_path.parent
        / "logs"
        / trial["algorithm"]
        / _seed_directory(trial["seed"])
        / f"attempt_{attempt_number:02d}.log"
    ).resolve()


def _source_matches(experiment: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    current = source_state(experiment["project_root"])
    planned = experiment["source"]
    matches = (
        current["content_sha256"] == planned["content_sha256"]
        and current["git_commit"] == planned["git_commit"]
    )
    return matches, current


def _source_attestation(source: dict[str, Any]) -> dict[str, Any]:
    """Keep the source facts needed to validate one trial attempt."""
    return {
        "content_sha256": source.get("content_sha256"),
        "git_commit": source.get("git_commit"),
        "scientific_clean": bool(source.get("scientific_clean")),
    }


def _load_child_status(manifest_path: Path) -> str | None:
    if not manifest_path.is_file():
        return None
    try:
        return _read_json(manifest_path).get("status")
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _run_logged_subprocess(
    command: Sequence[str],
    *,
    cwd: str | Path,
    log_path: str | Path,
) -> int:
    """Run a child while teeing combined output to console and disk."""
    resolved_log_path = Path(log_path).resolve()
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_log_path.open(
        mode="x",
        encoding="utf-8",
        newline="",
    ) as log_stream:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            if process.stdout is None:
                raise RuntimeError("subprocess output pipe was not created")
            for chunk in process.stdout:
                log_stream.write(chunk)
                log_stream.flush()
                sys.stdout.write(chunk)
                sys.stdout.flush()
            return_code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            log_stream.flush()
            os.fsync(log_stream.fileno())
    return return_code


def _default_trial_executor(
    experiment: dict[str, Any],
    trial: dict[str, Any],
    attempt: dict[str, Any],
) -> int:
    command = [
        sys.executable,
        "-u",
        "-m",
        "src.training.multiseed",
        "trial",
        "--algorithm",
        trial["algorithm"],
        "--seed",
        str(trial["seed"]),
        "--profile",
        experiment["profile"],
        "--artifact-root",
        experiment["artifact_root"],
        "--run-id",
        attempt["run_id"],
    ]
    print(
        f"\n=== {trial['algorithm']} | training seed {trial['seed']} | "
        f"attempt {attempt['attempt']} ===",
        flush=True,
    )
    return _run_logged_subprocess(
        command,
        cwd=experiment["project_root"],
        log_path=attempt["log_path"],
    )


def _default_result_writer(experiment_path: Path) -> Any:
    from src.training.aggregate_runs import write_results

    return write_results(experiment_path, overwrite=True)


def _refresh_results(
    experiment_path: Path,
    experiment: dict[str, Any],
    writer: ResultWriter,
) -> None:
    try:
        writer(experiment_path)
    except Exception as error:  # noqa: BLE001 - persist integrity failures.
        failure = {
            "at_utc": _format_utc(_utc_now()),
            "error_type": type(error).__name__,
            "message": str(error),
            "status_before_failure": experiment.get("status"),
        }
        experiment.setdefault("aggregation_failures", []).append(failure)
        experiment["aggregation_error"] = failure
        experiment["status"] = "aggregation_failed"
        experiment["updated_at_utc"] = failure["at_utc"]
        _atomic_write_json(experiment_path, experiment)
        raise RuntimeError(
            "aggregate report failed; execution stopped because the study "
            "evidence could not be validated"
        ) from error
    if experiment.get("aggregation_error") is not None:
        experiment["aggregation_error"] = None
        experiment["updated_at_utc"] = _format_utc(_utc_now())
        _atomic_write_json(experiment_path, experiment)


def execute_experiment(
    experiment_path: str | Path,
    *,
    resume: bool = False,
    allow_dirty_source: bool = False,
    trial_executor: TrialExecutor | None = None,
    result_writer: ResultWriter | None = None,
) -> dict[str, Any]:
    """Execute pending trials sequentially and persist every transition."""
    path = Path(experiment_path).resolve()
    with _study_execution_lock(path):
        return _execute_experiment_locked(
            path,
            resume=resume,
            allow_dirty_source=allow_dirty_source,
            trial_executor=trial_executor,
            result_writer=result_writer,
        )


def _execute_experiment_locked(
    path: Path,
    *,
    resume: bool,
    allow_dirty_source: bool,
    trial_executor: TrialExecutor | None,
    result_writer: ResultWriter | None,
) -> dict[str, Any]:
    """Execute a study after its caller acquires the per-study lock."""
    experiment = _read_json(path)
    if experiment.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("unsupported experiment schema version")
    if experiment.get("track") != TRACK:
        raise ValueError("M1 runner only supports the exact OU benchmark")
    if experiment["status"] != "planned" and not resume:
        raise RuntimeError("existing studies require resume=True")

    matches, current_source = _source_matches(experiment)
    if not matches:
        raise RuntimeError(
            "source changed since this study was planned; create a new study"
        )
    if (
        experiment["profile"] == FULL_PROFILE
        and not current_source["scientific_clean"]
        and not allow_dirty_source
    ):
        raise RuntimeError(
            "full M1 execution requires committed, unchanged source; commit "
            "the implementation or pass --allow-dirty-source for a "
            "non-scientific run"
        )
    if not current_source["scientific_clean"]:
        experiment["scientific_protocol"] = False

    executor = trial_executor or _default_trial_executor
    writer = result_writer or _default_result_writer
    experiment["status"] = "running"
    experiment["updated_at_utc"] = _format_utc(_utc_now())
    _atomic_write_json(path, experiment)
    _refresh_results(path, experiment, writer)

    stop_for_source_change = False
    for trial in experiment["trials"]:
        if trial["status"] == "completed":
            continue
        if trial["status"] == "running":
            if not resume:
                raise RuntimeError("running trial requires resume=True")
            prior = trial["attempts"][-1]
            prior_manifest = Path(prior["manifest_path"])
            if _load_child_status(prior_manifest) == "completed":
                recovery_matches, recovery_source = _source_matches(
                    experiment
                )
                prior["source_after"] = _source_attestation(
                    recovery_source
                )
                prior["status"] = "completed"
                prior["finished_at_utc"] = (
                    prior.get("finished_at_utc") or _format_utc(_utc_now())
                )
                trial["status"] = "completed"
                if not recovery_matches:
                    experiment["status"] = "source_changed"
                    experiment["scientific_protocol"] = False
                    stop_for_source_change = True
                experiment["updated_at_utc"] = _format_utc(_utc_now())
                _atomic_write_json(path, experiment)
                _refresh_results(path, experiment, writer)
                if stop_for_source_change:
                    break
                continue
            prior["status"] = "interrupted"
            prior["finished_at_utc"] = _format_utc(_utc_now())
            prior["error"] = "runner resumed after an incomplete attempt"

        before_matches, source_before = _source_matches(experiment)
        if not before_matches:
            experiment["status"] = "source_changed"
            experiment["scientific_protocol"] = False
            stop_for_source_change = True
            experiment["updated_at_utc"] = _format_utc(_utc_now())
            _atomic_write_json(path, experiment)
            _refresh_results(path, experiment, writer)
            break

        attempt_number = len(trial["attempts"]) + 1
        run_id, run_directory, manifest_path = _attempt_paths(
            experiment,
            trial,
            attempt_number,
        )
        attempt = {
            "attempt": attempt_number,
            "run_id": run_id,
            "status": "running",
            "started_at_utc": _format_utc(_utc_now()),
            "finished_at_utc": None,
            "run_directory": str(run_directory),
            "manifest_path": str(manifest_path),
            "log_path": str(
                _attempt_log_path(path, trial, attempt_number)
            ),
            "source_before": _source_attestation(source_before),
            "source_after": None,
            "return_code": None,
            "error": None,
        }
        trial["attempts"].append(attempt)
        trial["status"] = "running"
        experiment["updated_at_utc"] = _format_utc(_utc_now())
        _atomic_write_json(path, experiment)

        try:
            return_code = executor(experiment, trial, attempt)
        except KeyboardInterrupt:
            after_matches, source_after = _source_matches(experiment)
            attempt["source_after"] = _source_attestation(source_after)
            attempt["status"] = "interrupted"
            attempt["error"] = "keyboard interrupt"
            attempt["finished_at_utc"] = _format_utc(_utc_now())
            trial["status"] = "interrupted"
            experiment["status"] = "interrupted"
            if not after_matches:
                experiment["scientific_protocol"] = False
            experiment["updated_at_utc"] = _format_utc(_utc_now())
            _atomic_write_json(path, experiment)
            _refresh_results(path, experiment, writer)
            raise
        except Exception as error:  # noqa: BLE001 - failure is study data.
            return_code = -1
            attempt["error"] = f"{type(error).__name__}: {error}"

        matches, source_after = _source_matches(experiment)
        attempt["source_after"] = _source_attestation(source_after)
        attempt["return_code"] = return_code
        attempt["finished_at_utc"] = _format_utc(_utc_now())
        child_status = _load_child_status(manifest_path)
        if return_code == 0 and child_status == "completed":
            attempt["status"] = "completed"
            trial["status"] = "completed"
        else:
            attempt["status"] = "failed"
            trial["status"] = "failed"
            if attempt["error"] is None:
                attempt["error"] = (
                    f"trial returned {return_code}; child manifest status "
                    f"was {child_status!r}"
                )

        if not matches:
            experiment["status"] = "source_changed"
            experiment["scientific_protocol"] = False
            stop_for_source_change = True
        experiment["updated_at_utc"] = _format_utc(_utc_now())
        _atomic_write_json(path, experiment)
        _refresh_results(path, experiment, writer)
        if stop_for_source_change:
            break

    if not stop_for_source_change:
        statuses = {trial["status"] for trial in experiment["trials"]}
        experiment["status"] = (
            "completed" if statuses == {"completed"} else "incomplete"
        )
    experiment["updated_at_utc"] = _format_utc(_utc_now())
    _atomic_write_json(path, experiment)
    _refresh_results(path, experiment, writer)
    return experiment


def run_single_trial(
    algorithm: str,
    *,
    seed: int,
    profile: str,
    artifact_root: str | Path,
    run_id: str,
) -> None:
    """Execute one child trial inside its isolated process."""
    cfg = make_training_config(
        algorithm,
        seed=seed,
        artifact_root=artifact_root,
        run_id=run_id,
        profile=profile,
    )
    ALGORITHM_SPECS[algorithm].trainer(cfg)


def render_plan(experiment: dict[str, Any]) -> str:
    """Render a concise, educational execution plan."""
    lines = [
        f"M1 study: {experiment['experiment_id']}",
        f"profile: {experiment['profile']}",
        f"algorithms: {', '.join(experiment['algorithms'])}",
        "training seeds: " + ", ".join(map(str, experiment["seeds"])),
        f"planned trained policies: {len(experiment['trials'])}",
        "estimated environment step() calls: "
        f"{experiment['protocol']['estimated_environment_steps']:,}",
        "execution: sequential, one isolated Python process per policy",
        "primary endpoint: held-out test regret per trained policy",
    ]
    if experiment["profile"] == FULL_PROFILE:
        lines.append(
            "full execution requires committed source and is intentionally "
            "not started by the plan command"
        )
    else:
        lines.append(
            "smoke profile is a wiring check, not scientific evidence"
        )
    return "\n".join(lines)


def _add_study_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_experiment_id: bool,
) -> None:
    parser.add_argument(
        "--experiment-id",
        required=require_experiment_id,
        default=(None if require_experiment_id else "m1_ou_ladder_v1"),
        help="Stable identifier shared by all trials in this study.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(DEFAULT_ALGORITHMS),
        help="Exact-OU learners to include.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Predeclared independent training seeds.",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default=FULL_PROFILE,
    )
    parser.add_argument(
        "--artifact-root",
        default=str(Path("exports") / "runs"),
    )
    parser.add_argument(
        "--study-root",
        default=str(Path("exports") / "studies"),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, run, and summarize M1 training-seed studies."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Print the predeclared workload without creating or running it.",
    )
    _add_study_arguments(plan_parser, require_experiment_id=False)

    run_parser = subparsers.add_parser(
        "run",
        help="Create and sequentially execute a study.",
    )
    _add_study_arguments(run_parser, require_experiment_id=True)
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Retry missing/failed trials without replacing completed ones.",
    )
    run_parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Permit a full non-scientific run from uncommitted source.",
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Regenerate JSON, CSV, and Markdown from recorded trials.",
    )
    summarize_parser.add_argument("--experiment-id", required=True)
    summarize_parser.add_argument(
        "--study-root",
        default=str(Path("exports") / "studies"),
    )

    trial_parser = subparsers.add_parser("trial", help=argparse.SUPPRESS)
    trial_parser.add_argument("--algorithm", required=True)
    trial_parser.add_argument("--seed", type=int, required=True)
    trial_parser.add_argument("--profile", choices=PROFILES, required=True)
    trial_parser.add_argument("--artifact-root", required=True)
    trial_parser.add_argument("--run-id", required=True)
    return parser


def _experiment_path(study_root: str | Path, experiment_id: str) -> Path:
    identifier = _validate_experiment_id(experiment_id)
    return Path(study_root).resolve() / identifier / "experiment.json"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "trial":
        run_single_trial(
            args.algorithm,
            seed=args.seed,
            profile=args.profile,
            artifact_root=args.artifact_root,
            run_id=args.run_id,
        )
        return

    if args.command == "summarize":
        from src.training.aggregate_runs import write_results

        path = _experiment_path(args.study_root, args.experiment_id)
        outputs = write_results(path, overwrite=True)
        for output in outputs.values():
            print(output)
        return

    build_arguments = {
        "algorithms": args.algorithms,
        "seeds": args.seeds,
        "profile": args.profile,
        "artifact_root": args.artifact_root,
    }
    if args.command == "plan":
        experiment = build_experiment(
            args.experiment_id,
            **build_arguments,
        )
        print(render_plan(experiment))
        return

    path = _experiment_path(args.study_root, args.experiment_id)
    if args.resume:
        if not path.is_file():
            parser.error(f"cannot resume missing study: {path}")
    else:
        source = source_state(Path(__file__).resolve().parents[2])
        if (
            args.profile == FULL_PROFILE
            and not source["scientific_clean"]
            and not args.allow_dirty_source
        ):
            parser.error(
                "full M1 execution requires committed source; use plan now, "
                "commit the implementation, then run"
            )
        path = create_experiment(
            args.experiment_id,
            study_root=args.study_root,
            source=source,
            **build_arguments,
        )
    experiment = execute_experiment(
        path,
        resume=args.resume,
        allow_dirty_source=args.allow_dirty_source,
    )
    print(
        f"study {experiment['experiment_id']} finished with status "
        f"{experiment['status']}"
    )


if __name__ == "__main__":
    main()
