"""Reproducible, non-overwriting artifacts for training experiments.

Learning curves are only meaningful when they can be tied back to the exact
configuration and source tree that produced them.  This module deliberately
does not know anything about PyTorch or a particular learner.  It creates a
unique run directory, snapshots configuration, records source/runtime
provenance, and exposes paths that existing trainers can save to.

Typical use::

    run = create_run(
        "dqn_core",
        seed=cfg.seed,
        training_config=cfg,
        environment_config=cfg.core,
    )
    run.append_metrics(
        step=10_000, split="validation", metrics={"regret": 0.2}
    )
    torch.save(model.state_dict(), run.best_checkpoint_path)
    torch.save(model.state_dict(), run.last_checkpoint_path)
    run.finish(summary={"best_regret": 0.2})

Each trainer remains responsible for *when* to save its best and last model.
In particular, replacing ``best.pt`` within one run is intentional when a new
best checkpoint is found.  The no-overwrite guarantee is between run
directories: creating an explicitly named run twice raises ``FileExistsError``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import secrets
import subprocess
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from importlib import metadata as importlib_metadata
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1

DEFAULT_DEPENDENCIES = (
    "torch",
    "gymnasium",
    "numpy",
    "onnx",
    "pytest",
    "tqdm",
)

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_FINAL_STATUSES = frozenset({"completed", "failed", "interrupted"})


def _to_jsonable(value: Any) -> Any:
    """Convert supported experiment values to deterministic JSON data.

    Unordered containers are intentionally rejected.  Silently serializing a
    set, for example, would make configuration hashes depend on iteration
    order.
    NumPy scalars are accepted through their scalar ``item()`` protocol without
    making NumPy a dependency of this infrastructure module.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return _to_jsonable(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "experiment artifacts cannot contain NaN or infinity"
            )
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("datetime values must include a timezone")
        return (
            value.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "experiment artifact mapping keys must be strings"
                )
            normalized[key] = _to_jsonable(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        scalar = item_method()
        if scalar is not value:
            return _to_jsonable(scalar)

    raise TypeError(
        f"unsupported experiment artifact value: {type(value).__name__}"
    )


def _canonical_json(value: Any) -> bytes:
    normalized = _to_jsonable(value)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    """Return a full SHA-256 hash of a JSON-compatible experiment value."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return one file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("artifact timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write_json(path: Path, value: Any, *, overwrite: bool) -> None:
    """Write through a same-directory temporary followed by atomic replace."""
    normalized = _to_jsonable(value)
    payload = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"

    if not overwrite and path.exists():
        raise FileExistsError(path)

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

        if not overwrite and path.exists():
            raise FileExistsError(path)
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


def _git_command(project_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _source_provenance(project_root: Path) -> dict[str, Any]:
    commit = _git_command(project_root, "rev-parse", "HEAD")
    status = _git_command(project_root, "status", "--porcelain")
    return {
        "git_commit": commit or None,
        "git_dirty": None if status is None else bool(status),
    }


def _dependency_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _validate_component(value: str, *, run_id: bool = False) -> str:
    pattern = _SAFE_RUN_ID if run_id else _SAFE_COMPONENT
    label = "run_id" if run_id else "algorithm"
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        allowed = "letters, digits, underscores, and hyphens"
        if run_id:
            allowed += ", and periods"
        raise ValueError(
            f"{label} must start with an alphanumeric character and contain "
            f"only {allowed}"
        )
    return value


def _seed_directory(seed: int | None) -> str:
    if seed is None:
        return "seed_unset"
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer or None")
    seed = int(seed)
    if seed < 0:
        return f"seed_neg{abs(seed):03d}"
    return f"seed_{seed:03d}"


def _allocate_run_directory(
    seed_root: Path,
    created_at: datetime,
    requested_id: str | None,
) -> tuple[Path, str]:
    seed_root.mkdir(parents=True, exist_ok=True)
    if requested_id is not None:
        identifier = _validate_component(requested_id, run_id=True)
        run_directory = seed_root / identifier
        run_directory.mkdir(exist_ok=False)
        return run_directory, identifier

    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    for _ in range(100):
        identifier = f"{timestamp}_{secrets.token_hex(4)}"
        run_directory = seed_root / identifier
        try:
            run_directory.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return run_directory, identifier
    raise FileExistsError(
        "could not allocate a unique experiment run directory"
    )


@dataclass(frozen=True)
class RunArtifacts:
    """Paths and append/finalize operations belonging to one experiment run."""

    run_id: str
    run_directory: Path
    manifest_path: Path
    config_path: Path
    metrics_path: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path

    def append_metrics(
        self,
        *,
        step: int,
        split: str,
        metrics: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Append one self-contained metric record to ``metrics.jsonl``.

        Multiple splits may share a step (for example, training loss and
        validation regret).  JSON Lines keeps partial training results readable
        even if a process stops before the run is finalized.
        """
        if isinstance(step, bool) or not isinstance(step, Integral):
            raise TypeError("metric step must be an integer")
        if int(step) < 0:
            raise ValueError("metric step must be non-negative")
        if not isinstance(split, str) or not split.strip():
            raise ValueError("metric split must be a non-empty string")
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metric metadata must be a mapping or None")

        record = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "timestamp_utc": _format_utc(timestamp or _utc_now()),
            "step": int(step),
            "split": split,
            "metrics": _to_jsonable(metrics),
        }
        if metadata is not None:
            record["metadata"] = _to_jsonable(metadata)
        line = _canonical_json(record).decode("utf-8") + "\n"
        with self.metrics_path.open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(line)

    def read_manifest(self) -> dict[str, Any]:
        """Read the current manifest from disk."""
        return _read_json(self.manifest_path)

    def finish(
        self,
        *,
        status: str = "completed",
        summary: Mapping[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Atomically mark a run completed, failed, or interrupted."""
        if status not in _FINAL_STATUSES:
            allowed = ", ".join(sorted(_FINAL_STATUSES))
            raise ValueError(f"final status must be one of: {allowed}")

        manifest = self.read_manifest()
        if manifest.get("status") != "running":
            raise RuntimeError("only a running experiment can be finalized")

        finished_at = _format_utc(timestamp or _utc_now())
        manifest["status"] = status
        manifest["timestamps"]["updated_at_utc"] = finished_at
        manifest["timestamps"]["finished_at_utc"] = finished_at
        manifest["summary"] = _to_jsonable(summary or {})
        _atomic_write_json(self.manifest_path, manifest, overwrite=True)


def create_run(
    algorithm: str,
    *,
    seed: int | None,
    training_config: Any,
    environment_config: Any,
    root: str | Path = Path("exports") / "runs",
    project_root: str | Path | None = None,
    run_id: str | None = None,
    dependency_names: Sequence[str] = DEFAULT_DEPENDENCIES,
    metadata: Mapping[str, Any] | None = None,
) -> RunArtifacts:
    """Create and describe a unique experiment run.

    ``run_id`` is primarily useful for orchestration and tests.  If supplied it
    is exclusive: an existing directory is never reused.  Without it, a UTC
    timestamp plus a random suffix provides readable, collision-resistant IDs.
    Configuration values are normalized before directory creation so an
    unserializable config cannot leave an apparently valid run behind.
    """
    algorithm = _validate_component(algorithm)
    seed_directory = _seed_directory(seed)
    normalized_training = _to_jsonable(training_config)
    normalized_environment = _to_jsonable(environment_config)
    normalized_metadata = _to_jsonable(metadata or {})

    created_at = _utc_now()
    source_root = (
        Path(project_root) if project_root is not None else Path.cwd()
    )
    source_provenance = _source_provenance(source_root)
    dependency_versions = _dependency_versions(dependency_names)

    root_path = Path(root)
    run_directory, identifier = _allocate_run_directory(
        root_path / algorithm / seed_directory,
        created_at,
        run_id,
    )

    config_path = run_directory / "config.json"
    metrics_path = run_directory / "metrics.jsonl"
    manifest_path = run_directory / "manifest.json"
    best_checkpoint_path = run_directory / "best.pt"
    last_checkpoint_path = run_directory / "last.pt"

    config_document = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "training": normalized_training,
        "environment": normalized_environment,
    }
    _atomic_write_json(config_path, config_document, overwrite=False)
    metrics_path.touch(exist_ok=False)

    timestamp = _format_utc(created_at)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": identifier,
        "algorithm": algorithm,
        "seed": None if seed is None else int(seed),
        "status": "running",
        "timestamps": {
            "created_at_utc": timestamp,
            "updated_at_utc": timestamp,
            "finished_at_utc": None,
        },
        "configuration": {
            "path": config_path.name,
            "sha256": stable_hash(config_document),
            "training_sha256": stable_hash(normalized_training),
            "environment_sha256": stable_hash(normalized_environment),
        },
        "source": source_provenance,
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependencies": dependency_versions,
        },
        "artifacts": {
            "best_checkpoint": best_checkpoint_path.name,
            "last_checkpoint": last_checkpoint_path.name,
            "metrics": metrics_path.name,
        },
        "metadata": normalized_metadata,
    }
    _atomic_write_json(manifest_path, manifest, overwrite=False)

    return RunArtifacts(
        run_id=identifier,
        run_directory=run_directory,
        manifest_path=manifest_path,
        config_path=config_path,
        metrics_path=metrics_path,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
    )
