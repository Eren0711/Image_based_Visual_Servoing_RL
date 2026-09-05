"""Run provenance and immutable resolved-config recording."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA_VERSION = 1
DIRECT_DEPENDENCIES = (
    "numpy",
    "scipy",
    "gymnasium",
    "stable-baselines3",
    "torch",
    "matplotlib",
    "PyYAML",
    "tensorboard",
    "tqdm",
    "rich",
    "quadprog",
)


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: Sequence[str], repository_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def collect_git_state(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    status = _git(["status", "--porcelain"], repository_root)
    tracked_diff = _git(["diff", "--binary", "HEAD"], repository_root)
    return {
        "commit": _git(["rev-parse", "HEAD"], repository_root),
        "branch": _git(["branch", "--show-current"], repository_root),
        "dirty": bool(status) if status is not None else None,
        "status_porcelain": status.splitlines() if status else [],
        "tracked_diff_sha256": (
            hashlib.sha256(tracked_diff.encode("utf-8")).hexdigest()
            if tracked_diff is not None else None
        ),
    }


def collect_dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in DIRECT_DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def initialize_run_manifest(
    run_dir: str | Path,
    *,
    run_id: str,
    run_kind: str,
    config: Mapping[str, Any],
    seed: int,
    command: Sequence[str],
    runtime_options: Mapping[str, Any] | None = None,
    source_config: str | Path | None = None,
    source_model: str | Path | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Create a run directory, resolved config, and initial manifest."""
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=False)
    config_copy = deepcopy(dict(config))
    resolved_path = destination / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(config_copy, sort_keys=False), encoding="utf-8"
    )

    source_model_path = Path(source_model).expanduser() if source_model else None
    if source_model_path and not source_model_path.exists():
        zip_candidate = Path(f"{source_model_path}.zip")
        if zip_candidate.exists():
            source_model_path = zip_candidate
    source_config_path = (
        Path(source_config).expanduser().resolve() if source_config else None
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "run_kind": run_kind,
        "status": "running",
        "created_at_utc": utc_now(),
        "completed_at_utc": None,
        "seed": int(seed),
        "command": list(command),
        "runtime_options": dict(runtime_options or {}),
        "config": {
            "source_path": str(source_config_path) if source_config_path else None,
            "source_sha256": (
                file_sha256(source_config_path)
                if source_config_path and source_config_path.is_file() else None
            ),
            "path": "resolved_config.yaml",
            "semantic_sha256": semantic_sha256(config_copy),
        },
        "source_model": (
            {
                "path": str(source_model_path),
                "sha256": file_sha256(source_model_path),
            }
            if source_model_path and source_model_path.is_file()
            else None
        ),
        "output_model": None,
        "git": collect_git_state(repository_root),
        "platform": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": collect_dependency_versions(),
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def finalize_manifest(
    manifest_path: str | Path,
    *,
    status: str,
    output_model: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Finalize an existing manifest after success or a controlled failure."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = status
    completed_at = utc_now()
    manifest["completed_at_utc"] = completed_at
    try:
        started = datetime.fromisoformat(manifest["created_at_utc"])
        completed = datetime.fromisoformat(completed_at)
        manifest["wall_time_seconds"] = max(
            0.0, (completed - started).total_seconds()
        )
    except (KeyError, TypeError, ValueError):
        manifest["wall_time_seconds"] = None
    if output_model is not None:
        model_path = Path(output_model)
        zip_candidate = Path(f"{model_path}.zip")
        if not model_path.exists() and zip_candidate.exists():
            model_path = zip_candidate
        manifest["output_model"] = {
            "path": str(model_path),
            "sha256": file_sha256(model_path) if model_path.is_file() else None,
        }
    if extra:
        manifest.update(dict(extra))
    _write_json(path, manifest)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
