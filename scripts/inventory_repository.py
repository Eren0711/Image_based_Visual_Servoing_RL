#!/usr/bin/env python3
"""Build a deterministic, read-only inventory of historical run artifacts.

The Stable-Baselines3 archives are treated as untrusted data: this script reads
ZIP members and JSON only.  It never imports, unpickles, or loads a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
PROJECT_MODULE_PREFIXES = ("envs", "models", "observers", "safety", "runtime")
PROVENANCE_FILENAMES = {
    "manifest": "manifest.json",
    "resolved_config": "resolved_config.yaml",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest without loading the whole artifact in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def walk_values(value: Any) -> Iterable[Any]:
    """Yield every nested JSON value without executing serialized payloads."""

    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_values(child)


def safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def extract_shape(space: Any) -> list[int] | None:
    if not isinstance(space, dict):
        return None
    shape = space.get("_shape")
    if not isinstance(shape, list) or not shape:
        return None
    dimensions: list[int] = []
    for item in shape:
        dimension = safe_int(item)
        if dimension is None:
            return None
        dimensions.append(dimension)
    return dimensions


def discover_provenance(checkpoint: Path, logs_root: Path, repo_root: Path) -> dict[str, Any]:
    """Find only explicit manifests/config snapshots in checkpoint ancestors."""

    found: dict[str, str | None] = {key: None for key in PROVENANCE_FILENAMES}
    current = checkpoint.parent
    while current == logs_root or logs_root in current.parents:
        for key, filename in PROVENANCE_FILENAMES.items():
            candidate = current / filename
            if found[key] is None and candidate.is_file():
                found[key] = relative_posix(candidate, repo_root)
        if current == logs_root:
            break
        current = current.parent

    present = sum(item is not None for item in found.values())
    if present == len(found):
        status = "recorded"
    elif present:
        status = "partial"
    else:
        status = "unknown"
    return {"status": status, **found}


def inspect_sb3_archive(path: Path) -> dict[str, Any]:
    """Read metadata from an SB3 ZIP without deserializing pickle fields."""

    result: dict[str, Any] = {
        "archive_status": "ok",
        "observation_shape": None,
        "seed": None,
        "stable_baselines3_version": None,
        "project_module_references": [],
    }
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = set(archive.namelist())
            if "_stable_baselines3_version" in members:
                version = archive.read("_stable_baselines3_version").decode(
                    "utf-8", errors="replace"
                ).strip()
                result["stable_baselines3_version"] = version or None

            if "data" not in members:
                result["archive_status"] = "metadata_missing"
                return result

            try:
                data = json.loads(archive.read("data").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                result["archive_status"] = "metadata_invalid_json"
                return result

            if not isinstance(data, dict):
                result["archive_status"] = "metadata_not_object"
                return result

            result["observation_shape"] = extract_shape(data.get("observation_space"))
            result["seed"] = safe_int(data.get("seed"))

            module_references: set[str] = set()
            for value in walk_values(data):
                if not isinstance(value, dict):
                    continue
                module = value.get("__module__")
                if not isinstance(module, str):
                    continue
                if module in PROJECT_MODULE_PREFIXES or module.startswith(
                    tuple(prefix + "." for prefix in PROJECT_MODULE_PREFIXES)
                ):
                    module_references.add(module)
            result["project_module_references"] = sorted(module_references)
    except (OSError, zipfile.BadZipFile):
        result["archive_status"] = "invalid_zip"
    return result


def directory_provenance(paths: list[dict[str, Any]]) -> str:
    statuses = {item["provenance"]["status"] for item in paths}
    if not statuses or statuses == {"unknown"}:
        return "unknown"
    if statuses == {"recorded"}:
        return "recorded"
    return "partial"


def summarize_directory(
    directory: Path,
    checkpoints: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    prefix = relative_posix(directory, repo_root).rstrip("/") + "/"
    matching = [item for item in checkpoints if item["path"].startswith(prefix)]
    has_eval_artifacts = (directory / "eval").is_dir() and any(
        item.is_file() for item in (directory / "eval").rglob("*")
    )
    return {
        "path": relative_posix(directory, repo_root),
        "checkpoint_count": len(matching),
        "checkpoint_bytes": sum(item["bytes"] for item in matching),
        "has_eval_artifacts": has_eval_artifacts,
        "provenance_status": directory_provenance(matching),
    }


def build_inventory(repo_root: Path) -> dict[str, Any]:
    logs_root = repo_root / "logs"
    if not logs_root.is_dir():
        raise FileNotFoundError(f"Historical artifact root is missing: {logs_root}")

    checkpoint_paths = sorted(
        (path for path in logs_root.rglob("*.zip") if path.is_file()),
        key=lambda item: relative_posix(item, repo_root),
    )
    checkpoints: list[dict[str, Any]] = []
    for path in checkpoint_paths:
        metadata = inspect_sb3_archive(path)
        checkpoints.append(
            {
                "path": relative_posix(path, repo_root),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                **metadata,
                "provenance": discover_provenance(path, logs_root, repo_root),
            }
        )

    hash_groups: dict[str, list[str]] = defaultdict(list)
    for checkpoint in checkpoints:
        hash_groups[checkpoint["sha256"]].append(checkpoint["path"])
    duplicate_groups = [
        {"sha256": digest, "paths": sorted(paths)}
        for digest, paths in sorted(hash_groups.items())
        if len(paths) > 1
    ]

    stages_root = logs_root / "stages"
    stage_directories = (
        sorted(
            (path for path in stages_root.iterdir() if path.is_dir()),
            key=lambda item: item.name,
        )
        if stages_root.is_dir()
        else []
    )
    top_level_directories = sorted(
        (path for path in logs_root.iterdir() if path.is_dir() and path != stages_root),
        key=lambda item: item.name,
    )
    stages = [summarize_directory(path, checkpoints, repo_root) for path in stage_directories]
    top_level = [
        summarize_directory(path, checkpoints, repo_root) for path in top_level_directories
    ]

    observation_dimensions = Counter()
    seed_values = Counter()
    versions = Counter()
    archive_statuses = Counter()
    provenance_statuses = Counter()
    module_references = Counter()
    for checkpoint in checkpoints:
        shape = checkpoint["observation_shape"]
        shape_label = "unknown" if shape is None else "x".join(str(item) for item in shape)
        observation_dimensions[shape_label] += 1
        seed = checkpoint["seed"]
        seed_values["unknown" if seed is None else str(seed)] += 1
        version = checkpoint["stable_baselines3_version"]
        versions[version or "unknown"] += 1
        archive_statuses[checkpoint["archive_status"]] += 1
        provenance_statuses[checkpoint["provenance"]["status"]] += 1
        module_references.update(checkpoint["project_module_references"])

    stage4b_paths = [
        stage["path"]
        for stage in stages
        if Path(stage["path"]).name.lower().startswith("stage4b")
    ]
    recursive_directories = sum(1 for path in logs_root.rglob("*") if path.is_dir())

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "repository_root": ".",
            "artifact_root": "logs",
            "method": "read ZIP members and JSON only; no model load or unpickling",
            "timestamp_policy": "omitted for deterministic regeneration",
        },
        "layout": {
            "recursive_directory_count": recursive_directories,
            "stage_directory_count": len(stages),
            "top_level_directory_count_excluding_stages": len(top_level),
            "checkpoint_parent_directory_count": len(
                {str(Path(item["path"]).parent) for item in checkpoints}
            ),
            "manifest_backed_run_directory_count": sum(
                1
                for path in logs_root.rglob("manifest.json")
                if path.is_file()
            ),
            "stage_directories_with_checkpoints": sum(
                stage["checkpoint_count"] > 0 for stage in stages
            ),
            "stage_directories_with_eval_artifacts_only": sum(
                stage["checkpoint_count"] == 0 and stage["has_eval_artifacts"]
                for stage in stages
            ),
            "stages": stages,
            "top_level_directories": top_level,
        },
        "checkpoint_summary": {
            "count": len(checkpoints),
            "bytes": sum(item["bytes"] for item in checkpoints),
            "unique_sha256_count": len(hash_groups),
            "duplicate_hash_group_count": len(duplicate_groups),
            "observation_shapes": dict(sorted(observation_dimensions.items())),
            "seed_values": dict(sorted(seed_values.items())),
            "stable_baselines3_versions": dict(sorted(versions.items())),
            "archive_statuses": dict(sorted(archive_statuses.items())),
            "provenance_statuses": dict(sorted(provenance_statuses.items())),
            "project_module_references": dict(sorted(module_references.items())),
        },
        "known_ambiguities": [
            {
                "id": "missing_run_provenance",
                "status": "unknown",
                "affected_checkpoint_count": provenance_statuses["unknown"],
                "note": (
                    "A checkpoint without an adjacent manifest and resolved_config.yaml "
                    "cannot be mapped reliably to an exact command, wrapper stack, or data split."
                ),
            },
            {
                "id": "stage4b_cbf_training_contradiction",
                "status": "ambiguous",
                "affected_stage_paths": stage4b_paths,
                "note": (
                    "Historical prose describes Stage 4b/HardNet training as using HOCBF in "
                    "the loop, while the audited train.py makes --stage4b enable wind, "
                    "intermittent detection, and domain randomization but not --cbf. "
                    "Directory names are not accepted as proof; the claim needs primary run metadata."
                ),
            },
            {
                "id": "checkpoint_import_path_compatibility",
                "status": "risk",
                "observed_project_module_references": sorted(module_references),
                "note": (
                    "SB3/cloudpickle archives can embed Python module paths. Renaming or moving "
                    "envs, models, observers, or safety may make historical checkpoints unloadable; "
                    "retain compatibility modules or migrate archives in a separately verified step."
                ),
            },
            {
                "id": "unmapped_observation_shapes",
                "status": "unknown",
                "observed_shapes": dict(sorted(observation_dimensions.items())),
                "note": (
                    "The archive shape is observable, but the exact wrapper composition that "
                    "produced it is unknown without a run manifest. Do not infer it from the shape "
                    "or directory name alone."
                ),
            },
        ],
        "duplicate_hash_groups": duplicate_groups,
        "checkpoints": checkpoints,
    }


def format_bytes(byte_count: int) -> str:
    return f"{byte_count:,} B ({byte_count / (1024 * 1024):.1f} MiB)"


def render_markdown(inventory: dict[str, Any]) -> str:
    layout = inventory["layout"]
    summary = inventory["checkpoint_summary"]
    lines = [
        "# Legacy Artifact Inventory",
        "",
        "This file is generated by `python scripts/inventory_repository.py`. It records the",
        "historical `logs/` tree without loading models, moving files, or deleting artifacts.",
        "The companion JSON is the machine-readable source. No generation timestamp is stored,",
        "so an unchanged repository produces byte-for-byte identical output.",
        "",
        "## Scope and headline counts",
        "",
        "| Item | Count |",
        "|---|---:|",
        f"| Immediate directories under `logs/stages/` | {layout['stage_directory_count']} |",
        (
            "| Other immediate directories under `logs/` "
            f"| {layout['top_level_directory_count_excluding_stages']} |"
        ),
        f"| Directories recursively under `logs/` | {layout['recursive_directory_count']} |",
        f"| Distinct checkpoint parent directories | {layout['checkpoint_parent_directory_count']} |",
        f"| Runs with an explicit `manifest.json` | {layout['manifest_backed_run_directory_count']} |",
        f"| Stage directories containing checkpoints | {layout['stage_directories_with_checkpoints']} |",
        (
            "| Stage directories containing evaluation artifacts but no checkpoint "
            f"| {layout['stage_directories_with_eval_artifacts_only']} |"
        ),
        f"| Model ZIP files | {summary['count']} |",
        f"| Model ZIP bytes | {format_bytes(summary['bytes'])} |",
        f"| Unique SHA-256 values | {summary['unique_sha256_count']} |",
        f"| Duplicate SHA-256 groups | {summary['duplicate_hash_group_count']} |",
        "",
        "Here, a *stage directory* means one immediate child of `logs/stages/`. The other",
        "top-level `logs/` directories are reported separately because the legacy layout mixes",
        "shared `models/`, `eval/`, and `tensorboard/` directories with run-like directories;",
        "the inventory does not guess that each is an independent run.",
        "",
        "## Checkpoint metadata",
        "",
        "Metadata is read from SB3 ZIP members (`data` and",
        "`_stable_baselines3_version`) as text/JSON. Serialized fields are never decoded.",
        "",
        "### Observation shapes",
        "",
        "| Shape | Checkpoints |",
        "|---|---:|",
    ]
    for shape, count in summary["observation_shapes"].items():
        lines.append(f"| `{shape}` | {count} |")

    lines.extend(
        [
            "",
            "### Explicit training seeds",
            "",
            "| Seed value | Checkpoints |",
            "|---|---:|",
        ]
    )
    for seed, count in summary["seed_values"].items():
        lines.append(f"| `{seed}` | {count} |")

    lines.extend(
        [
            "",
            "### Stable-Baselines3 versions",
            "",
            "| Version | Checkpoints |",
            "|---|---:|",
        ]
    )
    for version, count in summary["stable_baselines3_versions"].items():
        lines.append(f"| `{version}` | {count} |")

    lines.extend(
        [
            "",
            "### Provenance status",
            "",
            "| Status | Checkpoints |",
            "|---|---:|",
        ]
    )
    for status, count in summary["provenance_statuses"].items():
        lines.append(f"| `{status}` | {count} |")

    lines.extend(
        [
            "",
            "`unknown` means no explicit ancestor `manifest.json` or",
            "`resolved_config.yaml` was found. It must not be replaced with a guess based on a",
            "folder name, report caption, or observation dimension.",
            "",
            "## Known ambiguity and compatibility risks",
            "",
            "- **Stage 4b is ambiguous.** Historical prose describes Stage 4b/HardNet training",
            "  as having HOCBF in the loop. In the audited launcher, `--stage4b` enables wind,",
            "  intermittent detection, and domain randomization but does not itself enable",
            "  `--cbf`. Exact run manifests are absent, so neither account is treated as proven.",
            "- **Import paths are part of checkpoint compatibility.** SB3/cloudpickle archives",
            "  may refer to Python modules by name. In particular, the inventory observes",
            "  `safety.hardnet_policy` in historical metadata. Keep compatibility paths for",
            "  `envs/`, `models/`, `observers/`, and `safety/` when reorganizing the repository.",
            "- **Observation dimensions do not identify a pipeline.** The 18- and 22-element",
            "  shapes, for example, cannot be mapped safely to a wrapper stack from the ZIP",
            "  alone. They remain legacy/unknown until a reproducible loader is demonstrated.",
            "- **Duplicate files are retained.** Matching SHA-256 values identify byte-for-byte",
            "  duplicates, but this inventory does not delete or consolidate them.",
            "",
            "## Stage directory summary",
            "",
            "| Stage directory | ZIPs | Bytes | Eval artifacts | Provenance |",
            "|---|---:|---:|:---:|:---:|",
        ]
    )
    for stage in layout["stages"]:
        lines.append(
            f"| `{stage['path']}` | {stage['checkpoint_count']} "
            f"| {stage['checkpoint_bytes']:,} "
            f"| {'yes' if stage['has_eval_artifacts'] else 'no'} "
            f"| {stage['provenance_status']} |"
        )

    lines.extend(
        [
            "",
            "## Duplicate groups",
            "",
            "The full SHA-256 and path list for every group is stored in",
            "`legacy_artifact_inventory.json`.",
            "",
            "| SHA-256 prefix | Copies |",
            "|---|---:|",
        ]
    )
    for group in inventory["duplicate_hash_groups"]:
        lines.append(f"| `{group['sha256'][:16]}` | {len(group['paths'])} |")
    if not inventory["duplicate_hash_groups"]:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Regeneration and verification",
            "",
            "```bash",
            "python scripts/inventory_repository.py",
            "python scripts/inventory_repository.py --check",
            "```",
            "",
            "`--check` performs a fresh read-only scan and fails if either generated file is",
            "stale. The scan may take a short while because every model ZIP is hashed.",
            "",
        ]
    )
    return "\n".join(lines)


def serialized_outputs(inventory: dict[str, Any]) -> tuple[str, str]:
    json_text = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    markdown_text = render_markdown(inventory)
    return json_text, markdown_text


def check_output(path: Path, expected: str) -> bool:
    try:
        actual = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"missing: {path}", file=sys.stderr)
        return False
    if actual != expected:
        print(f"stale: {path}", file=sys.stderr)
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; fail if committed inventory outputs are stale",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_directory = repo_root / "docs" / "current"
    json_path = output_directory / "legacy_artifact_inventory.json"
    markdown_path = output_directory / "legacy_artifact_inventory.md"

    inventory = build_inventory(repo_root)
    json_text, markdown_text = serialized_outputs(inventory)
    if args.check:
        is_current = check_output(json_path, json_text) & check_output(
            markdown_path, markdown_text
        )
        if is_current:
            print("legacy artifact inventory is current")
            return 0
        return 1

    output_directory.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")
    markdown_path.write_text(markdown_text, encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
