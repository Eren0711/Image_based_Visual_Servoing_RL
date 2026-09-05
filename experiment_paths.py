"""Experiment output path helpers.

Centralizes stage-based output directories so training, evaluation,
visualization, and observer diagnostics write to the same layout.
"""

from datetime import datetime, timezone
from pathlib import Path
import re


DEFAULT_STAGE = "stage1a"
DEFAULT_OUTPUT_ROOT = "./logs/stages"
PROJECT_ROOT = Path(__file__).resolve().parent
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def resolve_stage(config: dict, stage_arg: str = None) -> str:
    """Resolve the active stage name from CLI args or config."""
    experiment_cfg = config.get("experiment", {})
    stage = stage_arg or experiment_cfg.get("stage", DEFAULT_STAGE)
    stage = str(stage).strip()
    if not stage:
        raise ValueError("Experiment stage cannot be empty")
    if (
        Path(stage).is_absolute()
        or ".." in Path(stage).parts
        or not _SAFE_ID.fullmatch(stage)
    ):
        raise ValueError(f"Invalid experiment stage name: {stage!r}")
    return stage


def get_output_root(config: dict) -> Path:
    """Return the configured root directory for all stage artifacts."""
    experiment_cfg = config.get("experiment", {})
    root = Path(experiment_cfg.get("output_root", DEFAULT_OUTPUT_ROOT))
    return root if root.is_absolute() else PROJECT_ROOT / root


def create_run_id(stage: str, seed: int, now: datetime | None = None) -> str:
    """Create a collision-resistant, readable run ID in UTC."""
    instant = now or datetime.now(timezone.utc)
    timestamp = instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    safe_stage = re.sub(r"[^A-Za-z0-9._-]+", "-", stage).strip("-_")
    return f"{safe_stage}__seed-{int(seed)}__{timestamp}"


def validate_run_id(run_id: str) -> str:
    value = str(run_id).strip()
    if not _SAFE_ID.fullmatch(value) or ".." in value:
        raise ValueError(f"Invalid run ID: {run_id!r}")
    return value


def get_stage_paths(config: dict, stage_arg: str = None) -> dict:
    """Return standard artifact directories for one experiment stage."""
    stage = resolve_stage(config, stage_arg)
    root = get_output_root(config)
    stage_dir = root / stage
    return {
        "stage": stage,
        "root": root,
        "stage_dir": stage_dir,
        "models": stage_dir / "models",
        "tensorboard": stage_dir / "tensorboard",
        "eval": stage_dir / "eval",
        "videos": stage_dir / "videos",
        "depth_test": stage_dir / "depth_test",
    }


def get_run_paths(
    config: dict,
    *,
    seed: int,
    run_id: str | None = None,
    stage_arg: str | None = None,
) -> dict:
    """Return collision-safe paths for a new training or evaluation run."""
    stage = resolve_stage(config, stage_arg)
    resolved_run_id = validate_run_id(run_id or create_run_id(stage, seed))
    root = get_output_root(config)
    run_dir = root / resolved_run_id
    return {
        "stage": stage,
        "run_id": resolved_run_id,
        "root": root,
        "run_dir": run_dir,
        # Compatibility alias for code that previously consumed stage_dir.
        "stage_dir": run_dir,
        "models": run_dir / "models",
        "tensorboard": run_dir / "tensorboard",
        "eval": run_dir / "evaluation",
        "videos": run_dir / "videos",
        "depth_test": run_dir / "diagnostics" / "depth",
        "manifest": run_dir / "manifest.json",
    }


def ensure_stage_dirs(paths: dict, *keys: str) -> None:
    """Create selected stage artifact directories."""
    for key in keys:
        paths[key].mkdir(parents=True, exist_ok=True)


def default_model_path(paths: dict) -> str:
    """Return the default Stable-Baselines3 model path for a stage."""
    return str(paths["models"] / "ibvs_ppo_final")
