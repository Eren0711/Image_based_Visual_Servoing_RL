"""Experiment output path helpers.

Centralizes stage-based output directories so training, evaluation,
visualization, and observer diagnostics write to the same layout.
"""

from pathlib import Path


DEFAULT_STAGE = "stage1a"
DEFAULT_OUTPUT_ROOT = "./logs/stages"


def resolve_stage(config: dict, stage_arg: str = None) -> str:
    """Resolve the active stage name from CLI args or config."""
    experiment_cfg = config.get("experiment", {})
    stage = stage_arg or experiment_cfg.get("stage", DEFAULT_STAGE)
    stage = str(stage).strip()
    if not stage:
        raise ValueError("Experiment stage cannot be empty")
    if Path(stage).is_absolute() or ".." in Path(stage).parts:
        raise ValueError(f"Invalid experiment stage name: {stage!r}")
    return stage


def get_output_root(config: dict) -> Path:
    """Return the configured root directory for all stage artifacts."""
    experiment_cfg = config.get("experiment", {})
    return Path(experiment_cfg.get("output_root", DEFAULT_OUTPUT_ROOT))


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


def ensure_stage_dirs(paths: dict, *keys: str) -> None:
    """Create selected stage artifact directories."""
    for key in keys:
        paths[key].mkdir(parents=True, exist_ok=True)


def default_model_path(paths: dict) -> str:
    """Return the default Stable-Baselines3 model path for a stage."""
    return str(paths["models"] / "ibvs_ppo_final")
