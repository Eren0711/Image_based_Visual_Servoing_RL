"""Command-line interface for the canonical evaluation protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from evaluation.runner import run_evaluation
from evaluation.suites import load_suite
from project_config import load_yaml_config, validate_canonical_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "canonical" / "fixed_camera_intercept_v1.yaml"
DEFAULT_SUITE = ROOT / "evaluation" / "suites" / "validation_v1.yaml"


def _resolve_repository_input(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    return (ROOT / candidate).resolve()


def _resolve_model_input(value: str) -> Path:
    candidate = Path(value).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(ROOT / candidate)
    for base in candidates:
        if base.is_file():
            return base.resolve()
        zip_candidate = Path(f"{base}.zip")
        if zip_candidate.is_file():
            return zip_candidate.resolve()
    return (candidate if candidate.is_absolute() else ROOT / candidate).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired, seed-locked canonical drone-interception evaluation."
        )
    )
    parser.add_argument("--model", required=True, help="PPO checkpoint (.zip optional)")
    parser.add_argument(
        "--suite",
        default=str(DEFAULT_SUITE),
        help="Versioned evaluation suite YAML",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Frozen canonical system YAML",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New output directory; it must not already exist",
    )
    parser.add_argument("--run-id", help="Optional explicit evaluation identifier")
    parser.add_argument(
        "--experiment-id",
        default="clean_ppo_interception_mvp_v1",
        help="Stable experiment identifier stored in every episode record",
    )
    parser.add_argument(
        "--method-id",
        default="M1_ppo",
        help="Method identifier stored in every episode record",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Logical model identifier (default: checkpoint filename)",
    )
    parser.add_argument(
        "--condition-id",
        default="clean",
        help="Evaluation condition identifier",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (the suite default is deterministic)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Stable-Baselines3 device selection (default: auto)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _resolve_repository_input(args.config)
    suite_path = _resolve_repository_input(args.suite)
    model_path = _resolve_model_input(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {args.model}")

    config = load_yaml_config(config_path)
    validate_canonical_config(config)
    suite = load_suite(suite_path)

    # Lazy import keeps suite/schema tooling usable without the training stack.
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device=args.device)
    summary = run_evaluation(
        model=model,
        config=config,
        suite=suite,
        output_dir=args.output,
        deterministic=False if args.stochastic else None,
        evaluation_id=args.run_id,
        source_config=config_path,
        source_model=model_path,
        experiment_id=args.experiment_id,
        method_id=args.method_id,
        model_id=args.model_id or model_path.stem,
        condition_id=args.condition_id,
        command=[sys.executable, *sys.argv] if argv is None else [
            sys.executable,
            "-m",
            "evaluation",
            *argv,
        ],
    )
    overall = summary["overall"]
    print(
        f"Evaluation complete: {summary['evaluation_id']} | "
        f"episodes={overall['episode_count']} | "
        f"success_rate={overall['outcome_rates']['success']:.3f}"
    )
    print(f"Artifacts: {Path(args.output).expanduser().resolve()}")
    return 0
