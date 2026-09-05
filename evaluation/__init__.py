"""Canonical, reproducible evaluation protocol for fixed-camera interception."""

from evaluation.runner import run_evaluation, validate_model_environment
from evaluation.suites import EvaluationSuite, Scenario, load_suite

__all__ = [
    "EvaluationSuite",
    "Scenario",
    "load_suite",
    "run_evaluation",
    "validate_model_environment",
]
