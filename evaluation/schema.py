"""Versioned episode-record schema used by canonical evaluation."""

from __future__ import annotations

import math
import string
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from project_config import CANONICAL_TARGET_MODES


EPISODE_SCHEMA_VERSION = 1
CANONICAL_OUTCOMES = (
    "success",
    "flight_envelope_violation",
    "fov_loss",
    "timeout",
)


class EvaluationSchemaError(ValueError):
    """Raised when an evaluation record violates the public schema."""


@dataclass(frozen=True)
class EpisodeRecord:
    """One terminal episode result; all units are encoded in field names."""

    schema_version: int
    evaluation_id: str
    experiment_id: str
    method_id: str
    model_id: str
    condition_id: str
    config_sha256: str
    model_sha256: str | None
    episode_id: str
    scenario_id: str
    suite_id: str
    seed: int
    seed_bundle: dict[str, int | None]
    target_mode: str
    deterministic: bool
    outcome: str
    terminated: bool
    truncated: bool
    steps: int
    duration_s: float
    total_reward: float
    initial_distance_m: float
    final_distance_m: float
    min_distance_m: float
    terminal_closure_rate_mps: float
    intercept_time_s: float | None
    fov_retention_fraction: float
    fov_loss_steps: int
    fov_loss_duration_s: float
    max_consecutive_fov_loss_steps: int
    min_fov_margin: float
    image_error_initial: float
    image_error_final: float
    image_error_mean: float
    image_error_rms: float
    image_error_max: float
    max_abs_pitch_deg: float
    max_abs_roll_deg: float
    envelope_violation_steps: int
    envelope_violation_duration_s: float
    projection_metrics_status: str
    projection_active_fraction: float | None
    projection_intervention_l2_mean: float | None
    projection_intervention_l2_max: float | None
    projection_residual_max: float | None
    projection_infeasible_steps: int | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        validate_episode_record(value)
        return value


REQUIRED_EPISODE_FIELDS = frozenset(EpisodeRecord.__dataclass_fields__)


def _finite_number(record: Mapping[str, Any], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationSchemaError(f"{key} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationSchemaError(f"{key} must be finite")
    return number


def _validate_sha256(value: str, key: str) -> None:
    if len(value) != 64 or any(
        character not in string.hexdigits for character in value
    ):
        raise EvaluationSchemaError(
            f"{key} must be a 64-character SHA-256 hex digest"
        )


def validate_episode_record(record: Mapping[str, Any]) -> None:
    """Validate one serialized record before writing or summarizing it."""
    missing = REQUIRED_EPISODE_FIELDS - set(record)
    extra = set(record) - REQUIRED_EPISODE_FIELDS
    if missing or extra:
        raise EvaluationSchemaError(
            f"episode fields mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    if record["schema_version"] != EPISODE_SCHEMA_VERSION:
        raise EvaluationSchemaError(
            f"unsupported episode schema version {record['schema_version']!r}"
        )
    for key in (
        "evaluation_id",
        "experiment_id",
        "method_id",
        "model_id",
        "condition_id",
        "config_sha256",
        "episode_id",
        "scenario_id",
        "suite_id",
    ):
        if not isinstance(record[key], str) or not record[key].strip():
            raise EvaluationSchemaError(f"{key} must be a non-empty string")
    if record["model_sha256"] is not None and (
        not isinstance(record["model_sha256"], str)
        or not record["model_sha256"].strip()
    ):
        raise EvaluationSchemaError("model_sha256 must be a string or null")
    _validate_sha256(record["config_sha256"], "config_sha256")
    if record["model_sha256"] is not None:
        _validate_sha256(record["model_sha256"], "model_sha256")
    if record["target_mode"] not in CANONICAL_TARGET_MODES:
        raise EvaluationSchemaError(
            f"target_mode must be one of {CANONICAL_TARGET_MODES}"
        )
    if record["outcome"] not in CANONICAL_OUTCOMES:
        raise EvaluationSchemaError(
            f"outcome must be one of {CANONICAL_OUTCOMES}"
        )
    if not isinstance(record["seed"], int) or isinstance(record["seed"], bool):
        raise EvaluationSchemaError("seed must be an integer")
    if record["seed"] < 0:
        raise EvaluationSchemaError("seed must be non-negative")
    if not isinstance(record["seed_bundle"], dict):
        raise EvaluationSchemaError("seed_bundle must be a mapping")
    for namespace, seed in record["seed_bundle"].items():
        if not isinstance(namespace, str):
            raise EvaluationSchemaError("seed_bundle keys must be strings")
        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        ):
            raise EvaluationSchemaError(
                f"seed_bundle[{namespace!r}] must be a non-negative integer or null"
            )
    for key in ("deterministic", "terminated", "truncated"):
        if not isinstance(record[key], bool):
            raise EvaluationSchemaError(f"{key} must be boolean")
    if record["terminated"] == record["truncated"]:
        raise EvaluationSchemaError(
            "exactly one of terminated and truncated must be true"
        )
    if record["outcome"] == "timeout" and not record["truncated"]:
        raise EvaluationSchemaError("timeout must be recorded as truncated")
    if record["outcome"] != "timeout" and not record["terminated"]:
        raise EvaluationSchemaError(
            "non-timeout canonical outcomes must be recorded as terminated"
        )

    for key in (
        "steps",
        "fov_loss_steps",
        "max_consecutive_fov_loss_steps",
        "envelope_violation_steps",
    ):
        if not isinstance(record[key], int) or isinstance(record[key], bool):
            raise EvaluationSchemaError(f"{key} must be an integer")
        if record[key] < (1 if key == "steps" else 0):
            raise EvaluationSchemaError(f"{key} is outside its valid range")
    if record["max_consecutive_fov_loss_steps"] > record["steps"]:
        raise EvaluationSchemaError(
            "max_consecutive_fov_loss_steps cannot exceed steps"
        )
    if record["envelope_violation_steps"] > record["steps"]:
        raise EvaluationSchemaError("envelope_violation_steps cannot exceed steps")
    if record["fov_loss_steps"] > record["steps"]:
        raise EvaluationSchemaError("fov_loss_steps cannot exceed steps")

    nonnegative_fields = (
        "duration_s",
        "initial_distance_m",
        "final_distance_m",
        "min_distance_m",
        "fov_loss_duration_s",
        "image_error_initial",
        "image_error_final",
        "image_error_mean",
        "image_error_rms",
        "image_error_max",
        "max_abs_pitch_deg",
        "max_abs_roll_deg",
        "envelope_violation_duration_s",
    )
    numbers = {key: _finite_number(record, key) for key in nonnegative_fields}
    _finite_number(record, "total_reward")
    if any(value < 0.0 for value in numbers.values()):
        raise EvaluationSchemaError(
            "distance/time/error/attitude metrics cannot be negative"
        )
    retention = _finite_number(record, "fov_retention_fraction")
    if not 0.0 <= retention <= 1.0:
        raise EvaluationSchemaError("fov_retention_fraction must be in [0, 1]")
    if numbers["min_distance_m"] > min(
        numbers["initial_distance_m"], numbers["final_distance_m"]
    ) + 1e-9:
        raise EvaluationSchemaError(
            "min_distance_m cannot exceed initial or final distance"
        )
    if numbers["image_error_max"] + 1e-9 < max(
        numbers["image_error_initial"], numbers["image_error_final"],
        numbers["image_error_mean"], numbers["image_error_rms"],
    ):
        raise EvaluationSchemaError("image_error_max is inconsistent")
    _finite_number(record, "terminal_closure_rate_mps")
    _finite_number(record, "min_fov_margin")

    projection_status = record["projection_metrics_status"]
    if projection_status not in {"not_applicable", "partial", "unavailable"}:
        raise EvaluationSchemaError("invalid projection_metrics_status")
    projection_float_fields = (
        "projection_active_fraction",
        "projection_intervention_l2_mean",
        "projection_intervention_l2_max",
        "projection_residual_max",
    )
    for key in projection_float_fields:
        value = record[key]
        if value is not None and _finite_number(record, key) < 0.0:
            raise EvaluationSchemaError(f"{key} cannot be negative")
    projection_infeasible = record["projection_infeasible_steps"]
    if projection_infeasible is not None and (
        not isinstance(projection_infeasible, int)
        or isinstance(projection_infeasible, bool)
        or not 0 <= projection_infeasible <= record["steps"]
    ):
        raise EvaluationSchemaError(
            "projection_infeasible_steps must be null or an in-range integer"
        )
    if record["projection_active_fraction"] is not None and not (
        0.0 <= record["projection_active_fraction"] <= 1.0
    ):
        raise EvaluationSchemaError("projection_active_fraction must be in [0, 1]")
    if projection_status != "partial" and any(
        record[key] is not None
        for key in (*projection_float_fields, "projection_infeasible_steps")
    ):
        raise EvaluationSchemaError(
            "projection metrics require projection_metrics_status='partial'"
        )

    intercept_time = record["intercept_time_s"]
    if record["outcome"] == "success":
        if (
            intercept_time is None
            or _finite_number(record, "intercept_time_s") < 0.0
        ):
            raise EvaluationSchemaError("success requires intercept_time_s")
    elif intercept_time is not None:
        raise EvaluationSchemaError(
            "intercept_time_s must be null for unsuccessful episodes"
        )
