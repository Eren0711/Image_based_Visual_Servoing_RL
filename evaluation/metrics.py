"""Aggregate canonical summaries exclusively from committed JSONL records."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from evaluation.schema import (
    CANONICAL_OUTCOMES,
    EPISODE_SCHEMA_VERSION,
    validate_episode_record,
)
from runtime.manifest import file_sha256


SUMMARY_SCHEMA_VERSION = 1


def read_episode_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {source}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"record at {source}:{line_number} is not an object")
            validate_episode_record(record)
            records.append(record)
    if not records:
        raise ValueError(f"no episode records found in {source}")
    return records


def _mean(records: Iterable[dict[str, Any]], key: str) -> float:
    return float(statistics.fmean(float(record[key]) for record in records))


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    outcomes = Counter(record["outcome"] for record in records)
    success_times = [
        float(record["intercept_time_s"])
        for record in records
        if record["intercept_time_s"] is not None
    ]
    metric_keys = (
        "steps",
        "duration_s",
        "total_reward",
        "initial_distance_m",
        "final_distance_m",
        "min_distance_m",
        "terminal_closure_rate_mps",
        "fov_retention_fraction",
        "fov_loss_steps",
        "fov_loss_duration_s",
        "max_consecutive_fov_loss_steps",
        "min_fov_margin",
        "image_error_mean",
        "image_error_rms",
        "image_error_max",
        "max_abs_pitch_deg",
        "max_abs_roll_deg",
        "envelope_violation_steps",
        "envelope_violation_duration_s",
    )
    projection_float_keys = (
        "projection_active_fraction",
        "projection_intervention_l2_mean",
        "projection_intervention_l2_max",
        "projection_residual_max",
    )
    projection_means = {}
    for key in projection_float_keys:
        values = [
            float(record[key])
            for record in records
            if record[key] is not None
        ]
        projection_means[key] = (
            float(statistics.fmean(values)) if values else None
        )
    infeasible_values = [
        int(record["projection_infeasible_steps"])
        for record in records
        if record["projection_infeasible_steps"] is not None
    ]
    projection_means["projection_infeasible_steps"] = (
        float(statistics.fmean(infeasible_values))
        if infeasible_values else None
    )

    return {
        "episode_count": count,
        "outcome_counts": {
            outcome: int(outcomes.get(outcome, 0))
            for outcome in CANONICAL_OUTCOMES
        },
        "outcome_rates": {
            outcome: float(outcomes.get(outcome, 0) / count)
            for outcome in CANONICAL_OUTCOMES
        },
        "metric_means": {
            key: _mean(records, key)
            for key in metric_keys
        },
        "successful_intercept_time_mean_s": (
            float(statistics.fmean(success_times)) if success_times else None
        ),
        "projection_status_counts": dict(
            sorted(
                Counter(
                    record["projection_metrics_status"] for record in records
                ).items()
            )
        ),
        "projection_metric_means": projection_means,
    }


def summarize_jsonl(path: str | Path) -> dict[str, Any]:
    """Build a summary by reopening JSONL; no rollout state is accepted."""
    source = Path(path)
    records = read_episode_jsonl(source)
    evaluation_ids = {record["evaluation_id"] for record in records}
    suite_ids = {record["suite_id"] for record in records}
    if len(evaluation_ids) != 1 or len(suite_ids) != 1:
        raise ValueError("all JSONL records must belong to one evaluation and suite")

    by_mode: dict[str, Any] = {}
    for mode in sorted({record["target_mode"] for record in records}):
        by_mode[mode] = _aggregate(
            [record for record in records if record["target_mode"] == mode]
        )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "episode_schema_version": EPISODE_SCHEMA_VERSION,
        "evaluation_id": next(iter(evaluation_ids)),
        "suite_id": next(iter(suite_ids)),
        "source_jsonl": {
            "path": source.name,
            "sha256": file_sha256(source),
        },
        "overall": _aggregate(records),
        "by_target_mode": by_mode,
    }


def write_summary_from_jsonl(
    episodes_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    summary = summarize_jsonl(episodes_path)
    destination = Path(summary_path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return summary
