"""Loading and validation for immutable paired evaluation suites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from project_config import CANONICAL_TARGET_MODES
from experiment_paths import validate_run_id


SUITE_SCHEMA_VERSION = 1


class EvaluationSuiteError(ValueError):
    """Raised when a suite cannot define an unambiguous scenario bank."""


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    suite_id: str
    seed: int
    target_mode: str


@dataclass(frozen=True)
class EvaluationSuite:
    schema_version: int
    suite_id: str
    description: str
    deterministic: bool
    paired_across_modes: bool
    target_modes: tuple[str, ...]
    seeds: tuple[int, ...]
    source_path: Path

    @property
    def episode_count(self) -> int:
        return len(self.seeds) * len(self.target_modes)

    def scenarios(self) -> Iterator[Scenario]:
        """Yield a seed-major Cartesian product for paired comparisons."""
        for seed in self.seeds:
            for target_mode in self.target_modes:
                yield Scenario(
                    scenario_id=(
                        f"{self.suite_id}/seed-{seed}/{target_mode}"
                    ),
                    suite_id=self.suite_id,
                    seed=seed,
                    target_mode=target_mode,
                )

    def to_manifest_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "description": self.description,
            "deterministic": self.deterministic,
            "paired_across_modes": self.paired_across_modes,
            "target_modes": list(self.target_modes),
            "seeds": list(self.seeds),
            "episode_count": self.episode_count,
            "source_path": str(self.source_path),
        }


def load_suite(path: str | Path) -> EvaluationSuite:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise EvaluationSuiteError(f"suite must be a YAML mapping: {source}")

    required = {
        "schema_version",
        "suite_id",
        "description",
        "deterministic",
        "paired_across_modes",
        "target_modes",
        "seeds",
    }
    missing = required - set(raw)
    extra = set(raw) - required
    if missing or extra:
        raise EvaluationSuiteError(
            f"suite fields mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    if raw["schema_version"] != SUITE_SCHEMA_VERSION:
        raise EvaluationSuiteError(
            f"unsupported suite schema version {raw['schema_version']!r}"
        )
    suite_id = raw["suite_id"]
    if not isinstance(suite_id, str) or not suite_id.strip():
        raise EvaluationSuiteError("suite_id must be a non-empty string")
    try:
        validate_run_id(suite_id)
    except ValueError as error:
        raise EvaluationSuiteError(f"unsafe suite_id: {suite_id!r}") from error
    description = raw["description"]
    if not isinstance(description, str) or not description.strip():
        raise EvaluationSuiteError("description must be a non-empty string")
    if not isinstance(raw["deterministic"], bool):
        raise EvaluationSuiteError("deterministic must be boolean")
    if raw["paired_across_modes"] is not True:
        raise EvaluationSuiteError(
            "canonical suites must set paired_across_modes: true"
        )

    modes = raw["target_modes"]
    if not isinstance(modes, list) or modes != CANONICAL_TARGET_MODES:
        raise EvaluationSuiteError(
            "target_modes must list all canonical modes in canonical order: "
            f"{CANONICAL_TARGET_MODES}"
        )
    seeds = raw["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise EvaluationSuiteError("seeds must be a non-empty explicit list")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in seeds
    ):
        raise EvaluationSuiteError("all seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise EvaluationSuiteError("suite seeds must be unique")

    suite = EvaluationSuite(
        schema_version=SUITE_SCHEMA_VERSION,
        suite_id=suite_id,
        description=description,
        deterministic=raw["deterministic"],
        paired_across_modes=True,
        target_modes=tuple(modes),
        seeds=tuple(seeds),
        source_path=source,
    )
    scenario_ids = [scenario.scenario_id for scenario in suite.scenarios()]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise EvaluationSuiteError("generated scenario IDs are not unique")
    return suite


def assert_disjoint_suites(*suites: EvaluationSuite) -> None:
    """Reject seed leakage between any pair of named evaluation suites."""
    for index, left in enumerate(suites):
        for right in suites[index + 1 :]:
            overlap = sorted(set(left.seeds) & set(right.seeds))
            if overlap:
                raise EvaluationSuiteError(
                    f"seed leakage between {left.suite_id!r} and "
                    f"{right.suite_id!r}: {overlap}"
                )
