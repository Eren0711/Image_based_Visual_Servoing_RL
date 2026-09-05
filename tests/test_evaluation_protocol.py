"""Acceptance tests for the versioned, paired Phase-1 evaluator."""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from evaluation.metrics import summarize_jsonl
from evaluation.runner import run_evaluation
from evaluation.suites import assert_disjoint_suites, load_suite


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SUITE = ROOT / "evaluation" / "suites" / "validation_v1.yaml"
TEST_SUITE = ROOT / "evaluation" / "suites" / "test_v1.yaml"


class _TinyEvaluationEnv(gym.Env):
    """Two-step deterministic environment exposing the canonical info API."""

    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def __init__(self):
        super().__init__()
        self.target_modes = ["cruise"]
        self.target_mode = "cruise"
        self.steps = 0

    def set_target_modes(self, modes):
        self.target_modes = list(modes)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.target_mode = self.target_modes[0]
        self.steps = 0
        observation = np.zeros(16, dtype=np.float32)
        return observation, {
            "target_mode": self.target_mode,
            "relative_distance": 4.0,
            "image_error": 0.2,
            "in_fov": True,
            "fov_margin": 0.5,
            "seed_bundle": {"scenario": seed, "target_guidance": seed + 1},
        }

    def step(self, action):
        self.steps += 1
        terminated = self.steps == 2
        distance = 4.0 - self.steps
        info = {
            "target_mode": self.target_mode,
            "relative_distance": distance,
            "image_error": 0.2 / (self.steps + 1),
            "in_fov": self.steps == 1,
            "fov_margin": 0.25 if self.steps == 1 else -0.10,
            "pitch_deg": float(self.steps),
            "roll_deg": float(self.steps) / 2.0,
            "attitude_violation": False,
            "episode_outcome": "success" if terminated else "running",
        }
        return (
            np.full(16, self.steps / 10.0, dtype=np.float32),
            1.0,
            terminated,
            False,
            info,
        )


class _ZeroPolicy:
    observation_space = _TinyEvaluationEnv.observation_space
    action_space = _TinyEvaluationEnv.action_space

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        return np.zeros(4, dtype=np.float32), None


def _tiny_config() -> dict:
    return {
        "policy_contract": {
            "observation": {"dimension": 16},
            "action": {"dimension": 4},
        },
        "env": {"max_steps": 3},
        "interceptor": {"dt": 0.02},
    }


def _write_smoke_suite(tmp_path: Path, suite_id: str = "protocol_smoke_v1"):
    path = tmp_path / f"{suite_id}.yaml"
    path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                f"suite_id: {suite_id}",
                "description: Fast canonical evaluator test fixture.",
                "deterministic: true",
                "paired_across_modes: true",
                "target_modes: [cruise, steady_turn, weave]",
                "seeds: [10000]",
                "",
            )
        ),
        encoding="utf-8",
    )
    return load_suite(path)


def test_committed_suites_are_paired_balanced_and_disjoint():
    validation = load_suite(VALIDATION_SUITE)
    held_out = load_suite(TEST_SUITE)
    assert_disjoint_suites(validation, held_out)

    assert validation.seeds == tuple(range(10_000, 10_050))
    assert held_out.seeds == tuple(range(20_000, 20_200))
    assert validation.episode_count == 150
    assert held_out.episode_count == 600
    for suite in (validation, held_out):
        scenarios = list(suite.scenarios())
        assert len({item.scenario_id for item in scenarios}) == len(scenarios)
        for seed in suite.seeds:
            assert [
                item.target_mode for item in scenarios if item.seed == seed
            ] == ["cruise", "steady_turn", "weave"]


def test_runner_writes_manifest_jsonl_and_jsonl_derived_summary(tmp_path):
    suite = _write_smoke_suite(tmp_path)
    source_model = tmp_path / "policy.zip"
    source_model.write_bytes(b"test-policy")
    output_dir = tmp_path / "evaluation"
    summary = run_evaluation(
        model=_ZeroPolicy(),
        config=_tiny_config(),
        suite=suite,
        output_dir=output_dir,
        environment_factory=_TinyEvaluationEnv,
        evaluation_id="protocol-smoke",
        command=["pytest"],
        source_model=source_model,
    )

    episodes_path = output_dir / "episodes.jsonl"
    records = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    assert len(records) == 3
    assert {record["target_mode"] for record in records} == {
        "cruise",
        "steady_turn",
        "weave",
    }
    assert all(record["outcome"] == "success" for record in records)
    assert all(record["steps"] == 2 for record in records)
    assert all(record["fov_retention_fraction"] == 0.5 for record in records)
    assert all(record["fov_loss_steps"] == 1 for record in records)
    assert all(record["terminal_closure_rate_mps"] == 50.0 for record in records)
    assert all(record["projection_metrics_status"] == "not_applicable" for record in records)
    assert all(record["config_sha256"] for record in records)
    assert all(record["model_sha256"] for record in records)
    assert summary == summarize_jsonl(episodes_path)

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["source_model"]["sha256"] == records[0]["model_sha256"]
    assert manifest["runtime_options"]["suite"]["episode_count"] == 3
    assert manifest["evaluation"]["episodes_sha256"]
    assert (output_dir / "resolved_config.yaml").is_file()
    assert (output_dir / "resolved_suite.yaml").is_file()
    assert (output_dir / "summary.json").is_file()

    replay_dir = tmp_path / "evaluation-replay"
    run_evaluation(
        model=_ZeroPolicy(),
        config=_tiny_config(),
        suite=suite,
        output_dir=replay_dir,
        environment_factory=_TinyEvaluationEnv,
        evaluation_id="protocol-smoke",
        command=["pytest"],
        source_model=source_model,
    )
    assert episodes_path.read_bytes() == (replay_dir / "episodes.jsonl").read_bytes()
    assert (output_dir / "summary.json").read_bytes() == (
        replay_dir / "summary.json"
    ).read_bytes()


def test_evaluation_output_directory_cannot_be_reused(tmp_path):
    suite = _write_smoke_suite(tmp_path, "collision-test")
    output_dir = tmp_path / "immutable"
    run_evaluation(
        model=_ZeroPolicy(),
        config=_tiny_config(),
        suite=suite,
        output_dir=output_dir,
        environment_factory=_TinyEvaluationEnv,
        evaluation_id="first",
    )
    try:
        run_evaluation(
            model=_ZeroPolicy(),
            config=_tiny_config(),
            suite=suite,
            output_dir=output_dir,
            environment_factory=_TinyEvaluationEnv,
            evaluation_id="second",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("Evaluation records were allowed to mix")
