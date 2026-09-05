"""Executable checks for the frozen Phase-0 system contract."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from envs.interception_env import InterceptionEnv
from models.target_6dof import SixDOFTarget
from project_config import (
    ScopeConfigError,
    active_scope_overrides,
    is_canonical_config_path,
    validate_canonical_config,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = (
    ROOT / "configs" / "canonical" / "fixed_camera_intercept_v1.yaml"
)


def load_canonical_config() -> dict:
    with CANONICAL_CONFIG.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_canonical_scope_values_are_frozen():
    config = load_canonical_config()
    validate_canonical_config(config)

    assert config["canonical"]["id"] == "fixed_camera_intercept_v1"
    assert config["canonical"]["status"] == "frozen"
    assert config["interceptor"]["model"] == "multicopter_6dof"
    assert config["target"]["model"] == "sixdof"
    assert config["target"]["maneuver_modes"] == [
        "cruise",
        "steady_turn",
        "weave",
    ]
    assert config["env"]["d_success"] == 2.0
    assert config["env"]["fov_loss_limit"] == 15
    assert config["env"]["max_steps"] == 500
    assert config["env"]["terminate_on_attitude_violation"] is True
    assert config["outcome_contract"]["terminal_precedence"] == [
        "success",
        "flight_envelope_violation",
        "fov_loss",
        "timeout",
    ]
    assert config["policy_contract"]["observation"]["dimension"] == 16
    assert config["policy_contract"]["action"]["dimension"] == 4

    pipeline = config["pipeline"]
    assert pipeline["raw_images"] is False
    assert pipeline["noise_delay"] is False
    assert pipeline["dkf_wrapper"] is False
    assert pipeline["intermittent_detection"] is False
    assert pipeline["wind"] is False
    assert pipeline["external_cbf"] is False
    assert pipeline["hardnet"] is False


def test_canonical_runtime_matches_vehicle_camera_and_space_contracts():
    env = InterceptionEnv(load_canonical_config())
    observation, _ = env.reset(seed=2026)

    assert isinstance(env.target, SixDOFTarget)
    assert env.target.v_max == 10.0
    assert env.target.a_max == 5.0
    assert env.get_target_modes() == ["cruise", "steady_turn", "weave"]
    assert observation.shape == (16,)
    assert env.observation_space.shape == (16,)
    assert env.action_space.shape == (4,)

    # Rigid forward mount: body +x maps to camera +z optical axis.
    body_forward_in_camera = env.camera.R_c_b @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        body_forward_in_camera, np.array([0.0, 0.0, 1.0]), atol=1e-7
    )


def test_canonical_reset_distribution_matches_frozen_initial_conditions():
    env = InterceptionEnv(load_canonical_config())
    seen_modes = set()

    for seed in range(64):
        _, info = env.reset(seed=seed)
        speed = float(np.linalg.norm(env.target.velocity))
        np.testing.assert_array_equal(env.interceptor.position, np.zeros(3))
        assert 10.0 <= info["relative_distance"] <= 30.0
        assert 2.0 <= speed <= 5.0  # 20–50% of the target's 10 m/s limit
        assert info["in_fov"] is True
        assert info["fov_margin"] > 0.2
        seen_modes.add(str(env.target.maneuver_mode))

    assert seen_modes == {"cruise", "steady_turn", "weave"}


def test_canonical_attitude_envelope_is_a_terminal_failure():
    env = InterceptionEnv(load_canonical_config())
    env.reset(seed=2)

    outcome = "running"
    terminated = False
    terminal_reward = 0.0
    for _ in range(50):
        _, terminal_reward, terminated, truncated, info = env.step(
            np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        )
        outcome = info["episode_outcome"]
        if terminated or truncated:
            break

    assert terminated is True
    assert outcome == "flight_envelope_violation"
    assert info["attitude_violation"] is True
    # The explicit terminal penalty prevents early envelope failure from
    # becoming a reward-hacking shortcut.
    assert terminal_reward < -90.0


def test_default_environment_uses_canonical_config():
    env = InterceptionEnv()
    assert env.config["canonical"]["id"] == "fixed_camera_intercept_v1"
    assert env.get_target_modes() == ["cruise", "steady_turn", "weave"]
    assert is_canonical_config_path(CANONICAL_CONFIG)


def test_declared_canonical_config_cannot_bypass_validation_with_id_drift():
    config = deepcopy(load_canonical_config())
    config["canonical"]["id"] = "renamed_without_a_v2_contract"
    try:
        InterceptionEnv(config)
    except ScopeConfigError as error:
        assert "Expected canonical.id" in str(error)
    else:
        raise AssertionError("Canonical ID drift bypassed environment validation")


def test_scope_validator_rejects_target_mode_drift():
    config = deepcopy(load_canonical_config())
    config["target"]["maneuver_modes"].append("break_turn")

    try:
        validate_canonical_config(config)
    except ScopeConfigError as error:
        assert "target.maneuver_modes" in str(error)
    else:
        raise AssertionError("Scope validator accepted a non-canonical mode")


def test_scope_validator_rejects_drift_anywhere_in_frozen_file():
    mutations = (
        ("camera", "alpha_hfov", 0.5),
        ("interceptor", "yaw_rate_max", 2.0),
        ("target", "turn_accel_frac", 0.1),
        ("reward", "w_intercept", 999.0),
    )
    for section, key, value in mutations:
        config = deepcopy(load_canonical_config())
        config[section][key] = value
        try:
            validate_canonical_config(config)
        except ScopeConfigError as error:
            assert "complete semantic digest" in str(error)
        else:
            raise AssertionError(
                f"Scope validator accepted drift in {section}.{key}"
            )

    config = deepcopy(load_canonical_config())
    config["policy_contract"]["observation"]["fields"][0]["indices"] = []
    try:
        validate_canonical_config(config)
    except ScopeConfigError as error:
        assert "complete semantic digest" in str(error)
    else:
        raise AssertionError("Scope validator accepted observation-order drift")


def test_scope_changing_cli_flags_are_detected():
    args = SimpleNamespace(
        wind=True,
        hardnet=False,
        feasibility_coef=None,
        curriculum=False,
        maneuver_curriculum=True,
    )
    assert active_scope_overrides(args) == ["--wind", "--maneuver-curriculum"]
