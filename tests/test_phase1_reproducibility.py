"""Phase-1 checks for the shared environment and seed contract."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from runtime.environment import EnvironmentOptions, build_environment
from runtime.seeding import derive_seed


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = (
    ROOT / "configs" / "canonical" / "fixed_camera_intercept_v1.yaml"
)


def _exploratory_stochastic_config() -> dict:
    with CANONICAL_CONFIG.open("r", encoding="utf-8") as stream:
        config = deepcopy(yaml.safe_load(stream))

    # The stochastic wrappers are deliberately outside canonical v1. Removing
    # the canonical marker makes this a named test fixture, not a scope drift.
    config.pop("canonical")
    config["noise_delay"] = {"delay": 2, "sigma_noise": 0.03}
    config["stage4b"] = {
        "wind": {
            "sigma": 1.0,
            "theta": 0.5,
            "k_drag": 0.1,
            "randomization_ranges": {
                "sigma": [0.5, 2.0],
                "theta": [0.3, 1.0],
                "k_drag": [0.05, 0.2],
            },
        },
        "detection": {
            "beta_1": 8.0,
            "beta_2": 4.0,
            "beta_3": 1.0,
            "sigma_base": 0.0,
            "sigma_slope": 0.0005,
            "randomization_ranges": {
                "beta_1": [6.0, 12.0],
                "beta_2": [2.0, 6.0],
                "beta_3": [0.0, 2.0],
            },
        },
    }
    return config


def _rollout(env, seed: int, steps: int = 12):
    observation, reset_info = env.reset(seed=seed)
    observations = [observation.copy()]
    stochastic_info = []
    action = np.zeros(4, dtype=np.float32)
    for _ in range(steps):
        observation, reward, terminated, truncated, info = env.step(action)
        observations.append(observation.copy())
        stochastic_info.append(
            (
                float(reward),
                tuple(info.get("wind", {}).get("v_wind", ())),
                bool(info.get("det", {}).get("valid", False)),
            )
        )
        if terminated or truncated:
            break
    return np.asarray(observations), reset_info["seed_bundle"], stochastic_info


def test_namespaced_seeds_are_stable_and_separated():
    assert derive_seed(42, "wind_process") == derive_seed(42, "wind_process")
    assert derive_seed(42, "wind_process") != derive_seed(42, "noise_delay")
    assert derive_seed(42, "wind_process") != derive_seed(43, "wind_process")


def test_shared_builder_uses_one_documented_wrapper_order():
    options = EnvironmentOptions(
        use_noise_delay=True,
        use_wind=True,
        use_intermittent_detection=True,
        domain_randomize=True,
    )
    env = build_environment(_exploratory_stochastic_config(), options)
    names = []
    current = env
    while hasattr(current, "env"):
        names.append(type(current).__name__)
        current = current.env
    names.append(type(current).__name__)
    env.close()

    assert names == [
        "NoiseDelayWrapper",
        "IntermittentDetectionWrapper",
        "WindWrapper",
        "InterceptionEnv",
    ]


def test_full_stochastic_stack_replays_from_the_same_master_seed():
    options = EnvironmentOptions(
        use_noise_delay=True,
        use_wind=True,
        use_intermittent_detection=True,
        domain_randomize=True,
    )
    env = build_environment(_exploratory_stochastic_config(), options)
    first = _rollout(env, 2026)
    replay = _rollout(env, 2026)
    changed = _rollout(env, 2027)
    env.close()

    np.testing.assert_array_equal(first[0], replay[0])
    assert first[1:] == replay[1:]
    assert set(first[1]) >= {
        "scenario",
        "target_guidance",
        "wind_domain_randomization",
        "wind_process",
        "intermittent_detection",
        "noise_delay",
    }
    assert len(set(first[1].values())) == len(first[1])
    assert not np.array_equal(first[0], changed[0])


def test_invalid_wrapper_combinations_fail_before_construction():
    config = _exploratory_stochastic_config()
    for options in (
        EnvironmentOptions(use_dkf=True),
        EnvironmentOptions(use_cbf=True, use_cbf_context=True),
        EnvironmentOptions(domain_randomize=True),
        EnvironmentOptions(cbf_method="unknown"),
    ):
        try:
            build_environment(config, options)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid environment options were accepted")
