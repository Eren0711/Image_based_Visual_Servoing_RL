"""Canonical project configuration helpers and Phase-0 scope validation."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent
CANONICAL_CONFIG_PATH = (
    ROOT / "configs" / "canonical" / "fixed_camera_intercept_v1.yaml"
)
CANONICAL_ID = "fixed_camera_intercept_v1"
CANONICAL_TARGET_MODES = ["cruise", "steady_turn", "weave"]
CANONICAL_OUTCOME_PRECEDENCE = [
    "success",
    "flight_envelope_violation",
    "fov_loss",
    "timeout",
]
# SHA-256 of the parsed YAML serialized as sorted compact JSON. This freezes
# every semantic value in the canonical file, while deliberately ignoring
# comments and formatting. An intentional v2 must use a new ID and digest.
CANONICAL_CONFIG_DIGEST = (
    "6f58cb299d055f5bd976eb44864317934a89c8791f91e944fd169b537b71cdea"
)
SCOPE_CHANGING_TRAIN_FLAGS = {
    "noise_delay": "--noise-delay",
    "dkf": "--dkf",
    "cbf": "--cbf",
    "wind": "--wind",
    "intermittent_det": "--intermittent-det",
    "domain_randomize": "--domain-randomize",
    "stage4b": "--stage4b",
    "hardnet": "--hardnet",
    "feasibility_coef": "--feasibility-coef",
    "curriculum": "--curriculum",
    "maneuver_curriculum": "--maneuver-curriculum",
}


class ScopeConfigError(ValueError):
    """Raised when a config contradicts the frozen canonical contract."""


def load_yaml_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ScopeConfigError(f"Configuration must be a YAML mapping: {path}")
    return config


def load_canonical_config() -> dict:
    return load_yaml_config(CANONICAL_CONFIG_PATH)


def is_canonical_config(config: Mapping[str, Any]) -> bool:
    return config.get("canonical", {}).get("id") == CANONICAL_ID


def is_canonical_config_path(path: str | Path) -> bool:
    """Return whether a path resolves to the repository's canonical YAML."""
    return Path(path).expanduser().resolve() == CANONICAL_CONFIG_PATH.resolve()


def canonical_config_digest(config: Mapping[str, Any]) -> str:
    """Return a stable digest of all parsed canonical configuration values."""
    payload = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_canonical_config(config: Mapping[str, Any]) -> None:
    """Reject ambiguity or drift in the frozen Phase-0 system definition."""
    if not is_canonical_config(config):
        raise ScopeConfigError(
            f"Expected canonical.id={CANONICAL_ID!r}"
        )

    expected_values = {
        "canonical.status": (config["canonical"].get("status"), "frozen"),
        "interceptor.model": (
            config.get("interceptor", {}).get("model"),
            "multicopter_6dof",
        ),
        "interceptor.v_max": (config.get("interceptor", {}).get("v_max"), 15.0),
        "interceptor.a_max": (config.get("interceptor", {}).get("a_max"), 10.0),
        "interceptor.dt": (config.get("interceptor", {}).get("dt"), 0.02),
        "interceptor.max_pitch_deg": (
            config.get("interceptor", {}).get("max_pitch_deg"),
            35.0,
        ),
        "interceptor.max_roll_deg": (
            config.get("interceptor", {}).get("max_roll_deg"),
            35.0,
        ),
        "target.model": (config.get("target", {}).get("model"), "sixdof"),
        "target.inherit_interceptor_limits": (
            config.get("target", {}).get("inherit_interceptor_limits"),
            False,
        ),
        "target.v_max": (config.get("target", {}).get("v_max"), 10.0),
        "target.a_max": (config.get("target", {}).get("a_max"), 5.0),
        "target.maneuver_modes": (
            config.get("target", {}).get("maneuver_modes"),
            CANONICAL_TARGET_MODES,
        ),
        "target.initial_speed_fraction_range": (
            config.get("target", {}).get("initial_speed_fraction_range"),
            [0.2, 0.5],
        ),
        "camera.model": (
            config.get("camera", {}).get("model"),
            "pinhole_monocular",
        ),
        "camera.mounting": (
            config.get("camera", {}).get("mounting"),
            "rigid_forward",
        ),
        "camera.R_c_b_euler": (
            config.get("camera", {}).get("R_c_b_euler"),
            [0.0, -1.5707963267948966, 0.0],
        ),
        "env.d_success": (config.get("env", {}).get("d_success"), 2.0),
        "env.fov_loss_limit": (
            config.get("env", {}).get("fov_loss_limit"),
            15,
        ),
        "env.max_steps": (config.get("env", {}).get("max_steps"), 500),
        "env.init_distance_range": (
            config.get("env", {}).get("init_distance_range"),
            [10.0, 30.0],
        ),
        "env.terminate_on_attitude_violation": (
            config.get("env", {}).get("terminate_on_attitude_violation"),
            True,
        ),
        "env.attitude_violation_grace_steps": (
            config.get("env", {}).get("attitude_violation_grace_steps"),
            1,
        ),
        "outcome_contract.terminal_precedence": (
            config.get("outcome_contract", {}).get("terminal_precedence"),
            CANONICAL_OUTCOME_PRECEDENCE,
        ),
        "policy_contract.observation.dimension": (
            config.get("policy_contract", {})
            .get("observation", {})
            .get("dimension"),
            16,
        ),
        "policy_contract.action.dimension": (
            config.get("policy_contract", {})
            .get("action", {})
            .get("dimension"),
            4,
        ),
        "training.algorithm": (
            config.get("training", {}).get("algorithm"),
            "PPO",
        ),
        "training.policy": (
            config.get("training", {}).get("policy"),
            "MlpPolicy",
        ),
    }
    errors = [
        f"{name}: expected {expected!r}, got {actual!r}"
        for name, (actual, expected) in expected_values.items()
        if actual != expected
    ]

    pipeline = config.get("pipeline", {})
    disabled_features = (
        "raw_images",
        "noise_delay",
        "dkf_wrapper",
        "intermittent_detection",
        "wind",
        "external_cbf",
        "hardnet",
    )
    for key in disabled_features:
        if pipeline.get(key) is not False:
            errors.append(f"pipeline.{key}: expected false")

    exclusions = set(config.get("canonical", {}).get("exclusions", []))
    required_exclusions = {
        "raw_image_processing",
        "physical_contact_or_capture",
        "equal_agility_impossibility_theorem",
        "gimballed_camera",
        "recurrent_policy",
        "adversarial_multi_agent_training",
    }
    missing_exclusions = sorted(required_exclusions - exclusions)
    if missing_exclusions:
        errors.append(f"canonical.exclusions missing {missing_exclusions}")

    actual_digest = canonical_config_digest(config)
    if actual_digest != CANONICAL_CONFIG_DIGEST:
        errors.append(
            "complete semantic digest: canonical values changed "
            f"(expected {CANONICAL_CONFIG_DIGEST}, got {actual_digest})"
        )

    if errors:
        raise ScopeConfigError(
            "Canonical Phase-0 scope drift detected:\n- " + "\n- ".join(errors)
        )


def active_scope_overrides(args: Any) -> list[str]:
    """Return CLI options that would leave the clean canonical MVP scope."""
    return [
        flag
        for attribute, flag in SCOPE_CHANGING_TRAIN_FLAGS.items()
        if bool(getattr(args, attribute, False))
    ]
