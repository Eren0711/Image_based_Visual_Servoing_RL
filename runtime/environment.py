"""Single environment-construction path for training and evaluation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Callable

import gymnasium as gym

from envs.interception_env import InterceptionEnv


@dataclass(frozen=True)
class EnvironmentOptions:
    """Explicit optional wrapper profile layered over the base environment."""

    use_noise_delay: bool = False
    use_dkf: bool = False
    use_imu_dkf: bool = True
    use_cbf: bool = False
    cbf_method: str = "hocbf"
    cbf_alpha_fov: float = 100.0
    cbf_alpha_att: float = 100.0
    cbf_horizon_fov: int = 1
    cbf_horizon_att: int = 8
    cbf_safety_margin: float = 0.10
    use_wind: bool = False
    use_intermittent_detection: bool = False
    domain_randomize: bool = False
    use_cbf_context: bool = False

    def validate(self) -> None:
        if self.use_dkf and not self.use_noise_delay:
            raise ValueError("DKF requires the noise/delay measurement wrapper")
        if self.use_cbf and self.use_cbf_context:
            raise ValueError(
                "External CBF and in-policy CBF context are mutually exclusive"
            )
        if self.cbf_method not in {"hocbf", "bisection"}:
            raise ValueError(f"Unknown CBF method: {self.cbf_method!r}")
        if self.domain_randomize and not (
            self.use_wind or self.use_intermittent_detection
        ):
            raise ValueError(
                "Domain randomization requires wind and/or intermittent detection"
            )
        if self.cbf_horizon_fov < 1 or self.cbf_horizon_att < 1:
            raise ValueError("CBF horizons must be positive integers")
        if self.cbf_safety_margin < 0.0:
            raise ValueError("CBF safety margin cannot be negative")

    def to_dict(self) -> dict:
        return asdict(self)

    def active_components(self) -> list[str]:
        active = ["interception_env"]
        if self.use_wind:
            active.append("wind")
        if self.use_intermittent_detection:
            active.append("intermittent_detection")
        if self.use_noise_delay:
            active.append("noise_delay")
        if self.use_dkf:
            active.append("dkf")
        if self.use_cbf:
            active.append(f"external_cbf:{self.cbf_method}")
        if self.use_cbf_context:
            active.append("cbf_context")
        return active


def build_environment(
    config: dict,
    options: EnvironmentOptions | None = None,
) -> gym.Env:
    """Construct one environment using the repository's authoritative order.

    Wrapper order, from base toward the policy, is:
    Wind -> intermittent detection -> noise/delay -> DKF -> CBF/context.
    The config is deep-copied so a caller cannot mutate another worker's state.
    """
    resolved = options or EnvironmentOptions()
    resolved.validate()
    config_copy = deepcopy(config)
    env: gym.Env = InterceptionEnv(config=config_copy)

    if resolved.use_wind:
        from envs.wrappers.wind_wrapper import WindWrapper

        wind_cfg = config_copy.get("stage4b", {}).get("wind", {})
        env = WindWrapper(
            env,
            sigma=wind_cfg.get("sigma", 1.0),
            theta=wind_cfg.get("theta", 0.5),
            k_drag=wind_cfg.get("k_drag", 0.1),
            randomize_per_episode=resolved.domain_randomize,
            randomization_ranges=wind_cfg.get("randomization_ranges"),
        )

    if resolved.use_intermittent_detection:
        from envs.wrappers.intermittent_detection_wrapper import (
            IntermittentDetectionWrapper,
        )

        det_cfg = config_copy.get("stage4b", {}).get("detection", {})
        env = IntermittentDetectionWrapper(
            env,
            beta_1=det_cfg.get("beta_1", 8.0),
            beta_2=det_cfg.get("beta_2", 4.0),
            beta_3=det_cfg.get("beta_3", 1.0),
            sigma_base=det_cfg.get("sigma_base", 0.0),
            sigma_slope=det_cfg.get("sigma_slope", 0.0005),
            randomize_per_episode=resolved.domain_randomize,
            randomization_ranges=det_cfg.get("randomization_ranges"),
        )

    if resolved.use_noise_delay:
        from envs.wrappers.noise_delay_wrapper import NoiseDelayWrapper

        noise_cfg = config_copy.get("noise_delay", {})
        env = NoiseDelayWrapper(
            env,
            delay=noise_cfg.get("delay", 3),
            sigma_noise=noise_cfg.get("sigma_noise", 0.03),
        )

    if resolved.use_dkf:
        from envs.wrappers.dkf_wrapper import DKFWrapper

        dkf_cfg = config_copy.get("dkf", {})
        noise_cfg = config_copy.get("noise_delay", {})
        env = DKFWrapper(
            env,
            delay=noise_cfg.get("delay", 3),
            dt=config_copy["interceptor"]["dt"],
            sigma_pos_process=dkf_cfg.get("sigma_pos_process", 0.01),
            sigma_vel_process=dkf_cfg.get("sigma_vel_process", 0.5),
            sigma_measurement=dkf_cfg.get("sigma_measurement", 0.03),
            use_imu=resolved.use_imu_dkf,
        )

    if resolved.use_cbf:
        from envs.wrappers.cbf_wrapper import CBFWrapper

        env = CBFWrapper(
            env,
            method=resolved.cbf_method,
            alpha_fov=resolved.cbf_alpha_fov,
            alpha_attitude=resolved.cbf_alpha_att,
            horizon_fov=resolved.cbf_horizon_fov,
            horizon_attitude=resolved.cbf_horizon_att,
            attitude_safety_margin=resolved.cbf_safety_margin,
            in_fov_only=True,
        )

    if resolved.use_cbf_context:
        from envs.wrappers.cbf_context_wrapper import CBFContextWrapper

        env = CBFContextWrapper(
            env,
            alpha_fov=resolved.cbf_alpha_fov,
            alpha_attitude=resolved.cbf_alpha_att,
            attitude_safety_margin=resolved.cbf_safety_margin,
        )

    return env


def make_environment_factory(
    config: dict,
    options: EnvironmentOptions | None = None,
) -> Callable[[], gym.Env]:
    """Return an SB3-compatible factory with isolated config copies."""
    frozen_config = deepcopy(config)
    frozen_options = options or EnvironmentOptions()
    frozen_options.validate()

    def _factory() -> gym.Env:
        return build_environment(frozen_config, frozen_options)

    return _factory
