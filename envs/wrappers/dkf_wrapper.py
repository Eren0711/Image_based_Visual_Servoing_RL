"""
DKF Observation Wrapper — Stage 2b
=====================================
Gymnasium ObservationWrapper that filters the noisy/delayed image
observations through the Disturbance Kalman Filter (DKF).

This wrapper sits on top of the NoiseDelayWrapper:
  Base env → NoiseDelayWrapper → DKFWrapper → RL agent

The DKF receives the noisy, delayed image position and produces
a filtered estimate of the CURRENT image position and velocity.
The filtered values replace the corrupted obs[0:4].

The key benefit: the DKF compensates for both noise AND delay,
giving the agent a much better estimate of where the target IS NOW
(not where it WAS D steps ago).
"""

import numpy as np
import gymnasium as gym

from observers.dkf import DisturbanceKalmanFilter


class DKFWrapper(gym.ObservationWrapper):
    """Filter image observations through the Disturbance Kalman Filter.

    Replaces noisy/delayed image position and velocity observations
    with DKF-filtered estimates.

    Args:
        env:   Gymnasium environment (typically wrapped by NoiseDelayWrapper).
        delay: Measurement delay in timesteps (must match NoiseDelayWrapper).
        dt:    Timestep duration.
        sigma_pos_process:   DKF position process noise std.
        sigma_vel_process:   DKF velocity process noise std.
        sigma_measurement:   DKF measurement noise std (should match noise wrapper).
    """

    def __init__(
        self,
        env: gym.Env,
        delay: int = 3,
        dt: float = 0.02,
        sigma_pos_process: float = 0.01,
        sigma_vel_process: float = 0.5,
        sigma_measurement: float = 0.03,
    ):
        super().__init__(env)
        self.delay = delay
        self.dt = dt

        # Cache FOV params before instantiating the DKF (needed for unit scaling)
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'camera'):
            self._fov_params = unwrapped.camera.get_fov_params()
        else:
            self._fov_params = None

        # --- Unit conversion on sigma_measurement ---
        # NoiseDelayWrapper adds noise in FOV-normalized [-1, 1] coords, but the
        # DKF operates internally on raw p_bar = (x_c/z_c, y_c/z_c). The two are
        # related by p_bar = obs_normalized * tan(half_fov). Without this
        # conversion the DKF over-estimates measurement variance by ~2×, which
        # makes the Kalman gain too small and lags the target estimate.
        # Use geometric mean of tan_half_hfov and tan_half_vfov for the scalar R.
        if self._fov_params is not None:
            tan_h = self._fov_params['tan_half_hfov']
            tan_v = self._fov_params['tan_half_vfov']
            self._meas_scale = float(np.sqrt(tan_h * tan_v))
            sigma_meas_raw = sigma_measurement * self._meas_scale
        else:
            self._meas_scale = 1.0
            sigma_meas_raw = sigma_measurement

        self.dkf = DisturbanceKalmanFilter(
            dt=dt,
            delay=delay,
            sigma_pos_process=sigma_pos_process,
            sigma_vel_process=sigma_vel_process,
            sigma_measurement=sigma_meas_raw,
        )

    def reset(self, **kwargs):
        """Reset the wrapper, DKF, and underlying environment."""
        obs, info = self.env.reset(**kwargs)

        # Mark info
        info['dkf_active'] = True

        # Initialize DKF with the initial (possibly noisy) observation
        p_bar_init = self._denormalize_p_bar(obs[0:2])
        self.dkf.reset(p_bar_init=p_bar_init)

        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Filter image observations through the DKF.

        1. Extract the (noisy, delayed) image position from obs[0:2]
        2. Denormalize from FOV coords to raw p_bar
        3. Feed through DKF: predict + update
        4. Get filtered current-time estimate
        5. Renormalize and replace obs[0:4]

        Args:
            obs: Observation (possibly noisy/delayed). Shape (16,).

        Returns:
            np.ndarray (16,): Observation with DKF-filtered image data.
        """
        obs = obs.copy()

        # Extract noisy delayed measurement
        p_bar_noisy = self._denormalize_p_bar(obs[0:2])

        # Check if target is in FOV (obs[4])
        in_fov = obs[4] > 0.5

        # DKF step
        if in_fov:
            dkf_result = self.dkf.step(z=p_bar_noisy)
        else:
            # No measurement — prediction only
            dkf_result = self.dkf.step(z=None)

        # Replace image position with DKF estimate
        p_bar_filtered = dkf_result['p_bar']
        dp_bar_filtered = dkf_result['dp_bar']

        # Renormalize back to FOV coordinates
        obs[0:2] = self._normalize_p_bar(p_bar_filtered)

        # Replace image velocity with DKF estimate
        max_dp = 10.0
        obs[2] = np.clip(dp_bar_filtered[0] / max_dp, -1.0, 1.0).astype(np.float32)
        obs[3] = np.clip(dp_bar_filtered[1] / max_dp, -1.0, 1.0).astype(np.float32)

        return obs

    def step(self, action):
        """Step the environment and filter observation through DKF."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self.observation(obs)

        # Add DKF diagnostics to info
        dkf_state = self.dkf.get_state()
        info['dkf_p_bar'] = dkf_state[:2].copy()
        info['dkf_dp_bar'] = dkf_state[2:].copy()
        info['dkf_P_diag'] = np.diag(self.dkf.P).copy()

        return obs, reward, terminated, truncated, info

    def _denormalize_p_bar(self, p_bar_normalized: np.ndarray) -> np.ndarray:
        """Convert FOV-normalized image coordinates back to raw p_bar.

        In the base env: obs[0] = p_bar_x / tan_half_hfov
        Here we reverse: p_bar_x = obs[0] * tan_half_hfov

        Args:
            p_bar_normalized: FOV-normalized coords. Shape (2,).

        Returns:
            np.ndarray (2,): Raw normalized image coords [x_c/z_c, y_c/z_c].
        """
        if self._fov_params is not None:
            tan_h = self._fov_params['tan_half_hfov']
            tan_v = self._fov_params['tan_half_vfov']
            return np.array([
                p_bar_normalized[0] * tan_h,
                p_bar_normalized[1] * tan_v,
            ])
        else:
            # Fallback: assume fov_params not available
            return p_bar_normalized.copy()

    def _normalize_p_bar(self, p_bar_raw: np.ndarray) -> np.ndarray:
        """Convert raw p_bar back to FOV-normalized coordinates.

        Args:
            p_bar_raw: Raw normalized image coords. Shape (2,).

        Returns:
            np.ndarray (2,): FOV-normalized coords, clipped to [-1, 1].
        """
        if self._fov_params is not None:
            tan_h = self._fov_params['tan_half_hfov']
            tan_v = self._fov_params['tan_half_vfov']
            return np.clip(np.array([
                p_bar_raw[0] / tan_h,
                p_bar_raw[1] / tan_v,
            ], dtype=np.float32), -1.0, 1.0)
        else:
            return np.clip(p_bar_raw.astype(np.float32), -1.0, 1.0)
