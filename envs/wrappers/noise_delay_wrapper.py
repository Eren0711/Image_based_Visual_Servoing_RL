"""
Noise + Delay Wrapper — Stage 2b
==================================
Gymnasium ObservationWrapper that corrupts the image-plane observations
with Gaussian noise and introduces a fixed measurement delay.

This simulates realistic camera processing pipeline latency:
  - Image detection takes D timesteps to process
  - Detected coordinates have noise from detector imprecision

The wrapper modifies ONLY the image-related observation components:
  obs[0:2] — image position (p̄_x, p̄_y): delayed + noisy
  obs[2:4] — image velocity (dp̄/dt):    recomputed from noisy/delayed p̄

All other observation components (body velocity, euler angles, depth estimate,
previous action) are left unchanged — they come from the IMU / internal state.

Architecture:
  Base env → NoiseDelayWrapper → [optional DKFWrapper] → RL agent
"""

import numpy as np
import gymnasium as gym
from collections import deque


class NoiseDelayWrapper(gym.ObservationWrapper):
    """Add measurement noise and delay to image observations.

    Simulates realistic sensor imperfections:
      - D-step delay (camera processing latency)
      - Gaussian noise on image coordinates

    The wrapper maintains a FIFO buffer of past clean image observations
    and outputs the delayed + noisy version.

    Args:
        env: Base Gymnasium environment (InterceptionEnv).
        delay: Measurement delay in timesteps (D). Default 3 = 60ms at 50Hz.
        sigma_noise: Std of Gaussian noise on normalized image coords.
                     Default 0.03 (roughly 3% of FOV width).
        seed: Random seed for noise generation.
    """

    def __init__(
        self,
        env: gym.Env,
        delay: int = 3,
        sigma_noise: float = 0.03,
        seed: int = None,
    ):
        super().__init__(env)
        self.delay = delay
        self.sigma_noise = sigma_noise

        # Buffer for delayed image observations: stores (p_bar_normalized)
        # Each entry is [obs[0], obs[1]] (the normalized image position)
        self._p_bar_buffer = deque(maxlen=delay + 1)

        # Previous delayed+noisy p_bar for velocity recomputation
        self._prev_noisy_p_bar = np.zeros(2)

        # RNG for noise
        self._noise_rng = np.random.default_rng(seed)

        # Observation space unchanged (same shape and bounds)

    def reset(self, **kwargs):
        """Reset the wrapper and the underlying environment."""
        obs, info = self.env.reset(**kwargs)

        # Clear buffer and fill with initial observation
        self._p_bar_buffer.clear()
        initial_p_bar = obs[0:2].copy()
        for _ in range(self.delay + 1):
            self._p_bar_buffer.append(initial_p_bar.copy())

        self._prev_noisy_p_bar = initial_p_bar.copy()

        # Reseed noise RNG
        if hasattr(self.env, 'np_random'):
            seed = int(self.env.np_random.integers(0, 2**31))
            self._noise_rng = np.random.default_rng(seed)

        # Mark the info with wrapper metadata
        info['noise_delay_active'] = True
        info['delay_steps'] = self.delay
        info['sigma_noise'] = self.sigma_noise

        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply noise and delay to image components of the observation.

        Args:
            obs: Original observation from the base environment. Shape (16,).

        Returns:
            np.ndarray (16,): Modified observation with noisy/delayed image data.
        """
        obs = obs.copy()

        # --- Store current clean image position ---
        current_p_bar = obs[0:2].copy()
        self._p_bar_buffer.append(current_p_bar)

        # --- Retrieve delayed image position ---
        # The oldest entry in the buffer is from D steps ago
        delayed_p_bar = self._p_bar_buffer[0].copy()

        # --- Add Gaussian noise ---
        noise = self._noise_rng.normal(0.0, self.sigma_noise, size=2)
        noisy_delayed_p_bar = delayed_p_bar + noise.astype(np.float32)

        # Clip to valid range
        noisy_delayed_p_bar = np.clip(noisy_delayed_p_bar, -1.0, 1.0)

        # --- Update observation ---
        obs[0:2] = noisy_delayed_p_bar

        # --- Recompute image velocity from noisy/delayed data ---
        dt = self.env.dt
        dp_bar = (noisy_delayed_p_bar - self._prev_noisy_p_bar) / dt
        max_dp = 10.0
        obs[2] = np.clip(dp_bar[0] / max_dp, -1.0, 1.0).astype(np.float32)
        obs[3] = np.clip(dp_bar[1] / max_dp, -1.0, 1.0).astype(np.float32)

        # Store for next step
        self._prev_noisy_p_bar = noisy_delayed_p_bar.copy()

        return obs

    def step(self, action):
        """Step the environment and apply noise/delay to observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add ground-truth vs noisy comparison to info
        info['p_bar_clean'] = obs[0:2].copy()

        obs = self.observation(obs)

        info['p_bar_noisy_delayed'] = obs[0:2].copy()

        return obs, reward, terminated, truncated, info
