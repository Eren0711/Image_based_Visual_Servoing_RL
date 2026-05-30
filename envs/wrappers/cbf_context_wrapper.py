"""
CBF Context Wrapper — Stage 4a Phase 4 (HardNet)
=================================================
Appends the per-step CBF constraint coefficients (A, b) to the observation
so the HardNet policy's differentiable projection layer can consume them.

Augmented observation:
    obs_aug = [ base_obs (16) | A.flatten() (k*m) | b (k) ]

with the standard k=4 constraints (hfov, vfov, pitch, roll) and m=4 action
dims → 16 + 16 + 4 = 36 dims.

A and b encode the (normalized-action-space) CBF half-space constraints
    A_i · u ≥ b_i,    A = (scaled L_g h),   b = −L_f h − α·h
computed via the analytical proxy Lie derivatives (safety/cbf_lie.py), the
same ones the external HOCBF-QP filter uses. A is scaled by the action
ranges [a_max, a_max, a_max, yaw_rate_max] so the constraint is expressed in
the policy's normalized [-1, 1] action units.

Placement: OUTERMOST observation wrapper, after the DKF wrapper, so the
16-D filtered observation gets the 20-D context appended. The context is
computed from the env's privileged true state (pitch/roll/depth/etc.),
exactly like the HOCBF filter — this is legitimate because the CBF is a
safety supervisor, not part of the agent's perception.

    env → Wind → IntermittentDet → NoiseDelay → DKF → CBFContext → HardNet policy

No external action filter is needed in this configuration: safety is
enforced inside the policy by the projection layer.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from safety.cbf_lie import compute_lie_derivatives, state_from_env

_KEYS = ['hfov', 'vfov', 'pitch', 'roll']


class CBFContextWrapper(gym.Wrapper):
    """Append CBF (A, b) coefficients to the observation for HardNet.

    Args:
        env:        a Gymnasium env (typically the DKF-wrapped stack).
        alpha_fov:  class-K rate for the FOV CBF constraints (1/s).
        alpha_attitude: class-K rate for pitch/roll constraints (1/s).
        attitude_safety_margin: inner margin (rad) on max_pitch/max_roll,
            absorbing the proxy↔true dynamics mismatch (see cbf_lie.py).
        tau_rate:   attitude tracking time constant for the proxy. If None,
            read from the interceptor (falls back to 0.05).
    """

    def __init__(
        self,
        env: gym.Env,
        alpha_fov: float = 100.0,
        alpha_attitude: float = 100.0,
        attitude_safety_margin: float = 0.10,
        tau_rate: float = None,
    ):
        super().__init__(env)
        base = self.env.unwrapped
        self._base = base
        cam_fov = base.camera.get_fov_params()
        if tau_rate is None:
            tau_rate = float(getattr(base.interceptor, 'tau_rate', 0.05))
        self._params = {
            'a_max': float(base.a_max),
            'yaw_rate_max': float(base.yaw_rate_max),
            'dt': float(base.dt),
            'tan_half_hfov': cam_fov['tan_half_hfov'],
            'tan_half_vfov': cam_fov['tan_half_vfov'],
            'max_pitch': float(base.max_pitch),
            'max_roll': float(base.max_roll),
            'tau_rate': float(tau_rate),
            'attitude_safety_margin': float(attitude_safety_margin),
        }
        self._alphas = np.array(
            [alpha_fov, alpha_fov, alpha_attitude, alpha_attitude],
            dtype=np.float64,
        )
        self._scale = np.array([
            self._params['a_max'], self._params['a_max'],
            self._params['a_max'], self._params['yaw_rate_max'],
        ])
        self._k = len(_KEYS)
        self._m = 4
        self._ctx_dim = self._k * self._m + self._k  # 16 + 4 = 20

        # Expand observation space (base is Box(16,) in [-1, 1]; context is
        # unbounded Lie-derivative values, so use ±inf for the tail).
        base_space = env.observation_space
        assert isinstance(base_space, spaces.Box)
        low = np.concatenate([
            base_space.low.astype(np.float32),
            np.full(self._ctx_dim, -np.inf, dtype=np.float32),
        ])
        high = np.concatenate([
            base_space.high.astype(np.float32),
            np.full(self._ctx_dim, np.inf, dtype=np.float32),
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _compute_context(self) -> np.ndarray:
        """Compute the (A.flatten(), b) context vector from the env state."""
        state = state_from_env(self._base, self._params)
        lie = compute_lie_derivatives(state, self._params)
        A = np.vstack([lie[k]['Lgh'] for k in _KEYS]) * self._scale  # (4, 4)
        b = np.array(
            [-lie[k]['Lfh'] - self._alphas[i] * lie[k]['h']
             for i, k in enumerate(_KEYS)],
            dtype=np.float64,
        )
        return np.concatenate([A.flatten(), b]).astype(np.float32)

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        ctx = self._compute_context()
        return np.concatenate([np.asarray(obs, dtype=np.float32), ctx])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info
