"""
CBF Action-Filter Wrapper — Stage 4a Phase 4a.1
================================================
Gymnasium ActionWrapper that filters the policy action through a CBF-QP
solver before passing it to the wrapped env's step. This is the "safety
filter" / "Active Set Invariance Filter" pattern from Ames et al. 2019 §V-C.

Stack order (outside-in):
  agent → CBFWrapper.step(action) → DKFWrapper → NoiseDelayWrapper → env

The CBF wrapper sits outside the DKF/Noise wrappers because:
  - The agent's observation is the DKF-filtered estimate (with noise/delay).
  - The CBF-QP needs the *true* current state to predict h_{k+1}.
    Phase 4a.1 uses ground-truth state for the QP — this isolates the
    "does the CBF concept work" question from the "does the DKF estimate
    feed the CBF well enough" question. Phase 4a.2 would re-test with the
    DKF estimate going into the QP instead.

Info forwarding: the wrapper adds CBF diagnostic fields to the step info
dict (correction norm, active constraints, per-episode stats). Eval scripts
can collect these from the env step return value.
"""

import numpy as np
import gymnasium as gym

from safety.cbf_qp import CBFQPSolver
from safety.cbf_qp_hocbf import HOCBFQPSolver


class CBFWrapper(gym.Wrapper):
    """CBF-QP safety filter wrapper.

    Two solver methods available via the `method` arg:
      'bisection' (Phase 4a.1): 1D line-search on |u_RL|, horizon-predictor
          via Multicopter6DOFLite.step. Robust to nonlinearity but can only
          throttle, never redirect.
      'hocbf' (Phase 4a.3, default): proper CBF-QP with analytical Lie
          derivatives from a proxy control-affine linearization. Can pick
          any safe direction near u_RL, not just throttled. Bisection
          serves as fallback if the QP is infeasible.

    Args:
        env: a Gymnasium env (typically the full DKF-wrapped stack).
        method: 'hocbf' (default) or 'bisection'.
        alpha_fov, alpha_attitude: CBF margins. For 'bisection' these are
            per-step decay rates (1.0 = no margin). For 'hocbf' these are
            class-K extended rates (1/s) — higher = stronger pushback.
        in_fov_only: skip FOV constraints when target is out of FOV.
        horizon_fov, horizon_attitude: only used by 'bisection' (and by
            'hocbf' as the bisection-fallback prediction horizons).
    """

    def __init__(
        self,
        env: gym.Env,
        method: str = 'hocbf',
        alpha_fov: float = 5.0,
        alpha_attitude: float = 5.0,
        in_fov_only: bool = True,
        horizon_fov: int = 3,
        horizon_attitude: int = 15,
        tau_rate: float = 0.05,
        attitude_safety_margin: float = 0.15,
    ):
        super().__init__(env)
        self.method = str(method)
        if self.method not in ('hocbf', 'bisection'):
            raise ValueError(f"method must be 'hocbf' or 'bisection', got {method!r}")
        self.alpha_fov = float(alpha_fov)
        self.alpha_attitude = float(alpha_attitude)
        self.in_fov_only = bool(in_fov_only)
        self.horizon_fov = int(horizon_fov)
        self.horizon_attitude = int(horizon_attitude)
        self.tau_rate = float(tau_rate)
        self.attitude_safety_margin = float(attitude_safety_margin)

        self._solver = None

        self._ep_corrections = 0
        self._ep_steps = 0
        self._ep_violations = np.zeros(4, dtype=np.int64)

    def _build_solver(self) -> None:
        """Build the chosen solver from the unwrapped env's state."""
        base = self.env.unwrapped
        cam_fov = base.camera.get_fov_params()
        common = {
            'a_max': base.a_max,
            'yaw_rate_max': base.yaw_rate_max,
            'dt': base.dt,
            'tan_half_hfov': cam_fov['tan_half_hfov'],
            'tan_half_vfov': cam_fov['tan_half_vfov'],
            'max_pitch': base.max_pitch,
            'max_roll': base.max_roll,
            'in_fov_only': self.in_fov_only,
            'horizon_fov': self.horizon_fov,
            'horizon_attitude': self.horizon_attitude,
        }
        if self.method == 'bisection':
            params = {
                **common,
                'alpha_fov': self.alpha_fov,
                'alpha_attitude': self.alpha_attitude,
            }
            self._solver = CBFQPSolver(
                interceptor=base.interceptor,
                target=base.target,
                camera=base.camera,
                params=params,
            )
        else:  # 'hocbf'
            params = {
                **common,
                'alpha_fov': self.alpha_fov,
                'alpha_attitude': self.alpha_attitude,
                'tau_rate': self.tau_rate,
                'attitude_safety_margin': self.attitude_safety_margin,
                'env_unwrapped': base,
                'fallback': 'bisection',
            }
            self._solver = HOCBFQPSolver(
                interceptor=base.interceptor,
                target=base.target,
                camera=base.camera,
                params=params,
            )

    def reset(self, **kwargs):
        """Reset the inner env and rebuild the solver."""
        obs, info = self.env.reset(**kwargs)
        if self._solver is None:
            self._build_solver()
        self._solver.reset_stats()
        self._ep_corrections = 0
        self._ep_steps = 0
        self._ep_violations[:] = 0
        return obs, info

    def step(self, action):
        """Filter action through the CBF-QP, then step the inner env.

        Args:
            action: np.ndarray (4,), agent's proposed action in [-1, 1]^4.

        Returns:
            (obs, reward, terminated, truncated, info) with CBF diagnostics
            attached to info under the 'cbf' key.
        """
        if self._solver is None:
            self._build_solver()

        u_safe, cbf_info = self._solver.solve(action)
        obs, reward, terminated, truncated, info = self.env.step(u_safe)

        # Bookkeeping
        self._ep_steps += 1
        if cbf_info['corrected']:
            self._ep_corrections += 1
        for i in range(4):
            if cbf_info['h_now'][i] < 0:
                self._ep_violations[i] += 1

        # Attach diagnostics
        info = dict(info) if info is not None else {}
        info['cbf'] = {
            'corrected': cbf_info['corrected'],
            'correction_norm': cbf_info['correction_norm'],
            'feasible': cbf_info['feasible'],
            'h_now': cbf_info['h_now'].tolist(),
        }
        if terminated or truncated:
            stats = self._solver.get_stats()
            info['cbf_episode'] = {
                'n_steps': self._ep_steps,
                'n_corrections': self._ep_corrections,
                'correction_rate': (
                    self._ep_corrections / self._ep_steps
                    if self._ep_steps > 0 else 0.0
                ),
                'n_infeasible': stats['n_infeasible'],
                'avg_correction_norm': stats['avg_correction_norm'],
                'violations_per_constraint': self._ep_violations.tolist(),
            }
        return obs, reward, terminated, truncated, info
