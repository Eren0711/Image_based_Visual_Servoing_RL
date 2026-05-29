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


class CBFWrapper(gym.Wrapper):
    """CBF-QP safety filter wrapper.

    Args:
        env: a Gymnasium env (typically the full DKF-wrapped stack).
        alpha_fov: CBF margin for FOV constraints (per step). Higher =
            QP intervenes more aggressively. Default 1.0 = no margin
            (just keep h ≥ 0 at next step).
        alpha_attitude: CBF margin for pitch/roll constraints.
        in_fov_only: if True, the FOV constraints are skipped when the
            target is already out of FOV — the CBF doesn't try to recover
            (the DKF wrapper handles re-acquisition). Default True.

    Reads CBF parameters from the unwrapped env's camera and interceptor
    config (alpha_hfov, alpha_vfov, max_pitch, max_roll, a_max, etc.).
    """

    def __init__(
        self,
        env: gym.Env,
        alpha_fov: float = 0.8,
        alpha_attitude: float = 0.3,
        in_fov_only: bool = True,
        horizon_fov: int = 3,
        horizon_attitude: int = 15,
    ):
        super().__init__(env)
        self.alpha_fov = float(alpha_fov)
        self.alpha_attitude = float(alpha_attitude)
        self.in_fov_only = bool(in_fov_only)
        self.horizon_fov = int(horizon_fov)
        self.horizon_attitude = int(horizon_attitude)

        # Find the base env (deepest unwrapped) — it holds interceptor, target,
        # camera. We bind the solver lazily on first reset so any sub-wrappers
        # that hold their own state are already initialized.
        self._solver = None

        # Episode-aggregated stats
        self._ep_corrections = 0
        self._ep_steps = 0
        self._ep_violations = np.zeros(4, dtype=np.int64)

    def _build_solver(self) -> None:
        """Build the CBFQPSolver from the unwrapped env's state."""
        base = self.env.unwrapped
        cam_fov = base.camera.get_fov_params()
        params = {
            'a_max': base.a_max,
            'yaw_rate_max': base.yaw_rate_max,
            'dt': base.dt,
            'tan_half_hfov': cam_fov['tan_half_hfov'],
            'tan_half_vfov': cam_fov['tan_half_vfov'],
            'max_pitch': base.max_pitch,
            'max_roll': base.max_roll,
            'alpha_fov': self.alpha_fov,
            'alpha_attitude': self.alpha_attitude,
            'in_fov_only': self.in_fov_only,
            'horizon_fov': self.horizon_fov,
            'horizon_attitude': self.horizon_attitude,
        }
        self._solver = CBFQPSolver(
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
