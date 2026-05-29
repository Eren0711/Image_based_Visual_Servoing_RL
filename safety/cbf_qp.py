"""
Predictive CBF Safety Filter — Stage 4a Phase 4a.1
===================================================
A horizon-predictive safety filter that minimally scales back the agent's
action when it would cause a CBF violation within the prediction horizon.

This module was originally drafted as a proper CBF-QP (finite-difference
Jacobians + quadprog) following the construction in Ames et al. 2017.
That approach broke down on our system because:
  (a) The SO(3) attitude controller has an `omega_max` saturation clamp.
      Near commanded body accelerations of ~7.7 m/s² (= 79% of a_max),
      omega_d saturates, so any local finite-difference perturbation gives
      a near-zero Jacobian and the QP sees no gradient to follow.
  (b) The proper fix would be ECBF / HOCBF with explicit relative-degree
      handling and pole placement (Ames 2019 Thm. 8). That's a substantial
      rewrite involving analytical derivatives of the SO(3) PD law.

Instead, we use a 1D bisection filter:
  1. Predict h_i_min over a horizon by rolling out u for N steps.
  2. If h_i_min ≥ (1 − α_i) · h_i_now for all i: return u_rl (safe).
  3. Else: find the largest β ∈ [0, 1] such that β·u_rl is safe via
     bisection. Return β·u_rl.

This is *less* optimal than a full QP (we can only scale u_rl, not
re-direct it), but it has a hard safety guarantee against the horizon
predictor and is robust to all the saturation/nonlinearity issues that
trip up local linearization. Direction preservation is fine for our case
because the policy already produces sensible directions — when those
directions are too aggressive, throttling back is exactly what we want.

If Phase 4a.1 shows that direction-preservation is insufficient (e.g., the
CBF often kicks in for actions where a different *direction* would be safer
than just lower magnitude), Phase 4a.2 should upgrade to proper HOCBF.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from safety.cbf_constraints import evaluate_all


def _snapshot_interceptor(interceptor) -> dict:
    """Capture Multicopter6DOFLite state for snapshot/restore."""
    return {
        'position': interceptor.position.copy(),
        'velocity': interceptor.velocity.copy(),
        '_body_vel': interceptor._body_vel.copy(),
        '_yaw_rate': interceptor._yaw_rate,
        '_rotation': Rotation.from_matrix(interceptor._rotation.as_matrix()),
        'omega_body': interceptor.omega_body.copy(),
        'thrust': interceptor.thrust,
        '_body_accel': interceptor._body_accel.copy(),
        'pitch': interceptor.pitch,
        'roll': interceptor.roll,
        'yaw': interceptor.yaw,
    }


def _restore_interceptor(interceptor, snap: dict) -> None:
    """Restore Multicopter6DOFLite state from snapshot."""
    interceptor.position = snap['position'].copy()
    interceptor.velocity = snap['velocity'].copy()
    interceptor._body_vel = snap['_body_vel'].copy()
    interceptor._yaw_rate = snap['_yaw_rate']
    interceptor._rotation = Rotation.from_matrix(snap['_rotation'].as_matrix())
    interceptor.omega_body = snap['omega_body'].copy()
    interceptor.thrust = snap['thrust']
    interceptor._body_accel = snap['_body_accel'].copy()
    interceptor.pitch = snap['pitch']
    interceptor.roll = snap['roll']
    interceptor.yaw = snap['yaw']


def _predict_h_over_horizon(interceptor, target_pos_now, target_vel,
                            camera, scaled_action, snap: dict, params: dict,
                            horizons: np.ndarray, in_fov_only: bool
                            ) -> np.ndarray:
    """Apply scaled_action for max(horizons) steps starting from snap,
    return per-constraint minimum h, where constraint i uses only steps
    1..horizons[i].

    Per-constraint horizons because FOV (relative-degree-2 via position
    integration) and attitude (relative-degree-2 via attitude tracking lag)
    have different timescales. Short horizon (~3 steps) for FOV avoids the
    pessimism of "agent holds this action for 200ms" — the actual policy
    would change direction. Long horizon (~15 steps) for attitude is needed
    because attitude buildup happens over multiple tau_rate periods.

    Args:
        horizons: (4,) array of int horizons, one per constraint.

    Returns:
        h_min: (4,) array of per-constraint h-minimums over their horizons.
    """
    _restore_interceptor(interceptor, snap)
    max_h = int(np.max(horizons))
    h_min = np.full(4, np.inf, dtype=np.float64)
    for k in range(1, max_h + 1):
        interceptor.step(scaled_action)
        target_pos_k = target_pos_now + target_vel * (k * params['dt'])
        R_be = interceptor.get_rotation_matrix()
        p_r = interceptor.position - target_pos_k
        cam_result = camera.project(p_r, R_be)
        state_k = {
            'p_bar': cam_result['p_bar'],
            'in_fov': cam_result['in_fov'],
            'pitch': interceptor.pitch,
            'roll': interceptor.roll,
        }
        h_k = evaluate_all(state_k, params, in_fov_only)
        # Update min only for constraints whose horizon includes this step
        in_window = (k <= horizons)
        h_min = np.where(in_window, np.minimum(h_min, h_k), h_min)
    return h_min


class CBFQPSolver:
    """Predictive CBF safety filter via bisection on action magnitude.

    Despite the class name (kept for wrapper compatibility), this is no longer
    a QP solver — see module docstring.
    """

    def __init__(self, interceptor, target, camera, params: dict):
        self.interceptor = interceptor
        self.target = target
        self.camera = camera
        self.params = params
        self.a_max = float(params['a_max'])
        self.yaw_rate_max = float(params['yaw_rate_max'])
        self.dt = float(params['dt'])
        self.alphas = np.array([
            params['alpha_fov'],
            params['alpha_fov'],
            params['alpha_attitude'],
            params['alpha_attitude'],
        ], dtype=np.float64)
        self.in_fov_only = bool(params.get('in_fov_only', True))
        # Per-constraint horizons (in steps). FOV: short (~3 = 60ms),
        # attitude: long (~15 = 300ms, ~6 * tau_rate).
        self.horizons = np.array([
            int(params.get('horizon_fov', 3)),
            int(params.get('horizon_fov', 3)),
            int(params.get('horizon_attitude', 15)),
            int(params.get('horizon_attitude', 15)),
        ], dtype=np.int64)
        # Bisection tolerance / max iterations
        self._bisect_max_iter = int(params.get('bisect_max_iter', 12))
        self._bisect_tol = float(params.get('bisect_tol', 1e-3))

        # Stats
        self.n_solves = 0
        self.n_corrections = 0
        self.n_infeasible = 0
        self.correction_norm_sum = 0.0
        self.violations_per_constraint = np.zeros(4, dtype=np.int64)

    def reset_stats(self) -> None:
        self.n_solves = 0
        self.n_corrections = 0
        self.n_infeasible = 0
        self.correction_norm_sum = 0.0
        self.violations_per_constraint[:] = 0

    def _scale_action(self, u_norm: np.ndarray) -> np.ndarray:
        """Map normalized action [-1, 1]^4 → physical units."""
        return np.array([
            u_norm[0] * self.a_max,
            u_norm[1] * self.a_max,
            u_norm[2] * self.a_max,
            u_norm[3] * self.yaw_rate_max,
        ])

    def _is_safe(self, h_min: np.ndarray, threshold: np.ndarray) -> bool:
        """Check if predicted h_min satisfies CBF condition."""
        # Use a small tolerance to avoid spurious "unsafe" due to numerics
        return bool(np.all(h_min >= threshold - 1e-6))

    def solve(self, u_rl: np.ndarray) -> tuple:
        """Filter the RL action.

        Args:
            u_rl: (4,) normalized action ∈ [-1, 1]^4.

        Returns:
            (u_safe, info): u_safe is the safe action, info has diagnostics.
        """
        u_rl = np.asarray(u_rl, dtype=np.float64).copy()
        u_rl = np.clip(u_rl, -1.0, 1.0)

        # --- Capture current target & interceptor state ---
        target_pos_now = self.target.position.copy()
        target_vel = self.target.velocity.copy()
        snap = _snapshot_interceptor(self.interceptor)

        # --- h at current state (pre-step) ---
        R_be_now = self.interceptor.get_rotation_matrix()
        p_r_now = self.interceptor.position - target_pos_now
        cam_now = self.camera.project(p_r_now, R_be_now)
        state_now = {
            'p_bar': cam_now['p_bar'],
            'in_fov': cam_now['in_fov'],
            'pitch': self.interceptor.pitch,
            'roll': self.interceptor.roll,
        }
        h_now = evaluate_all(state_now, self.params, self.in_fov_only)

        # CBF threshold (per constraint)
        threshold = (1.0 - self.alphas) * h_now

        # Track violations of h_now (for diagnostics, not a stop condition)
        for i in range(4):
            if h_now[i] < 0:
                self.violations_per_constraint[i] += 1

        info = {
            'h_now': h_now.copy(),
            'corrected': False,
            'correction_norm': 0.0,
            'feasible': True,
        }

        # --- Test u_rl directly ---
        h_min_rl = _predict_h_over_horizon(
            self.interceptor, target_pos_now, target_vel, self.camera,
            self._scale_action(u_rl), snap, self.params,
            self.horizons, self.in_fov_only,
        )
        info['h_baseline'] = h_min_rl.copy()

        if self._is_safe(h_min_rl, threshold):
            _restore_interceptor(self.interceptor, snap)
            self.n_solves += 1
            return u_rl, info

        # --- u_rl unsafe: bisection on β ∈ [0, 1] for β·u_rl ---
        # Check β=0 (zero action). If even that's unsafe, the current state
        # is already past the point of return — mark infeasible and return 0.
        h_min_zero = _predict_h_over_horizon(
            self.interceptor, target_pos_now, target_vel, self.camera,
            self._scale_action(np.zeros(4)), snap, self.params,
            self.horizons, self.in_fov_only,
        )
        if not self._is_safe(h_min_zero, threshold):
            # Even doing nothing violates — return zero action and mark
            # infeasible. The wrapper will log this.
            _restore_interceptor(self.interceptor, snap)
            self.n_solves += 1
            self.n_infeasible += 1
            self.n_corrections += 1
            info['feasible'] = False
            info['corrected'] = True
            info['correction_norm'] = float(np.linalg.norm(u_rl))
            return np.zeros(4), info

        # Bisect: find largest β ∈ [0, 1] with β·u_rl safe.
        # Invariants: lo is always safe (start with lo=0), hi is always unsafe.
        lo, hi = 0.0, 1.0
        for _ in range(self._bisect_max_iter):
            mid = 0.5 * (lo + hi)
            h_min_mid = _predict_h_over_horizon(
                self.interceptor, target_pos_now, target_vel, self.camera,
                self._scale_action(mid * u_rl), snap, self.params,
                self.horizons, self.in_fov_only,
            )
            if self._is_safe(h_min_mid, threshold):
                lo = mid
            else:
                hi = mid
            if hi - lo < self._bisect_tol:
                break

        u_safe = lo * u_rl
        _restore_interceptor(self.interceptor, snap)

        self.n_solves += 1
        self.n_corrections += 1
        correction = float(np.linalg.norm(u_safe - u_rl))
        self.correction_norm_sum += correction
        info['corrected'] = True
        info['correction_norm'] = correction
        info['beta'] = lo  # the scaling factor chosen
        return u_safe, info

    def get_stats(self) -> dict:
        """Return solver statistics since last reset."""
        avg_correction = (
            self.correction_norm_sum / self.n_corrections
            if self.n_corrections > 0 else 0.0
        )
        return {
            'n_solves': self.n_solves,
            'n_corrections': self.n_corrections,
            'n_infeasible': self.n_infeasible,
            'correction_rate': (
                self.n_corrections / self.n_solves
                if self.n_solves > 0 else 0.0
            ),
            'avg_correction_norm': avg_correction,
            'violations_per_constraint': self.violations_per_constraint.copy(),
        }
