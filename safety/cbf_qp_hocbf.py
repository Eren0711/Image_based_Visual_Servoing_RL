"""
HOCBF Quadratic Program — Stage 4a Phase 3
============================================
A proper CBF-QP using analytical Lie derivatives (cbf_lie.py) from a
control-affine proxy linearization of the multicopter dynamics. Replaces
the bisection-based filter (cbf_qp.py) so the QP can find ANY safe action
near u_RL, not just a magnitude-scaled version.

QP formulation:

    minimize_u   ½ ‖u − u_RL‖²
    subject to   L_g h_i(x) · u  ≥  −L_f h_i(x) − α_i · h_i(x)    ∀ i
                 −1  ≤  u_j  ≤  1                                  ∀ j

Linear inequalities (4 CBF + 8 box = 12 total), quadratic objective, 4
variables. Solved with quadprog (active-set). Sub-millisecond per call.

----------------------------------------------------------------------------
Why this is the right shape (and the bisection wasn't):

The bisection-only filter could only *scale* u_RL toward zero — when the
proxy said u_RL was unsafe, the filter would dial back the action's
magnitude on all 4 axes uniformly. This loses information: maybe only one
axis was causing the violation; maybe a different *direction* (not just
smaller magnitude) would satisfy the barriers.

This QP can pick any u in [-1, 1]^4. It only modifies the dimensions that
matter for the active constraint, leaving the rest of u_RL untouched. As a
result, the correction norm should be much smaller than bisection's, and
the policy is left freer to track the target.

----------------------------------------------------------------------------
Infeasibility:

A QP infeasibility under the *current* state means there is no u in the
action box that keeps h_{k+1} ≥ (1-α) h_k for ALL constraints
simultaneously. This happens when the policy has already navigated into a
corner — typically deep attitude excursion or near-FOV-edge with target
moving sideways. In those cases we fall back to the bisection filter from
cbf_qp.py (which always returns *some* u — possibly zero — that the
predictor agrees is safe).

This dual-strategy is also what Ames et al. recommend for actuator-limited
systems (2019 §III): "if the constraints are not co-satisfiable, relax
toward a single safety-preserving fallback."
"""

import numpy as np
import quadprog

from safety.cbf_lie import compute_lie_derivatives, state_from_env
from safety.cbf_qp import CBFQPSolver as BisectionSolver


# Order of constraints in arrays
_CONSTRAINT_KEYS = ['hfov', 'vfov', 'pitch', 'roll']


class HOCBFQPSolver:
    """Proper CBF-QP solver using analytical Lie derivatives.

    Constructor signature matches CBFQPSolver (cbf_qp.py) so it can be
    swapped in by the wrapper without other changes.

    Args:
        interceptor, target, camera: env objects (snapshotted; only used by
            the bisection fallback).
        params: dict with keys:
            'a_max', 'yaw_rate_max', 'dt'    — action scaling.
            'tau_rate'                       — proxy attitude lag.
            'tan_half_hfov', 'tan_half_vfov' — FOV limits.
            'max_pitch', 'max_roll'          — attitude limits.
            'alpha_fov', 'alpha_attitude'    — CBF margins (1/s) for the
                 continuous-time formulation. (NOT the same as the
                 bisection α — those were per-step decay; these are
                 class-K extended rates.)
            'in_fov_only' (bool)             — skip FOV barriers when target
                                                out of FOV (default True).
            'fallback' (str)                 — 'bisection' | 'zero' | 'rl'.
                 Behavior when QP is infeasible. Default 'bisection'.
            'env_unwrapped' (object)         — the unwrapped env, needed by
                                                the state extractor.
    """

    def __init__(self, interceptor, target, camera, params: dict):
        self.interceptor = interceptor
        self.target = target
        self.camera = camera
        self.params = params
        self.a_max = float(params['a_max'])
        self.yaw_rate_max = float(params['yaw_rate_max'])
        self.dt = float(params['dt'])
        # Per-constraint α (s^-1). Higher = stronger pushback.
        self.alphas = np.array([
            params['alpha_fov'],       # hfov
            params['alpha_fov'],       # vfov
            params['alpha_attitude'],  # pitch
            params['alpha_attitude'],  # roll
        ], dtype=np.float64)
        self.in_fov_only = bool(params.get('in_fov_only', True))
        self.fallback_kind = str(params.get('fallback', 'bisection'))
        # env_unwrapped is needed to build the state dict for the Lie deriv
        # computation. The wrapper sets this in build_solver().
        self.env_unwrapped = params.get('env_unwrapped', None)

        # Bisection fallback (re-uses Phase 4a.1 code path)
        self._bisection = BisectionSolver(interceptor, target, camera, {
            'a_max': self.a_max,
            'yaw_rate_max': self.yaw_rate_max,
            'dt': self.dt,
            'tan_half_hfov': params['tan_half_hfov'],
            'tan_half_vfov': params['tan_half_vfov'],
            'max_pitch': params['max_pitch'],
            'max_roll': params['max_roll'],
            # Convert continuous-time α (1/s) → per-step decay (1 - α·dt)
            'alpha_fov': min(1.0, params['alpha_fov'] * self.dt),
            'alpha_attitude': min(1.0, params['alpha_attitude'] * self.dt),
            'horizon_fov': int(params.get('horizon_fov', 3)),
            'horizon_attitude': int(params.get('horizon_attitude', 15)),
            'in_fov_only': self.in_fov_only,
        })

        # Stats
        self.n_solves = 0
        self.n_corrections = 0
        self.n_infeasible = 0
        self.n_fallback = 0
        self.correction_norm_sum = 0.0
        self.violations_per_constraint = np.zeros(4, dtype=np.int64)

    def reset_stats(self) -> None:
        self.n_solves = 0
        self.n_corrections = 0
        self.n_infeasible = 0
        self.n_fallback = 0
        self.correction_norm_sum = 0.0
        self.violations_per_constraint[:] = 0
        self._bisection.reset_stats()

    def _scale_action(self, u_norm: np.ndarray) -> np.ndarray:
        """Map normalized [-1, 1]^4 → physical units (for fallback)."""
        return np.array([
            u_norm[0] * self.a_max,
            u_norm[1] * self.a_max,
            u_norm[2] * self.a_max,
            u_norm[3] * self.yaw_rate_max,
        ])

    def solve(self, u_rl: np.ndarray) -> tuple:
        """Filter the RL action through the HOCBF QP.

        Args:
            u_rl: (4,) normalized action ∈ [-1, 1]^4.

        Returns:
            (u_safe, info): u_safe ∈ [-1, 1]^4, info has diagnostics.
        """
        u_rl = np.asarray(u_rl, dtype=np.float64).copy()
        u_rl = np.clip(u_rl, -1.0, 1.0)

        # Build state dict and compute Lie derivatives
        state = state_from_env(self.env_unwrapped, self.params)
        lie = compute_lie_derivatives(state, self.params)

        # Pack into arrays in canonical order [hfov, vfov, pitch, roll]
        h_now = np.array([lie[k]['h'] for k in _CONSTRAINT_KEYS])
        Lf = np.array([lie[k]['Lfh'] for k in _CONSTRAINT_KEYS])
        # Lg is a (4, 4) matrix: rows = constraints, cols = action dims.
        # However, Lg is computed in *physical* action units. The CBF
        # constraint is L_f h + L_g h · u_physical + α h ≥ 0. We work in
        # normalized u (∈ [-1,1]^4), so rescale: u_physical_j = u_norm_j ·
        # scale_j, where scale = [a_max, a_max, a_max, yaw_rate_max]. So
        # L_g h · u_physical = (L_g h · diag(scale)) · u_norm = Lg_scaled · u_norm
        scale = np.array([self.a_max, self.a_max, self.a_max, self.yaw_rate_max])
        Lg = np.vstack([lie[k]['Lgh'] for k in _CONSTRAINT_KEYS]) * scale

        # Track violations
        for i in range(4):
            if h_now[i] < 0:
                self.violations_per_constraint[i] += 1

        info = {
            'h_now': h_now.copy(),
            'Lf': Lf.copy(),
            'Lg': Lg.copy(),
            'corrected': False,
            'correction_norm': 0.0,
            'feasible': True,
            'fallback_used': False,
        }

        # Active constraint mask:
        # - Always include pitch/roll
        # - Include FOV only if target is in FOV (else skip — see in_fov_only)
        active_mask = np.ones(4, dtype=bool)
        if self.in_fov_only and not state.get('in_fov', True):
            active_mask[0] = False  # hfov
            active_mask[1] = False  # vfov

        # Check whether u_rl already satisfies all active constraints
        lhs_rl = Lf + Lg @ u_rl + self.alphas * h_now
        violated = active_mask & (lhs_rl < -1e-6)
        if not np.any(violated):
            self.n_solves += 1
            return u_rl, info

        # Build the QP:
        #   min ½ ||u - u_rl||²  =  ½ u^T I u - u_rl^T u + const
        # quadprog: min ½ u^T G u - a^T u s.t. C^T u ≥ b
        G = np.eye(4)
        a_qp = u_rl.copy()

        # Active CBF constraints (rows where active_mask is True)
        Lg_active = Lg[active_mask]
        rhs_cbf = (-Lf - self.alphas * h_now)[active_mask]

        # Box constraints (always)
        C_box = np.vstack([np.eye(4), -np.eye(4)])  # 8×4
        b_box = np.concatenate([-np.ones(4), -np.ones(4)])  # 8

        C = np.vstack([Lg_active, C_box])
        b = np.concatenate([rhs_cbf, b_box])

        try:
            res = quadprog.solve_qp(G, a_qp, C.T, b, 0)
            u_safe = np.asarray(res[0], dtype=np.float64)
            u_safe = np.clip(u_safe, -1.0, 1.0)
            self.n_solves += 1
            correction = float(np.linalg.norm(u_safe - u_rl))
            if correction > 1e-6:
                self.n_corrections += 1
                self.correction_norm_sum += correction
                info['corrected'] = True
            info['correction_norm'] = correction
            return u_safe, info
        except ValueError:
            # Infeasible: fall back to bisection (which is guaranteed to
            # produce *some* safe action against its own horizon predictor).
            self.n_solves += 1
            self.n_infeasible += 1
            info['feasible'] = False
            info['fallback_used'] = True
            self.n_fallback += 1
            if self.fallback_kind == 'zero':
                u_safe = np.zeros(4)
            elif self.fallback_kind == 'rl':
                u_safe = u_rl
            else:
                # bisection (default)
                u_safe, _ = self._bisection.solve(u_rl)
            correction = float(np.linalg.norm(u_safe - u_rl))
            if correction > 1e-6:
                self.n_corrections += 1
                self.correction_norm_sum += correction
                info['corrected'] = True
            info['correction_norm'] = correction
            return u_safe, info

    def get_stats(self) -> dict:
        avg = (self.correction_norm_sum / self.n_corrections
               if self.n_corrections > 0 else 0.0)
        return {
            'n_solves': self.n_solves,
            'n_corrections': self.n_corrections,
            'n_infeasible': self.n_infeasible,
            'n_fallback': self.n_fallback,
            'correction_rate': (self.n_corrections / self.n_solves
                                if self.n_solves > 0 else 0.0),
            'avg_correction_norm': avg,
            'violations_per_constraint': self.violations_per_constraint.copy(),
        }
