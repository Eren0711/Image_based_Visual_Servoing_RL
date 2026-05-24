"""
Recursive Depth Estimator — Jacobian-Based
============================================
Provides a filtered, recursive estimate of target depth (z_c) using
the IBVS interaction matrix and known ego-motion.

The estimator tracks the inverse depth ρ = 1/z_c because:
    1. The image Jacobian equation is LINEAR in ρ (not in z_c)
    2. ρ has a bounded, well-behaved range for targets in front of the camera
    3. The Kalman filter update is simpler in the inverse-depth space

Architecture:
    At each timestep:
        1. Compute camera velocity in camera frame (from IMU / dynamics)
        2. Compute image feature velocity (finite differences or tracking)
        3. Use the Jacobian to get an instantaneous ρ estimate
        4. Fuse with the recursive filter (simple Kalman or EMA)

The filter handles:
    - Noisy instantaneous estimates (averaged over time)
    - Low-confidence measurements (weighted by observability metric)
    - Target behind camera or depth singularities (clamped)
    - Initial convergence from a prior guess

Reference: Section II-A.4 of arXiv:2404.08296
           DKF design in Section III of the same paper
"""

import numpy as np
from observers.interaction_matrix import InteractionMatrix


class DepthEstimator:
    """Recursive inverse-depth estimator using the image Jacobian.

    Maintains a filtered estimate of ρ = 1/z_c (inverse depth) by
    fusing instantaneous Jacobian-based measurements with a simple
    scalar Kalman filter.

    State model:
        ρ_{k+1} = ρ_k + w_k          (random walk — depth changes slowly)
        y_k     = ρ_k + n_k          (Jacobian-based measurement)

    The measurement noise variance is adapted based on the confidence
    (observability) of each Jacobian estimate.

    Attributes:
        rho_hat:     Current filtered inverse depth estimate.
        P:           Current estimation uncertainty (variance).
        z_hat:       Current depth estimate (1/rho_hat).
        confidence:  Latest measurement confidence.
    """

    def __init__(
        self,
        rho_init: float = 0.05,
        P_init: float = 1.0,
        Q: float = 0.001,
        R_base: float = 0.1,
        rho_min: float = 0.005,
        rho_max: float = 2.0,
        confidence_threshold: float = 0.01,
    ):
        """Initialize the depth estimator.

        Args:
            rho_init:   Initial inverse depth estimate (default: 1/20 = 20m).
            P_init:     Initial estimation variance (large = uncertain).
            Q:          Process noise variance (how fast ρ changes per step).
                        Larger Q → filter trusts measurements more.
            R_base:     Base measurement noise variance.
                        Actual R is scaled by 1/confidence.
            rho_min:    Minimum allowed inverse depth (= max distance).
                        Default 0.005 → 200m max distance.
            rho_max:    Maximum allowed inverse depth (= min distance).
                        Default 2.0 → 0.5m min distance.
            confidence_threshold: Minimum ||b||² to accept a measurement.
                        Below this, the measurement is rejected (no update).
        """
        self.Q = Q
        self.R_base = R_base
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.confidence_threshold = confidence_threshold

        # Filter state
        self.rho_hat = rho_init
        self.P = P_init
        self.confidence = 0.0

        # History (for debugging / visualization)
        self._history = {
            'rho_hat': [rho_init],
            'rho_meas': [],
            'z_hat': [1.0 / max(rho_init, 1e-9)],
            'confidence': [],
            'P': [P_init],
        }

    def reset(self, rho_init: float = 0.05, P_init: float = 1.0) -> None:
        """Reset the estimator to a new initial state.

        Args:
            rho_init: Initial inverse depth guess.
            P_init:   Initial uncertainty.
        """
        self.rho_hat = rho_init
        self.P = P_init
        self.confidence = 0.0
        self._history = {
            'rho_hat': [rho_init],
            'rho_meas': [],
            'z_hat': [1.0 / max(rho_init, 1e-9)],
            'confidence': [],
            'P': [P_init],
        }

    def update(
        self,
        p_bar: np.ndarray,
        p_bar_dot: np.ndarray,
        v_cam: np.ndarray,
        omega_cam: np.ndarray,
        v_target_cam: np.ndarray = None,
    ) -> dict:
        """Perform one estimation step: predict + Jacobian measurement update.

        This is the main method called at every simulation timestep.

        Args:
            p_bar:         Normalized image coords [p̄_x, p̄_y]. Shape (2,).
            p_bar_dot:     Image velocity [dp̄_x/dt, dp̄_y/dt]. Shape (2,).
            v_cam:         Camera translational velocity in camera frame. Shape (3,).
            omega_cam:     Camera angular velocity in camera frame. Shape (3,).
            v_target_cam:  Target velocity in camera frame (if known). Shape (3,)
                           or None (assumes static target).

        Returns:
            dict with keys:
                'rho_hat'    : float — filtered inverse depth.
                'z_hat'      : float — filtered depth = 1/ρ̂.
                'rho_meas'   : float — instantaneous Jacobian-based ρ.
                'confidence' : float — observability metric.
                'P'          : float — estimation variance.
                'updated'    : bool  — whether the measurement was used.
        """
        # === Prediction step (random walk model) ===
        # ρ_{k|k-1} = ρ_{k-1|k-1}     (no dynamics model for depth)
        # P_{k|k-1}  = P_{k-1|k-1} + Q
        rho_pred = self.rho_hat
        P_pred = self.P + self.Q

        # === Measurement: Jacobian-based inverse depth ===
        jac_result = InteractionMatrix.estimate_inverse_depth(
            p_bar=p_bar,
            p_bar_dot=p_bar_dot,
            v_cam=v_cam,
            omega_cam=omega_cam,
            v_target_cam=v_target_cam,
        )

        rho_meas = jac_result['rho']
        self.confidence = jac_result['confidence']

        # === Decide whether to update ===
        updated = False

        if self.confidence >= self.confidence_threshold and rho_meas > 0:
            # Measurement noise: inversely proportional to confidence.
            # High confidence (large ||b||²) → low noise → trust measurement.
            # Low confidence → high noise → mostly ignore.
            R = self.R_base / max(self.confidence, 1e-6)

            # === Kalman update ===
            # Innovation: y - H·x_pred (H = 1 for scalar state)
            innovation = rho_meas - rho_pred

            # Innovation variance: S = P_pred + R
            S = P_pred + R

            # Kalman gain: K = P_pred / S
            K = P_pred / S

            # State update: ρ̂ = ρ_pred + K · innovation
            self.rho_hat = rho_pred + K * innovation

            # Covariance update: P = (1 - K) · P_pred
            self.P = (1.0 - K) * P_pred

            updated = True
        else:
            # No update — use prediction only
            self.rho_hat = rho_pred
            self.P = P_pred

        # === Clamp to valid range ===
        self.rho_hat = np.clip(self.rho_hat, self.rho_min, self.rho_max)

        # === Compute depth ===
        z_hat = 1.0 / self.rho_hat

        # === Record history ===
        self._history['rho_hat'].append(self.rho_hat)
        self._history['rho_meas'].append(rho_meas)
        self._history['z_hat'].append(z_hat)
        self._history['confidence'].append(self.confidence)
        self._history['P'].append(self.P)

        return {
            'rho_hat': float(self.rho_hat),
            'z_hat': float(z_hat),
            'rho_meas': float(rho_meas),
            'confidence': float(self.confidence),
            'P': float(self.P),
            'updated': updated,
        }

    @property
    def z_hat(self) -> float:
        """Current depth estimate (convenience property)."""
        return 1.0 / max(self.rho_hat, 1e-9)

    def get_history(self) -> dict:
        """Return recorded estimation history for plotting.

        Returns:
            dict with numpy arrays:
                'rho_hat', 'rho_meas', 'z_hat', 'confidence', 'P'
        """
        return {
            key: np.array(val) for key, val in self._history.items()
        }
