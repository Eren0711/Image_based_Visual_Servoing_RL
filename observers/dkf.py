"""
Disturbance Kalman Filter (DKF) for Image-Plane State Estimation
=================================================================
Implements a Kalman Filter with delay-compensated measurements for
estimating the current image-plane state of the target from noisy,
delayed camera measurements.

Key features:
  1. State augmentation: tracks both position and velocity on the image plane
  2. Delay compensation: handles D-step measurement delay by maintaining
     a buffer of past predictions and performing retroactive updates
  3. Disturbance robustness: process noise models unpredictable target motion

State vector (4-dim):
    x = [p̄_x, p̄_y, dp̄_x/dt, dp̄_y/dt]ᵀ

Prediction model (constant-velocity):
    x_{k+1} = F · x_k + w_k
    F = [[1, 0, Δt, 0],
         [0, 1, 0,  Δt],
         [0, 0, 1,  0 ],
         [0, 0, 0,  1 ]]
    w_k ~ N(0, Q)

Measurement model (delayed, noisy):
    z_k = H · x_{k-D} + n_k
    H = [[1, 0, 0, 0],
         [0, 1, 0, 0]]
    n_k ~ N(0, R)

Delay compensation uses the "state augmentation" approach:
  - Maintain a circular buffer of past state predictions
  - When a delayed measurement z_k arrives (from time k-D):
    1. Retrieve the predicted state at time k-D
    2. Compute the innovation against that prediction
    3. Propagate the correction forward through the D intermediate steps

Reference: Section III of arXiv:2404.08296
"""

import numpy as np
from collections import deque


class DisturbanceKalmanFilter:
    """Delay-compensated Kalman filter for image-plane state estimation.

    Tracks the target's 2D image-plane position and velocity, filtering
    noisy and delayed measurements to produce a current-time state estimate.

    The filter is "disturbance-aware" through generous process noise on
    the velocity states, allowing it to track maneuvering targets without
    an explicit disturbance model.

    Attributes:
        x_hat:     Current state estimate [p̄_x, p̄_y, dp̄_x/dt, dp̄_y/dt].
        P:         Current 4×4 error covariance matrix.
        delay:     Measurement delay in timesteps.
        dt:        Timestep duration.
    """

    def __init__(
        self,
        dt: float = 0.02,
        delay: int = 3,
        sigma_pos_process: float = 0.01,
        sigma_vel_process: float = 0.5,
        sigma_measurement: float = 0.03,
        P_init_pos: float = 0.1,
        P_init_vel: float = 1.0,
    ):
        """Initialize the DKF.

        Args:
            dt:                  Timestep duration (s).
            delay:               Measurement delay D (in timesteps).
                                 D=3 at 50Hz = 60ms delay.
            sigma_pos_process:   Process noise std on position states.
                                 Small: position changes are driven by velocity.
            sigma_vel_process:   Process noise std on velocity states.
                                 Large: target velocity is unpredictable.
            sigma_measurement:   Measurement noise std on image coordinates.
                                 0.01-0.05 in normalized coords.
            P_init_pos:          Initial position uncertainty.
            P_init_vel:          Initial velocity uncertainty.
        """
        self.dt = dt
        self.delay = delay

        # --- State transition matrix F ---
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # --- Measurement matrix H ---
        # We only measure position, not velocity
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        # --- Process noise covariance Q ---
        # Position noise is small (driven by velocity model)
        # Velocity noise is large (target motion is unpredictable)
        self.Q = np.diag([
            sigma_pos_process ** 2,
            sigma_pos_process ** 2,
            sigma_vel_process ** 2,
            sigma_vel_process ** 2,
        ])

        # --- Measurement noise covariance R ---
        self.R = np.diag([
            sigma_measurement ** 2,
            sigma_measurement ** 2,
        ])

        # --- Initial state and covariance ---
        self.x_hat = np.zeros(4)
        self.P = np.diag([P_init_pos, P_init_pos, P_init_vel, P_init_vel])
        self._P_init = self.P.copy()

        # --- Delay compensation buffer ---
        # Stores (x_hat, P) at each past timestep for retroactive correction
        # Buffer length = delay + 1 (need current + D past states)
        max_buffer = max(delay + 2, 10)
        self._state_buffer = deque(maxlen=max_buffer)
        self._cov_buffer = deque(maxlen=max_buffer)

        # --- Diagnostics ---
        self._step_count = 0

        # --- IMU-aware prediction state (B-minimal upgrade) ---
        # Stores the previous step's camera-induced image velocity so we can
        # propagate only its change in the prediction step. Zero on init.
        self._prev_pbar_dot_cam = np.zeros(2)

    def reset(self, p_bar_init: np.ndarray = None) -> None:
        """Reset the filter to initial conditions.

        Args:
            p_bar_init: Initial image-plane position [p̄_x, p̄_y].
                        If provided, initializes position states.
                        Velocity is initialized to zero.
        """
        if p_bar_init is not None:
            self.x_hat = np.array([
                p_bar_init[0], p_bar_init[1], 0.0, 0.0
            ])
        else:
            self.x_hat = np.zeros(4)

        self.P = self._P_init.copy()
        self._state_buffer.clear()
        self._cov_buffer.clear()
        self._step_count = 0
        self._prev_pbar_dot_cam = np.zeros(2)

        # Fill buffer with initial state
        for _ in range(self.delay + 1):
            self._state_buffer.append(self.x_hat.copy())
            self._cov_buffer.append(self.P.copy())

    def predict(self, pbar_dot_cam: np.ndarray = None) -> np.ndarray:
        """Perform the prediction step: propagate state forward by one dt.

        Standard (constant-velocity) form:
            x_{k|k-1} = F · x_{k-1|k-1}
            P_{k|k-1} = F · P_{k-1|k-1} · Fᵀ + Q

        IMU-aware form (B-minimal — Stage 3a-noisy DKF upgrade):
        When `pbar_dot_cam` is provided, it represents the predicted image-plane
        velocity induced by camera ego-motion (computed from the interaction
        matrix and current body velocity/angular rates). The constant-velocity
        assumption is then applied to the TARGET'S contribution — the residual
        after subtracting the camera contribution — rather than to the total
        image velocity. This is much more physically realistic: target motion
        changes slowly, but camera-induced image motion changes abruptly when
        the drone pitches/rolls during aggressive maneuvers. Without this, the
        filter mispredicts during fast attitude changes and the agent loses
        the target (the failure mode at delay=3, sigma=0.03).

        Update equations under IMU-aware prediction:
            Δ_cam = pbar_dot_cam_k - pbar_dot_cam_{k-1}
            ṗ̄_k|k-1 = ṗ̄_{k-1|k-1} + Δ_cam      # carry forward delta from IMU
            p̄_k|k-1 = p̄_{k-1|k-1} + dt · ṗ̄_{k-1|k-1} + 0.5·dt·Δ_cam

        Args:
            pbar_dot_cam: np.ndarray (2,) — predicted camera-motion contribution
                to image-plane velocity at the current step, computed externally
                via L_s · [v_cam, ω_cam]. If None, fall back to pure CV prediction.

        Returns:
            np.ndarray (4,): Predicted state [p̄_x, p̄_y, dp̄_x/dt, dp̄_y/dt].
        """
        # Standard CV propagation
        self.x_hat = self.F @ self.x_hat

        # IMU-aware feedforward: carry forward the change in camera contribution
        if pbar_dot_cam is not None:
            pbar_dot_cam = np.asarray(pbar_dot_cam, dtype=np.float64)
            delta_cam = pbar_dot_cam - self._prev_pbar_dot_cam
            # Velocity state: add delta from camera (target's contribution stays constant)
            self.x_hat[2:] += delta_cam
            # Position state: mid-point correction from the velocity change
            self.x_hat[:2] += 0.5 * self.dt * delta_cam
            self._prev_pbar_dot_cam = pbar_dot_cam.copy()

        self.P = self.F @ self.P @ self.F.T + self.Q

        # Clamp covariance to prevent overflow
        self.P = np.clip(self.P, -1e6, 1e6)

        # Clamp state to prevent runaway estimates
        self.x_hat[:2] = np.clip(self.x_hat[:2], -5.0, 5.0)   # image pos bound
        self.x_hat[2:] = np.clip(self.x_hat[2:], -50.0, 50.0)  # image vel bound

        # Store prediction in buffer
        self._state_buffer.append(self.x_hat.copy())
        self._cov_buffer.append(self.P.copy())

        self._step_count += 1
        return self.x_hat.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """Perform the measurement update with delay compensation.

        The measurement z corresponds to the true image position at time
        (k - D), not time k. We use the stored state at time (k - D) to
        compute the innovation, then propagate the correction forward.

        Simplified delay compensation:
          1. Compute innovation against the state predicted at time (k-D)
          2. Compute Kalman gain using the covariance at time (k-D)
          3. Apply the gain-weighted innovation directly to the CURRENT state
             (this is an approximation that works well for small delays)

        Args:
            z: Noisy, delayed measurement [p̄_x, p̄_y] at time k-D.
               Shape (2,).

        Returns:
            np.ndarray (4,): Updated current state estimate.
        """
        z = np.asarray(z, dtype=np.float64)

        # Retrieve the predicted state at time (k - D)
        if len(self._state_buffer) > self.delay:
            # Index -1 is current, -(delay+1) is the delayed state
            idx = -(self.delay + 1)
            x_delayed = self._state_buffer[idx]
            P_delayed = self._cov_buffer[idx]
        else:
            # Not enough history yet — use current state
            x_delayed = self.x_hat
            P_delayed = self.P

        # Innovation: difference between measurement and predicted position
        y = z - self.H @ x_delayed  # shape (2,)

        # Innovation covariance at the delayed time
        S = self.H @ P_delayed @ self.H.T + self.R  # shape (2, 2)

        # Kalman gain (using delayed covariance)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular S — skip update
            return self.x_hat.copy()

        # Propagated Kalman gain: apply delayed gain to current state
        # K_current ≈ P_current · Hᵀ · S⁻¹  (approximate for small D)
        K = self.P @ self.H.T @ S_inv  # shape (4, 2)

        # State update
        self.x_hat = self.x_hat + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ self.H
        P_new = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Guard against numerical overflow / NaN in covariance
        if np.any(np.isnan(P_new)) or np.any(np.isinf(P_new)):
            # Reset to predicted covariance (skip this update)
            pass
        else:
            self.P = np.clip(P_new, -1e6, 1e6)

        # Clamp state
        self.x_hat[:2] = np.clip(self.x_hat[:2], -5.0, 5.0)
        self.x_hat[2:] = np.clip(self.x_hat[2:], -50.0, 50.0)

        # Guard state NaN
        if np.any(np.isnan(self.x_hat)):
            self.x_hat = np.zeros(4)

        # Update buffer with corrected state
        if len(self._state_buffer) > 0:
            self._state_buffer[-1] = self.x_hat.copy()
            self._cov_buffer[-1] = self.P.copy()

        return self.x_hat.copy()

    def step(self, z: np.ndarray = None,
             pbar_dot_cam: np.ndarray = None) -> dict:
        """Perform one full DKF step: predict + update (if measurement available).

        This is the main method to call at each timestep.

        Args:
            z: Noisy, delayed measurement [p̄_x, p̄_y]. Shape (2,).
               If None, only prediction is performed (no measurement).
            pbar_dot_cam: Optional (2,) — predicted camera-motion image velocity
               for IMU-aware prediction (see `predict` docstring).

        Returns:
            dict with:
                'x_hat'  : np.ndarray (4,) — current state estimate
                'p_bar'  : np.ndarray (2,) — estimated image position
                'dp_bar' : np.ndarray (2,) — estimated image velocity
                'P_diag' : np.ndarray (4,) — diagonal of covariance
        """
        # Prediction
        self.predict(pbar_dot_cam=pbar_dot_cam)

        # Update (if measurement available)
        if z is not None:
            self.update(z)

        return {
            'x_hat': self.x_hat.copy(),
            'p_bar': self.x_hat[:2].copy(),
            'dp_bar': self.x_hat[2:].copy(),
            'P_diag': np.diag(self.P).copy(),
        }

    def get_state(self) -> np.ndarray:
        """Return current state estimate.

        Returns:
            np.ndarray (4,): [p̄_x, p̄_y, dp̄_x/dt, dp̄_y/dt]
        """
        return self.x_hat.copy()

    def get_position(self) -> np.ndarray:
        """Return estimated image position.

        Returns:
            np.ndarray (2,): [p̄_x, p̄_y]
        """
        return self.x_hat[:2].copy()

    def get_velocity(self) -> np.ndarray:
        """Return estimated image velocity.

        Returns:
            np.ndarray (2,): [dp̄_x/dt, dp̄_y/dt]
        """
        return self.x_hat[2:].copy()
