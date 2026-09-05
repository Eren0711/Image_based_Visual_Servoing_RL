"""
Interceptor Drone Dynamics — Stage 3a: First-Order Inertia + Attitude Coupling
===============================================================================
Introduces physical realism into the drone's motion model.

Stage 2b → 3a changes:
    BEFORE: v_body(t+1) = clip(v_body(t) + a_cmd*dt, -v_max, v_max)
            pitch = 0, roll = 0  (kinematically level always)

    AFTER:  v_body(t+1) = v_body(t) + dt/τ * (v_cmd - v_body(t))
            v_cmd = clip(v_body + a_cmd*dt, -v_max, v_max)   ← commanded velocity
            pitch = -arctan(a_fwd / g)                        ← nose tilts with accel
            roll  =  arctan(a_lat / g)                        ← banks with lateral accel

This is a first-order lag model with time constant τ:
  - At τ=0.2s the drone takes ~200ms to respond to a step command
  - The drone "remembers" its current velocity and cannot stop instantly
  - Pitch and roll are DERIVED from acceleration, not free variables
  - Camera is body-fixed → tilted attitude moves the target on the image plane

Physical significance:
  - This is the first stage where pitch-camera coupling exists
  - Aggressive acceleration = nose-down = target drifts upward in FOV
  - The agent must learn this tradeoff to maintain visual lock

Coordinate convention: NED (z-down)
  - x: North (forward body)
  - y: East  (right body)
  - z: Down

Reference: Section II-A.2 of arXiv:2404.08296
"""

import numpy as np
from scipy.spatial.transform import Rotation


class InterceptorDrone:
    """First-order inertia interceptor drone with attitude-camera coupling.

    State:
        position    : np.ndarray (3,) — position in EFCS [x_e, y_e, z_e]
        velocity    : np.ndarray (3,) — velocity in EFCS
        _body_vel   : np.ndarray (3,) — velocity in body frame (persistent)
        yaw         : float           — heading angle (rad)
        pitch       : float           — pitch angle (rad), derived from fwd accel
        roll        : float           — roll angle (rad), derived from lat accel

    Stage 3a key differences from Stage 1b-2b:
        1. Velocity changes through a first-order lag (τ time constant).
        2. Pitch and roll are computed from the commanded acceleration,
           not fixed at zero.
        3. The body-to-earth rotation matrix now incorporates non-zero
           pitch and roll, which rotates the camera and changes where
           the target appears in the image.
    """

    def __init__(self, config: dict):
        """Initialize interceptor drone.

        Args:
            config: Interceptor section of the active configuration.
                Required: v_max, a_max, yaw_rate_max, dt
                Stage 3a: tau_velocity, max_pitch_deg, max_roll_deg (optional)
        """
        self.v_max        = config['v_max']
        self.a_max        = config['a_max']
        self.yaw_rate_max = config['yaw_rate_max']
        self.dt           = config['dt']

        # --- Stage 3a: inertia parameters ---
        # τ: velocity time constant (s). Smaller = more responsive but less realistic.
        # At 50Hz (dt=0.02s): τ=0.15s → ~7 steps to reach 63% of target velocity
        self.tau_vel = float(config.get('tau_velocity', 0.15))

        # Stage 3a-v2: yaw rate also gets a first-order lag.
        # v1 had no smoothing on yaw_rate, which produced bang-bang oscillation
        # because there was no physical cost for instantaneous reversals.
        self.tau_yaw = float(config.get('tau_yaw_rate', 0.10))

        # Attitude limits (for clamping derived pitch/roll)
        max_pitch_deg = float(config.get('max_pitch_deg', 35.0))
        max_roll_deg  = float(config.get('max_roll_deg',  35.0))
        self.max_pitch = np.deg2rad(max_pitch_deg)
        self.max_roll  = np.deg2rad(max_roll_deg)

        # --- State variables (initialized in reset) ---
        self.position  = np.zeros(3)
        self.velocity  = np.zeros(3)
        self._body_vel = np.zeros(3)   # persistent body-frame velocity
        self._v_cmd    = np.zeros(3)   # commanded body-frame velocity target
        self._yaw_rate = 0.0           # filtered (lagged) yaw rate
        self.yaw   = 0.0
        self.pitch = 0.0
        self.roll  = 0.0
        self._rotation = Rotation.identity()

        # Acceleration (for attitude computation and depth estimator)
        self._body_accel = np.zeros(3)

    def reset(self, position: np.ndarray, yaw: float) -> None:
        """Reset the drone to an initial state (stationary, level).

        Args:
            position: Initial position in EFCS [x_e, y_e, z_e].
            yaw:      Initial heading angle (rad).
        """
        self.position  = np.array(position, dtype=np.float64)
        self.velocity  = np.zeros(3)
        self._body_vel = np.zeros(3)
        self._v_cmd    = np.zeros(3)
        self._yaw_rate = 0.0
        self.yaw   = float(yaw)
        self.pitch = 0.0
        self.roll  = 0.0
        self._body_accel = np.zeros(3)
        self._update_rotation()

    def step(self, action: np.ndarray) -> None:
        """Integrate one timestep with first-order inertia and attitude coupling.

        Physics (Stage 3a):
          1. Compute commanded velocity target from action (acceleration command)
          2. First-order lag: v_body approaches v_cmd with time constant τ
          3. Derive pitch/roll from current commanded acceleration
          4. Update yaw
          5. Rebuild rotation matrix (now with non-zero pitch/roll)
          6. Transform body velocity → EFCS velocity
          7. Euler integrate position

        Key difference from Stage 1b-2b:
          - v_body does NOT instantly equal the commanded target
          - Pitch and roll are now non-zero and affect camera pointing

        Args:
            action: np.ndarray (4,) — [a_x, a_y, a_z, yaw_rate] in body frame.
                    Each value scaled to physical units BEFORE calling step().
        """
        action = np.asarray(action, dtype=np.float64)

        # --- 1. Clip and extract commands ---
        a_cmd    = np.clip(action[:3], -self.a_max, self.a_max)
        yaw_rate = np.clip(action[3],  -self.yaw_rate_max, self.yaw_rate_max)

        # --- 2. Compute target velocity (what we WANT, instantly) ---
        v_target = self._body_vel + a_cmd * self.dt
        # Clamp to physical speed limit
        speed_target = np.linalg.norm(v_target)
        if speed_target > self.v_max:
            v_target = v_target * (self.v_max / speed_target)

        # --- 3. First-order lag: approach v_target with time constant τ ---
        # v_body(t+1) = v_body(t) + (dt/τ) * (v_target - v_body(t))
        # As τ → 0: v_body instantly equals v_target (Stage 1b behavior)
        # As τ → ∞: v_body barely moves (unresponsive)
        alpha = self.dt / self.tau_vel   # blending factor ∈ (0, 1] for small dt/τ
        alpha = min(alpha, 1.0)          # ensure we never overshoot
        self._body_accel = (v_target - self._body_vel) * alpha / self.dt
        self._body_vel   = self._body_vel + alpha * (v_target - self._body_vel)

        # Safety clamp
        speed = np.linalg.norm(self._body_vel)
        if speed > self.v_max:
            self._body_vel = self._body_vel * (self.v_max / speed)

        # --- 4. Derive attitude from commanded acceleration ---
        # Physical interpretation: a quadrotor tilts its body to produce
        # horizontal acceleration. The tilt angle equals arctan(a_horiz / g).
        # NED convention: pitch-down = nose forward = positive x-acceleration
        # Roll-right     = positive y-acceleration
        g = 9.81
        # Use actual body acceleration (how fast velocity is actually changing)
        accel_x = self._body_accel[0]  # forward
        accel_y = self._body_accel[1]  # lateral (right)

        # In NED with forward-x convention:
        #   pitch_down (nose drops) when accelerating forward → -arctan(a_x/g)
        #   roll_right (right wing drops) when accelerating right → arctan(a_y/g)
        pitch_desired = np.clip(-np.arctan2(accel_x, g), -self.max_pitch, self.max_pitch)
        roll_desired  = np.clip( np.arctan2(accel_y, g), -self.max_roll,  self.max_roll)

        # First-order attitude tracking (attitude responds faster than velocity)
        tau_att = self.tau_vel * 0.3   # attitude responds ~3× faster
        alpha_att = min(self.dt / tau_att, 1.0)
        self.pitch += alpha_att * (pitch_desired - self.pitch)
        self.roll  += alpha_att * (roll_desired  - self.roll)

        # --- 5. Update yaw (with first-order lag on rate) ---
        # The commanded yaw_rate passes through a τ_yaw lag before integration,
        # so bang-bang commands no longer translate to bang-bang heading.
        alpha_yaw = min(self.dt / self.tau_yaw, 1.0)
        self._yaw_rate += alpha_yaw * (yaw_rate - self._yaw_rate)
        self.yaw += self._yaw_rate * self.dt
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi

        # --- 6. Rebuild rotation matrix (now with real pitch and roll) ---
        self._update_rotation()

        # --- 7. Transform body velocity to EFCS ---
        R_be = self.get_rotation_matrix()
        self.velocity = R_be @ self._body_vel

        # --- 8. Euler integration of position ---
        self.position = self.position + self.velocity * self.dt

    def _update_rotation(self) -> None:
        """Rebuild R_b^e from current [yaw, pitch, roll].

        Convention: ZYX intrinsic Euler angles (aerospace standard)
            R_b^e = Rz(yaw) @ Ry(pitch) @ Rx(roll)

        In Stage 3a pitch and roll are non-zero, so R_b^e now tilts the
        camera frame relative to horizontal — this is the key physical
        change from Stage 1b-2b.
        """
        self._rotation = Rotation.from_euler(
            'ZYX', [self.yaw, self.pitch, self.roll]
        )

    def get_rotation_matrix(self) -> np.ndarray:
        """Return the body-to-earth rotation matrix R_b^e (3×3).

        Transforms vectors from body frame {b} to EFCS {e}:
            v_e = R_b^e @ v_b
        """
        return self._rotation.as_matrix()

    def get_euler_angles(self) -> np.ndarray:
        """Return Euler angles [roll, pitch, yaw] in radians."""
        return np.array([self.roll, self.pitch, self.yaw])

    def get_body_velocity(self) -> np.ndarray:
        """Return the current velocity in the body frame."""
        return self._body_vel.copy()

    def get_body_acceleration(self) -> np.ndarray:
        """Return the current body-frame acceleration estimate.

        Used by the depth estimator (omega_body calculation).
        """
        return self._body_accel.copy()

    def get_state(self) -> dict:
        """Return the full state as a dictionary."""
        return {
            'position':  self.position.copy(),
            'velocity':  self.velocity.copy(),
            'R_be':      self.get_rotation_matrix(),
            'euler':     self.get_euler_angles(),
            'body_vel':  self.get_body_velocity(),
            'body_accel': self.get_body_acceleration(),
            'pitch':     self.pitch,
            'roll':      self.roll,
        }
