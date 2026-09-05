"""
6-DOF Multicopter Dynamics — Stage 3b (Lite)
=============================================
Full Newton-Euler equations of motion for a multicopter, plus a first-order
rate-loop response model for the inner attitude controller and a first-order
thrust response for the motor dynamics. Quaternion attitude representation
for numerical stability at large pitch/roll angles (up to ~50° per paper).

State (kept internal; exposed as the same attributes used by Stage 3a):
    position  : (3,) EFCS position
    velocity  : (3,) EFCS velocity
    quat      : (4,) attitude quaternion (scalar-last [x, y, z, w] — scipy
                Rotation convention; or stored via a Rotation object)
    omega_body: (3,) body-frame angular velocity [p, q, r]
    thrust    : scalar current motor thrust (filtered through tau_motor)

Drop-in compatibility with envs/interception_env.py:
    Exposes self.position, self.velocity, self.pitch, self.roll, self.yaw,
    self.max_pitch, self.max_roll, self._yaw_rate, self._body_vel, plus
    get_rotation_matrix(), get_euler_angles(), get_body_velocity(),
    get_body_acceleration(), get_state(). reset() and step() match the
    InterceptorDrone API. Action is the same 4-vector
    [a_x_body, a_y_body, a_z_body, yaw_rate], already scaled to physical
    units before this object sees it.

What's NOT included in "lite":
    - Aerodynamic drag (-R·D·R^T·v term from paper Eq. 1)
    - Motor mixing → individual rotor speeds → propeller forces
    - Gyroscopic torque on motors
    - Air resistance / wind
These add another layer of fidelity that we defer to a full 3b once the lite
version is validated.

What IS included:
    - Newton-Euler translational dynamics with attitude-coupled thrust
    - Quaternion attitude integration (no Euler-angle singularities)
    - First-order rate-loop response: ω̇_body = (1/τ_rate)·(ω_d - ω_body)
      → captures attitude-tracking lag, which is the dominant effect
        differentiating Stage 3b from Stage 3a's "pitch = -arctan(a/g)"
    - First-order thrust response: ḟ = (1/τ_motor)·(f_d - f)
      → captures motor bandwidth limit
    - Saturation of ω_body to physical limits

Coordinate convention: NED (x-North, y-East, z-Down). Thrust acts along
body -z (up in body frame). Gravity vector in EFCS is [0, 0, +g].

Reference: Section II-A.2 of arXiv:2404.08296.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from models.attitude_controller import AttitudeController


class Multicopter6DOFLite:
    """6-DOF multicopter with simplified motor + rate-loop dynamics.

    Drop-in replacement for InterceptorDrone with the same step/reset/getter
    API. Internally runs the AttitudeController inner loop and integrates
    rigid-body translation + quaternion attitude.
    """

    def __init__(self, config: dict):
        """Initialize the 6-DOF model from config.

        Args:
            config: Interceptor section of the active configuration. Must contain
                v_max, a_max, yaw_rate_max, dt, max_pitch_deg, max_roll_deg.
                The 6-DOF-specific parameters live under config['dynamics_6dof']
                with sensible defaults for a ~1 kg quad.
        """
        # Shared parameters (same names as InterceptorDrone)
        self.v_max = float(config['v_max'])
        self.a_max = float(config['a_max'])
        self.yaw_rate_max = float(config['yaw_rate_max'])
        self.dt = float(config['dt'])

        # Attitude limits (used by the env reward function)
        max_pitch_deg = float(config.get('max_pitch_deg', 35.0))
        max_roll_deg = float(config.get('max_roll_deg', 35.0))
        self.max_pitch = np.deg2rad(max_pitch_deg)
        self.max_roll = np.deg2rad(max_roll_deg)

        # 6-DOF parameters
        dyn_cfg = config.get('dynamics_6dof', {})
        self.mass = float(dyn_cfg.get('mass', 1.0))
        # Diagonal inertia tensor (kg·m²). Typical 1 kg quad ≈ diag(0.01, 0.01, 0.02).
        # We use the diagonal scalars; the lite model doesn't need the full
        # 3×3 tensor because we skip torques (rate-loop is direct first-order
        # tracking on ω). Kept here for documentation / future use.
        self.J = np.diag([
            float(dyn_cfg.get('J_xx', 0.01)),
            float(dyn_cfg.get('J_yy', 0.01)),
            float(dyn_cfg.get('J_zz', 0.02)),
        ])
        # Time constants
        self.tau_motor = float(dyn_cfg.get('tau_motor', 0.05))
        self.tau_rate = float(dyn_cfg.get('tau_rate', 0.05))
        # Thrust limits — thrust-to-weight of 3 matches the paper's hardware
        self.thrust_to_weight = float(dyn_cfg.get('thrust_to_weight', 3.0))
        self.f_max = self.thrust_to_weight * self.mass * 9.81
        # Angular velocity saturation
        self.omega_max = float(dyn_cfg.get('omega_max', 4.0))
        # Attitude controller gain
        k_att = float(dyn_cfg.get('k_attitude', 6.0))

        # Inner-loop attitude controller (PD on SO(3) per paper)
        self.attitude_ctrl = AttitudeController(
            mass=self.mass,
            g=9.81,
            f_max=self.f_max,
            k_attitude=k_att,
            omega_max=self.omega_max,
        )

        # State variables (initialized in reset)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)            # EFCS
        self._body_vel = np.zeros(3)           # body-frame velocity
        self._yaw_rate = 0.0                   # for compatibility w/ DKF wrapper
        self._rotation = Rotation.identity()   # current attitude
        self.omega_body = np.zeros(3)
        self.thrust = self.mass * 9.81         # start at hover thrust
        self._body_accel = np.zeros(3)
        # Euler angles cached as attributes (used by env obs builder)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    # ------------------------------------------------------------------ #
    # API methods — same shape as InterceptorDrone for drop-in compatibility
    # ------------------------------------------------------------------ #

    def reset(self, position: np.ndarray, yaw: float) -> None:
        """Reset to a hover state at the given position and heading."""
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.zeros(3)
        self._body_vel = np.zeros(3)
        self._yaw_rate = 0.0
        self.omega_body = np.zeros(3)
        self.thrust = self.mass * 9.81  # hover thrust
        self._body_accel = np.zeros(3)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = float(yaw)
        # Build initial rotation matrix from Euler angles (level + yaw)
        self._rotation = Rotation.from_euler('ZYX', [self.yaw, 0.0, 0.0])

    def step(self, action: np.ndarray) -> None:
        """Integrate one timestep of 6-DOF dynamics.

        Args:
            action: (4,) array [a_x_body, a_y_body, a_z_body, yaw_rate_cmd],
                already scaled to physical units by the env. Body-frame
                acceleration commands and a body-frame yaw rate.
        """
        action = np.asarray(action, dtype=np.float64)
        a_body_cmd = np.clip(action[:3], -self.a_max, self.a_max)
        yaw_rate_cmd = float(np.clip(action[3], -self.yaw_rate_max, self.yaw_rate_max))
        self._yaw_rate = yaw_rate_cmd  # for DKF wrapper compatibility

        # --- 1. Inner-loop attitude controller: (a_body, yaw_rate) → (f_d, ω_d) ---
        R_be = self.get_rotation_matrix()
        f_d, omega_d = self.attitude_ctrl.compute(
            a_body_cmd=a_body_cmd,
            yaw_rate_cmd=yaw_rate_cmd,
            R_be=R_be,
        )

        # --- 2. First-order tracking of desired angular velocity ---
        # Captures rate-loop bandwidth (the "lite" model's stand-in for the
        # motor + rotor-dynamics + torque-control chain).
        alpha_rate = min(self.dt / self.tau_rate, 1.0)
        self.omega_body += alpha_rate * (omega_d - self.omega_body)
        self.omega_body = np.clip(self.omega_body, -self.omega_max, self.omega_max)

        # --- 3. First-order thrust response ---
        alpha_motor = min(self.dt / self.tau_motor, 1.0)
        self.thrust += alpha_motor * (f_d - self.thrust)
        self.thrust = float(np.clip(self.thrust, 0.0, self.f_max))

        # --- 4. Attitude integration: q̇ = 0.5 · q ⊗ ω_body ---
        # Use scipy's Rotation: integrate by composing with a small rotation
        # of magnitude ω·dt around the body angular velocity axis.
        delta_rot = Rotation.from_rotvec(self.omega_body * self.dt)
        self._rotation = self._rotation * delta_rot  # body-frame composition

        # --- 5. Translational dynamics (Newton's 2nd law in EFCS) ---
        # Forces: gravity + thrust (along -body_z in EFCS, i.e., -R_be[:, 2])
        R_be = self.get_rotation_matrix()
        g_vec = np.array([0.0, 0.0, 9.81])         # NED: +z is down → gravity is +g·e_z
        thrust_vec_efcs = -self.thrust * R_be[:, 2]  # thrust pushes along -body_z
        accel_efcs = g_vec + thrust_vec_efcs / self.mass

        # Integrate velocity, clamp to v_max for safety (also limits in env)
        self.velocity += accel_efcs * self.dt
        speed = float(np.linalg.norm(self.velocity))
        if speed > self.v_max:
            self.velocity *= (self.v_max / speed)

        # Update body-frame velocity (used by reward's near-target brake term
        # and the DKF's IMU input)
        self._body_vel = R_be.T @ self.velocity
        # Store EFCS acceleration in body frame for depth-estimator API parity
        self._body_accel = R_be.T @ accel_efcs

        # --- 6. Position update (Euler) ---
        self.position += self.velocity * self.dt

        # --- 7. Cache Euler angles (used by env for obs + reward) ---
        # scipy's 'ZYX' returns [yaw, pitch, roll]
        euler = self._rotation.as_euler('ZYX')
        self.yaw = float(euler[0])
        self.pitch = float(euler[1])
        self.roll = float(euler[2])

    # ------------------------------------------------------------------ #
    # Getters (same shape as InterceptorDrone)
    # ------------------------------------------------------------------ #

    def get_rotation_matrix(self) -> np.ndarray:
        return self._rotation.as_matrix()

    def get_euler_angles(self) -> np.ndarray:
        """[roll, pitch, yaw] in radians."""
        return np.array([self.roll, self.pitch, self.yaw])

    def get_body_velocity(self) -> np.ndarray:
        return self._body_vel.copy()

    def get_body_acceleration(self) -> np.ndarray:
        return self._body_accel.copy()

    def get_state(self) -> dict:
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'R_be': self.get_rotation_matrix(),
            'euler': self.get_euler_angles(),
            'body_vel': self.get_body_velocity(),
            'body_accel': self.get_body_acceleration(),
            'pitch': self.pitch,
            'roll': self.roll,
            'yaw': self.yaw,
            'omega_body': self.omega_body.copy(),
            'thrust': float(self.thrust),
            'thrust_normalized': float(self.thrust / self.f_max),
        }
