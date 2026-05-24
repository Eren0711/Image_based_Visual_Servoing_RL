"""
Interceptor Drone Kinematic Model — Stage 1
============================================
Simplified kinematic model for the interceptor multicopter.
The agent commands body-frame velocities directly; pitch/roll are implicit
(heading-only orientation for Stage 1).

Coordinate convention: NED (z-down)
  - x: North (forward)
  - y: East  (right)
  - z: Down

Reference: Section II-A.2 of arXiv:2404.08296
  Full 6-DOF model (Eq. 1) is simplified here to kinematic velocity control.
  The rotation matrix R_b^e ∈ SO(3) is maintained via scipy Rotation to
  guarantee orthogonality at all times.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class InterceptorDrone:
    """Simplified kinematic interceptor drone model.

    State:
        position  : np.ndarray (3,) — position in EFCS [x_e, y_e, z_e]
        velocity  : np.ndarray (3,) — velocity in EFCS [vx_e, vy_e, vz_e]
        R_b_e     : Rotation        — body-to-earth rotation (scipy Rotation)
        yaw       : float           — heading angle (rad)

    Action (body frame):
        [v_x_cmd, v_y_cmd, v_z_cmd, yaw_rate_cmd]
        Each component is clipped to physical limits.
    """

    def __init__(self, config: dict):
        """Initialize interceptor drone.

        Args:
            config: Dictionary with keys from config.yaml['interceptor'].
                Required: v_max, yaw_rate_max, dt
        """
        self.v_max = config['v_max']
        self.yaw_rate_max = config['yaw_rate_max']
        self.dt = config['dt']

        # State variables (initialized in reset)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self._rotation = Rotation.identity()

    def reset(self, position: np.ndarray, yaw: float) -> None:
        """Reset the drone to an initial state.

        Args:
            position: Initial position in EFCS [x_e, y_e, z_e].
            yaw: Initial heading angle (rad), measured from North (x-axis)
                 positive toward East (y-axis) in NED.
        """
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.zeros(3)
        self.yaw = float(yaw)
        self.pitch = 0.0
        self.roll = 0.0
        self._update_rotation()

    def step(self, action: np.ndarray) -> None:
        """Integrate one timestep given body-frame velocity commands.

        The action is interpreted as velocity commands in the body frame:
          action[0]: v_x — forward velocity (body x-axis)
          action[1]: v_y — lateral velocity (body y-axis, rightward)
          action[2]: v_z — vertical velocity (body z-axis, downward in NED)
          action[3]: yaw_rate — heading rate (rad/s, positive = turn right)

        Integration:
          1. Clip actions to physical limits
          2. Update yaw angle
          3. Rebuild rotation matrix R_b^e
          4. Transform body velocity → EFCS velocity
          5. Update position via Euler integration

        Args:
            action: np.ndarray (4,) — [v_x, v_y, v_z, yaw_rate] in body frame.
        """
        action = np.asarray(action, dtype=np.float64)

        # --- 1. Clip to physical limits ---
        v_cmd_body = np.clip(action[:3], -self.v_max, self.v_max)
        yaw_rate = np.clip(action[3], -self.yaw_rate_max, self.yaw_rate_max)

        # --- 2. Update yaw angle ---
        self.yaw += yaw_rate * self.dt
        # Wrap yaw to [-π, π]
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi

        # --- 3. Rebuild rotation matrix ---
        self._update_rotation()

        # --- 4. Transform body velocity to EFCS ---
        R_be = self.get_rotation_matrix()  # 3×3 rotation body → earth
        self.velocity = R_be @ v_cmd_body

        # --- 5. Euler integration of position ---
        self.position = self.position + self.velocity * self.dt

    def _update_rotation(self) -> None:
        """Rebuild R_b^e from current Euler angles.

        In Stage 1, only yaw is actively controlled. Pitch and roll are zero
        (kinematic simplification). In later stages, these can be derived from
        the velocity/acceleration vector for a more realistic model.

        Convention: ZYX intrinsic Euler angles → R_b^e
          R_b^e = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        """
        self._rotation = Rotation.from_euler(
            'ZYX', [self.yaw, self.pitch, self.roll]
        )

    def get_rotation_matrix(self) -> np.ndarray:
        """Return the body-to-earth rotation matrix R_b^e as 3×3 ndarray.

        This matrix transforms vectors from the body frame {b} to EFCS {e}:
            v_e = R_b^e @ v_b

        Guaranteed to remain on SO(3) via scipy Rotation internals.
        """
        return self._rotation.as_matrix()

    def get_euler_angles(self) -> np.ndarray:
        """Return Euler angles [roll, pitch, yaw] in radians.

        Convention: ZYX intrinsic → extrinsic XYZ.
        """
        return np.array([self.roll, self.pitch, self.yaw])

    def get_body_velocity(self) -> np.ndarray:
        """Return the current velocity in the body frame.

        v_b = (R_b^e)^T @ v_e = R_e^b @ v_e
        """
        R_eb = self._rotation.inv().as_matrix()
        return R_eb @ self.velocity

    def get_state(self) -> dict:
        """Return the full state as a dictionary.

        Returns:
            dict with keys:
                'position'  : np.ndarray (3,) — EFCS position
                'velocity'  : np.ndarray (3,) — EFCS velocity
                'R_be'      : np.ndarray (3,3) — body-to-earth rotation
                'euler'     : np.ndarray (3,) — [roll, pitch, yaw]
                'body_vel'  : np.ndarray (3,) — velocity in body frame
        """
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'R_be': self.get_rotation_matrix(),
            'euler': self.get_euler_angles(),
            'body_vel': self.get_body_velocity(),
        }
