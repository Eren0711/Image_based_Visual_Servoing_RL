"""
PD Attitude Controller on SO(3) — Stage 3b inner loop
======================================================
Converts a desired body-frame acceleration command + desired yaw rate into
a desired thrust magnitude and a desired body angular velocity, following
the formulation in Section III-C of arXiv:2404.08296.

Pipeline (per call):
  1. Lift the agent's body-frame acceleration command into EFCS via the
     current R_b^e. This is the desired EFCS acceleration a_d_efcs.
  2. The thrust must produce m · (a_d_efcs - g). Compute its magnitude
     f_d (saturated to [0, f_max]) and direction n_fd (in EFCS).
  3. Build R_d as the smallest rotation that aligns the body thrust axis
     (-body_z) with n_fd, preserving yaw (the rotation has no z-component).
  4. PD law on attitude error: ω_attitude = -K · vex(R_d^T R - R^T R_d).
     This is the standard SO(3) tracking controller from Lee et al. 2010,
     Murray-Li-Sastry, and used in paper Eq. (26).
  5. Override the body-z component with the agent's yaw rate command —
     yaw is its own degree of freedom and the agent controls it directly.

The output (f_d, ω_d) is then passed to Multicopter6DOFLite which tracks
ω_d through a first-order rate-loop (motor + inner-loop bandwidth) and
applies f_d through a first-order thrust response.

Coordinate convention: NED (x-forward, y-right, z-down), thrust along
-body_z (up in body frame).
"""

import numpy as np


def _vex(S: np.ndarray) -> np.ndarray:
    """Inverse of the hat map: extract the 3-vector from a 3x3 skew matrix.

    For S = [[0, -c, b], [c, 0, -a], [-b, a, 0]], returns [a, b, c].
    """
    return 0.5 * np.array([S[2, 1] - S[1, 2],
                            S[0, 2] - S[2, 0],
                            S[1, 0] - S[0, 1]])


def _hat(v: np.ndarray) -> np.ndarray:
    """The hat map: cross-product matrix of a 3-vector."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


class AttitudeController:
    """SO(3)-based attitude controller for the 6-DOF multicopter inner loop.

    Single-shot computation; no internal state required beyond fixed gains
    and parameters. The PD action is purely a function of (R_be, a_body_cmd,
    yaw_rate_cmd).

    Args:
        mass:        Multicopter mass (kg).
        g:           Gravitational acceleration magnitude (m/s²).
        f_max:       Maximum thrust (N).
        k_attitude:  Proportional gain on attitude error (scalar, NaN-safe).
                     Higher = stiffer tracking, more aggressive ω_d commands.
        omega_max:   Saturation on the angular velocity command (rad/s).
    """

    def __init__(self,
                 mass: float = 1.0,
                 g: float = 9.81,
                 f_max: float = 30.0,
                 k_attitude: float = 6.0,
                 omega_max: float = 4.0):
        self.mass = float(mass)
        self.g = float(g)
        self.f_max = float(f_max)
        self.k_attitude = float(k_attitude)
        self.omega_max = float(omega_max)

        # Gravity vector in EFCS (NED: z points down, so g vector is +z·g)
        self._g_vec = np.array([0.0, 0.0, self.g])

        # Body-frame thrust axis is -e_3 (thrust pushes "up" in body, which is
        # -z given NED axes). We'll need this for tilt rotation construction.
        self._e3 = np.array([0.0, 0.0, 1.0])

    def compute(self,
                a_body_cmd: np.ndarray,
                yaw_rate_cmd: float,
                R_be: np.ndarray) -> tuple:
        """Compute (desired thrust, desired body angular velocity).

        Args:
            a_body_cmd:    (3,) body-frame acceleration command (m/s²).
                           These are the agent's a_x, a_y, a_z commands as
                           seen during Stage 3a — same semantics, so warm-
                           start transfers naturally.
            yaw_rate_cmd:  Scalar yaw rate command (rad/s) in body frame.
            R_be:          (3, 3) current body-to-earth rotation matrix.

        Returns:
            f_d:     Desired thrust magnitude (N), saturated to [0, f_max].
            omega_d: (3,) desired body-frame angular velocity (rad/s),
                     saturated component-wise to [-omega_max, omega_max].
        """
        a_body_cmd = np.asarray(a_body_cmd, dtype=np.float64)

        # --- 1. Body-frame command → EFCS desired acceleration ---
        a_d_efcs = R_be @ a_body_cmd

        # --- 2. Required net thrust force (in EFCS), then magnitude/direction ---
        # F_thrust = m * (a_d - g_vec). Note that g_vec is the gravity vector
        # ([0, 0, g] in NED), so we subtract it to get what thrust must add.
        f_required = self.mass * (a_d_efcs - self._g_vec)
        # In NED with thrust along -body_z, "up" thrust direction is -z in EFCS
        # when the drone is level. f_required's z-component should be negative
        # for net upward motion (hovering needs f_required = [0, 0, -m*g]).
        # The thrust magnitude is ||f_required|| with the convention that
        # positive thrust pushes in the +n_fd direction.
        f_mag = float(np.linalg.norm(f_required))
        if f_mag < 1e-6:
            # Degenerate command (e.g., free-fall — agent commanded
            # exactly -g). Default to current attitude and zero thrust.
            n_fd = -R_be @ self._e3  # current thrust direction in EFCS
            f_d = 0.0
        else:
            n_fd = f_required / f_mag
            f_d = float(np.clip(f_mag, 0.0, self.f_max))

        # --- 3. Desired rotation R_d: tilt that aligns -body_z with n_fd ---
        # Current thrust axis direction in EFCS: -R_be @ e_3 (= -R_be[:, 2])
        n_f_current = -R_be @ self._e3

        # Tilt rotation: rotates n_f_current to n_fd, axis = cross / sin(angle)
        c = float(np.dot(n_f_current, n_fd))
        c = float(np.clip(c, -1.0, 1.0))
        cross = np.cross(n_f_current, n_fd)
        s = float(np.linalg.norm(cross))

        if s < 1e-6:
            # Already aligned (or 180° opposite — but that requires a
            # negative-thrust command, which we clipped above). R_tilt = I.
            R_tilt = np.eye(3)
        else:
            axis = cross / s
            # Rodrigues' formula for rotation by angle θ with cos=c, sin=s
            K = _hat(axis)
            R_tilt = np.eye(3) + s * K + (1.0 - c) * (K @ K)

        # Compose: R_d rotates body axes so the thrust axis points along n_fd.
        # Applied as a *pre-multiplication* on R_be in the EFCS frame.
        R_d = R_tilt @ R_be

        # --- 4. SO(3) PD law on attitude error → desired angular velocity ---
        # ω = -k · vex(R_d^T · R_be - R_be^T · R_d)
        # This drives the body attitude toward R_d in the absence of yaw input.
        err_mat = R_d.T @ R_be - R_be.T @ R_d
        omega_attitude_body = -self.k_attitude * _vex(err_mat)

        # --- 5. Replace body-z component with the agent's yaw rate command ---
        # Yaw is a free DOF; the agent controls it directly via yaw_rate_cmd.
        # The roll/pitch components from the PD law steer toward R_d (which
        # is the "tilt to produce the desired acceleration" target).
        omega_d = omega_attitude_body.copy()
        omega_d[2] = float(yaw_rate_cmd)

        # --- 6. Saturate ω_d component-wise ---
        omega_d = np.clip(omega_d, -self.omega_max, self.omega_max)

        return f_d, omega_d
