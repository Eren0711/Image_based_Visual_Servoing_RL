"""
Pinhole Camera Model — Stage 1
================================
Implements the perspective projection from 3D world coordinates to 2D
normalized image coordinates, following Section II-A.4 of the paper.

Coordinate convention: NED (z-down) for earth and body frames.
Camera frame convention (standard CV):
    - camera x: image right
    - camera y: image down
    - camera z: optical axis (forward/depth)

Key equations from the paper:
    - LOS vector (Eq. 4):  n_t = -p_r / ||p_r||
    - Pinhole projection:  p̄ = [x_c/z_c, y_c/z_c]^T
    - FOV constraint (Eq. 5):
        |arctan(x_c1/x_c3)| ≤ α_hfov/2
        |arctan(x_c2/x_c3)| ≤ α_vfov/2

The camera-to-body rotation R_c^b is configurable (Contribution (i) of the
paper: scheme is suitable for any camera installation angle).

Reference: arXiv:2404.08296, Section II-A.4
"""

import numpy as np
from scipy.spatial.transform import Rotation


class PinholeCamera:
    """Pinhole camera model with FOV checking.

    The camera is rigidly mounted on the interceptor body (strapdown camera).
    Assumption 1 from the paper: the origin of CCS coincides with the origin
    of BCS (translation t_c^b = 0), and R_c^b is constant.

    Attributes:
        f_oc: Focal length (normalized units).
        alpha_hfov: Horizontal field of view (radians).
        alpha_vfov: Vertical field of view (radians).
        R_c_b: 3×3 rotation matrix from body frame to camera frame.
    """

    def __init__(self, config: dict):
        """Initialize the pinhole camera model.

        Args:
            config: Dictionary with keys from config.yaml['camera'].
                Required: f_oc, alpha_hfov, alpha_vfov, R_c_b_euler
        """
        self.f_oc = config['f_oc']
        self.alpha_hfov = config['alpha_hfov']
        self.alpha_vfov = config['alpha_vfov']

        # Build camera-to-body rotation from Euler angles
        # R_c_b_euler = [roll, pitch, yaw] defining the rotation from
        # body frame to camera frame.
        euler = config['R_c_b_euler']
        # R_c^b: rotation that transforms body-frame vectors into camera frame
        # Convention: extrinsic XYZ (equivalent to intrinsic ZYX)
        self._R_c_b_scipy = Rotation.from_euler('XYZ', euler)
        self.R_c_b = self._R_c_b_scipy.as_matrix()  # 3×3

        # Precompute half-FOV tangent limits for fast FOV checking
        self._tan_half_hfov = np.tan(self.alpha_hfov / 2.0)
        self._tan_half_vfov = np.tan(self.alpha_vfov / 2.0)

    def project(self, p_r_efcs: np.ndarray, R_b_e: np.ndarray) -> dict:
        """Project the target onto the camera image plane.

        Pipeline:
        1. Compute vector from interceptor to target in EFCS: d = -p_r
           (since p_r = p_interceptor - p_target, the target direction is -p_r)
        2. Transform to camera frame: d_cam = R_e^c @ d
           where R_e^c = R_c^b @ (R_b^e)^T = R_c^b @ R_e^b
        3. Check that target is in front of camera (z_c > 0)
        4. Apply pinhole projection: p̄ = [x_c/z_c, y_c/z_c]
        5. Check FOV constraints

        Args:
            p_r_efcs: Relative position vector p_r = p_interceptor - p_target
                      in EFCS. Shape (3,).
            R_b_e: Body-to-earth rotation matrix R_b^e. Shape (3, 3).

        Returns:
            dict with keys:
                'p_bar'      : np.ndarray (2,) — normalized image coords
                                [p̄_x, p̄_y]. Zero if target not visible.
                'in_fov'     : bool — True if target is within the FOV.
                'fov_margin' : float — minimum margin to FOV boundary,
                                normalized to [0, 1]. 1 = center, 0 = edge.
                'n_t'        : np.ndarray (3,) — LOS unit vector from
                                interceptor toward target in EFCS.
                'p_camera'   : np.ndarray (3,) — target position in camera
                                frame.
                'depth'      : float — distance along optical axis (z_c).
        """
        p_r = np.asarray(p_r_efcs, dtype=np.float64)
        R_be = np.asarray(R_b_e, dtype=np.float64)

        # --- 1. Direction from interceptor to target in EFCS ---
        d_efcs = -p_r  # vector pointing toward target

        rel_dist = np.linalg.norm(p_r)

        # LOS unit vector (Eq. 4): n_t = -p_r / ||p_r||
        if rel_dist > 1e-9:
            n_t = d_efcs / rel_dist
        else:
            n_t = np.array([1.0, 0.0, 0.0])

        # --- 2. Transform to camera frame ---
        # R_e^b = (R_b^e)^T
        R_e_b = R_be.T
        # R_e^c = R_b^c @ R_e^b = R_c_b @ R_e^b
        #   (R_c_b maps body→camera, R_e_b maps earth→body)
        R_e_c = self.R_c_b @ R_e_b
        p_camera = R_e_c @ d_efcs

        # --- 3. Check target is in front of camera ---
        z_c = p_camera[2]
        if z_c <= 1e-9:
            # Target is behind the camera
            return {
                'p_bar': np.zeros(2),
                'in_fov': False,
                'fov_margin': 0.0,
                'n_t': n_t,
                'p_camera': p_camera,
                'depth': 0.0,
            }

        # --- 4. Pinhole projection ---
        x_c = p_camera[0]
        y_c = p_camera[1]
        p_bar_x = x_c / z_c  # normalized image x
        p_bar_y = y_c / z_c  # normalized image y
        p_bar = np.array([p_bar_x, p_bar_y])

        # --- 5. FOV constraint check (Eq. 5) ---
        angle_h = np.abs(np.arctan2(x_c, z_c))  # horizontal angle
        angle_v = np.abs(np.arctan2(y_c, z_c))  # vertical angle
        half_hfov = self.alpha_hfov / 2.0
        half_vfov = self.alpha_vfov / 2.0

        in_fov = (angle_h <= half_hfov) and (angle_v <= half_vfov)

        # Compute FOV margin: how far from the boundary (0 = edge, 1 = center)
        if in_fov:
            margin_h = 1.0 - (angle_h / half_hfov) if half_hfov > 0 else 1.0
            margin_v = 1.0 - (angle_v / half_vfov) if half_vfov > 0 else 1.0
            fov_margin = min(margin_h, margin_v)
        else:
            fov_margin = 0.0

        return {
            'p_bar': p_bar,
            'in_fov': in_fov,
            'fov_margin': fov_margin,
            'n_t': n_t,
            'p_camera': p_camera,
            'depth': float(z_c),
        }

    def get_R_c_b(self) -> np.ndarray:
        """Return the camera-to-body rotation matrix R_c^b.

        This is the constant rotation that maps body-frame vectors into the
        camera coordinate system.

        Returns:
            np.ndarray (3, 3) — rotation matrix.
        """
        return self.R_c_b.copy()

    def get_fov_params(self) -> dict:
        """Return FOV parameters for visualization / debugging.

        Returns:
            dict with 'alpha_hfov', 'alpha_vfov', 'f_oc',
                       'tan_half_hfov', 'tan_half_vfov'.
        """
        return {
            'alpha_hfov': self.alpha_hfov,
            'alpha_vfov': self.alpha_vfov,
            'f_oc': self.f_oc,
            'tan_half_hfov': self._tan_half_hfov,
            'tan_half_vfov': self._tan_half_vfov,
        }
