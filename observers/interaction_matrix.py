"""
IBVS Interaction Matrix (Image Jacobian)
=========================================
Implements the interaction matrix L_s that relates image feature velocity
to the camera spatial velocity for a point target, following the standard
IBVS formulation and Section II-A.4 of arXiv:2404.08296.

Core equation (static target, camera moves):

    ṗ̄ = L_s · ᶜv

where:
    ṗ̄ = [dp̄_x/dt, dp̄_y/dt]ᵀ    — image feature velocity (2×1)
    ᶜv = [v_x, v_y, v_z, ω_x, ω_y, ω_z]ᵀ  — camera spatial velocity (6×1)
    L_s ∈ ℝ²ˣ⁶                  — interaction matrix

For a MOVING target, the equation generalizes to:

    ṗ̄ = L_trans(p̄) · (1/z_c) · v_rel + L_rot(p̄) · ω_c

where:
    v_rel = v_target - v_camera    (relative translational velocity in camera frame)
    ω_c                            (camera angular velocity in camera frame)
    z_c                            (depth: distance along optical axis)

Key insight: the rotational contribution is depth-INDEPENDENT.
Therefore, after subtracting the known rotational part, we can
estimate depth from the translational residual.

Coordinate convention:
    Camera frame: x-right, y-down, z-forward (optical axis)
    NED body:     x-forward, y-right, z-down
"""

import numpy as np


class InteractionMatrix:
    """IBVS interaction matrix computation and depth estimation.

    The interaction matrix (image Jacobian) L_s encodes how the 2D image
    coordinates of a target point change in response to the 6-DOF camera
    motion. For a point at normalized image coordinates p̄ = (p̄_x, p̄_y)
    and depth z_c:

        L_s = | -1/z  0    p̄_x/z   p̄_x·p̄_y   -(1+p̄_x²)   p̄_y  |
              |  0   -1/z  p̄_y/z   (1+p̄_y²)   -p̄_x·p̄_y  -p̄_x  |

    This class provides:
        1. Full L_s computation
        2. Decomposition into L_trans (depth-dependent) and L_rot (depth-free)
        3. Instantaneous depth estimation from measured image velocity
           and known ego-motion
    """

    @staticmethod
    def compute(p_bar: np.ndarray, z_c: float) -> np.ndarray:
        """Compute the full 2×6 interaction matrix L_s.

        Args:
            p_bar: Normalized image coordinates [p̄_x, p̄_y]. Shape (2,).
            z_c:   Depth of the target along the camera optical axis (> 0).

        Returns:
            np.ndarray (2, 6): The interaction matrix L_s.

        Raises:
            ValueError: If z_c ≤ 0 (target behind camera).
        """
        if z_c <= 1e-9:
            raise ValueError(f"Depth z_c must be positive, got {z_c:.6f}")

        px, py = float(p_bar[0]), float(p_bar[1])
        rho = 1.0 / z_c  # inverse depth

        L_s = np.array([
            [-rho,  0.0,  px * rho,    px * py,    -(1.0 + px**2),  py],
            [ 0.0, -rho,  py * rho,    1.0 + py**2, -px * py,      -px],
        ])
        return L_s

    @staticmethod
    def compute_rotational(p_bar: np.ndarray) -> np.ndarray:
        """Compute the depth-independent rotational part of L_s.

        The rotational sub-matrix L_rot ∈ ℝ²ˣ³ relates image velocity
        to the camera's angular velocity ONLY. It does not depend on depth:

            ṗ̄_rot = L_rot · ω_c

        This is the key property that enables depth estimation: we can
        subtract L_rot · ω_c from the measured ṗ̄ to isolate the
        depth-dependent translational contribution.

        Args:
            p_bar: Normalized image coordinates [p̄_x, p̄_y]. Shape (2,).

        Returns:
            np.ndarray (2, 3): Rotational interaction sub-matrix L_rot.
        """
        px, py = float(p_bar[0]), float(p_bar[1])

        L_rot = np.array([
            [px * py,       -(1.0 + px**2),  py],
            [1.0 + py**2,   -px * py,       -px],
        ])
        return L_rot

    @staticmethod
    def compute_translational_coefficient(p_bar: np.ndarray) -> np.ndarray:
        """Compute the translational coefficient matrix B(p̄) such that:

            ṗ̄_trans = (1/z_c) · B(p̄) · v_rel

        where v_rel is the relative translational velocity (target - camera)
        in the camera frame.

        After separating the rotational contribution:
            ṗ̄_trans = ṗ̄ - L_rot · ω_c

        And B(p̄) does NOT depend on depth:
            B(p̄) = | 1   0   -p̄_x |
                    | 0   1   -p̄_y |

        This makes the equation LINEAR in the scalar ρ = 1/z_c:
            ṗ̄_trans = ρ · B(p̄) · v_rel = ρ · b

        where b = B(p̄) · v_rel is a known 2-vector (if v_rel is known).

        Args:
            p_bar: Normalized image coordinates [p̄_x, p̄_y]. Shape (2,).

        Returns:
            np.ndarray (2, 3): Translational coefficient matrix B.
        """
        px, py = float(p_bar[0]), float(p_bar[1])

        B = np.array([
            [1.0,  0.0,  -px],
            [0.0,  1.0,  -py],
        ])
        return B

    @staticmethod
    def estimate_inverse_depth(
        p_bar: np.ndarray,
        p_bar_dot: np.ndarray,
        v_cam: np.ndarray,
        omega_cam: np.ndarray,
        v_target_cam: np.ndarray = None,
    ) -> dict:
        """Estimate inverse depth ρ = 1/z_c from the image Jacobian equation.

        Given:
            - Measured image velocity ṗ̄
            - Known camera angular velocity ω_c
            - Known (or estimated) translational velocities
            - Current image coordinates p̄

        Procedure:
            1. Subtract the rotational contribution:
                 ṗ̄_trans = ṗ̄ - L_rot(p̄) · ω_c
            2. Compute the translational coefficient:
                 b = B(p̄) · v_rel
               where v_rel = v_target - v_camera (in camera frame).
               If v_target is unknown, use v_rel ≈ -v_cam (static target).
            3. Solve for ρ via least squares:
                 ṗ̄_trans = ρ · b
                 ρ = (bᵀ · ṗ̄_trans) / (bᵀ · b)

        Args:
            p_bar:         Normalized image coordinates [p̄_x, p̄_y]. Shape (2,).
            p_bar_dot:     Image feature velocity [dp̄_x/dt, dp̄_y/dt]. Shape (2,).
            v_cam:         Camera translational velocity in camera frame. Shape (3,).
            omega_cam:     Camera angular velocity in camera frame. Shape (3,).
            v_target_cam:  Target velocity in camera frame. Shape (3,) or None.
                           If None, the target is assumed stationary (v_rel = -v_cam).

        Returns:
            dict with keys:
                'rho'         : float — estimated inverse depth (1/z_c).
                                Positive means target is in front.
                'z_c'         : float — estimated depth (z_c = 1/ρ). Inf if ρ ≈ 0.
                'confidence'  : float — ||b||², indicating how reliable the
                                estimate is. Low values mean poor observability
                                (e.g., camera moving along optical axis only).
                'p_bar_trans' : np.ndarray (2,) — translational residual.
                'b'           : np.ndarray (2,) — the expected translational
                                image velocity direction.
        """
        p_bar = np.asarray(p_bar, dtype=np.float64)
        p_bar_dot = np.asarray(p_bar_dot, dtype=np.float64)
        v_cam = np.asarray(v_cam, dtype=np.float64)
        omega_cam = np.asarray(omega_cam, dtype=np.float64)

        # --- 1. Subtract rotational contribution ---
        L_rot = InteractionMatrix.compute_rotational(p_bar)
        p_bar_dot_rot = L_rot @ omega_cam
        p_bar_dot_trans = p_bar_dot - p_bar_dot_rot

        # --- 2. Compute relative velocity ---
        if v_target_cam is not None:
            v_rel = np.asarray(v_target_cam, dtype=np.float64) - v_cam
        else:
            # Assume static target: v_rel = 0 - v_cam = -v_cam
            v_rel = -v_cam

        # --- 3. Compute b = B(p̄) · v_rel ---
        B = InteractionMatrix.compute_translational_coefficient(p_bar)
        b = B @ v_rel  # shape (2,)

        # --- 4. Solve for ρ = 1/z_c ---
        b_dot_b = np.dot(b, b)  # = ||b||²
        confidence = b_dot_b

        if b_dot_b < 1e-10:
            # Poor observability: camera is not translating (or moving
            # directly along optical axis with p̄ ≈ 0). Cannot estimate depth.
            return {
                'rho': 0.0,
                'z_c': float('inf'),
                'confidence': confidence,
                'p_bar_trans': p_bar_dot_trans,
                'b': b,
            }

        rho = np.dot(b, p_bar_dot_trans) / b_dot_b

        # Depth must be positive (target in front of camera)
        if rho < 1e-9:
            z_c_est = float('inf')
        else:
            z_c_est = 1.0 / rho

        return {
            'rho': float(rho),
            'z_c': float(z_c_est),
            'confidence': float(confidence),
            'p_bar_trans': p_bar_dot_trans,
            'b': b,
        }

    @staticmethod
    def compute_camera_velocity(
        v_interceptor_efcs: np.ndarray,
        omega_body: np.ndarray,
        R_b_e: np.ndarray,
        R_c_b: np.ndarray,
    ) -> tuple:
        """Transform interceptor velocity to camera-frame quantities.

        Computes the camera's translational and angular velocity
        expressed in the camera coordinate frame. These are needed
        as inputs to the depth estimation.

        Pipeline:
            v_cam   = R_c^b · (R_b^e)ᵀ · v_interceptor^e
            ω_cam   = R_c^b · ω_body

        Args:
            v_interceptor_efcs: Interceptor velocity in EFCS. Shape (3,).
            omega_body:         Angular velocity in body frame [p, q, r]. Shape (3,).
            R_b_e:              Body-to-earth rotation matrix R_b^e. Shape (3, 3).
            R_c_b:              Camera-to-body rotation matrix R_c^b. Shape (3, 3).
                                Transforms body-frame vectors into camera frame.

        Returns:
            (v_cam, omega_cam): Tuple of translational and angular velocity,
                                both in camera frame. Each shape (3,).
        """
        R_e_b = R_b_e.T           # earth-to-body
        R_e_c = R_c_b @ R_e_b    # earth-to-camera

        v_cam = R_e_c @ v_interceptor_efcs
        omega_cam = R_c_b @ omega_body

        return v_cam, omega_cam
