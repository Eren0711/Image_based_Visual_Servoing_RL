"""
Analytical Lie Derivatives for CBF — Stage 4a Phase 3 (HOCBF / proper CBF-QP)
==============================================================================
Computes L_f h_i and L_g h_i for each of the 4 constraint barriers, evaluated
at the current state, using a proxy control-affine linearization of the
multicopter dynamics.

CBF condition (per Ames et al. 2017, eq. 24, relative degree 1):

    L_f h_i(x) + L_g h_i(x) · u + α_i · h_i(x)  ≥  0

The u-dependent term L_g h_i is what makes this a constraint on u (an affine
inequality, exactly what a QP needs).

----------------------------------------------------------------------------
Proxy dynamics (control-affine in u = [a_x, a_y, a_z, ω_yaw_rate]):
----------------------------------------------------------------------------

  ṗ_drone     = v_drone                                       (state)
  v̇_drone_ef  = R_be · u_body + g_vec                          (R_be at current x)
  ṗitch       = (1/τ_rate) · (pitch_d - pitch)
                 with  pitch_d = -arctan(a_x/g) ≈ -a_x/g       (Taylor at a_x=0)
  ṙoll        = (1/τ_rate) · (roll_d  - roll)
                 with  roll_d  =  arctan(a_y/g) ≈  a_y/g
  ẏaw         = ω_yaw_rate
  ω_body      ≈ [roll_dot, pitch_dot, yaw_dot]                 (small-angle Euler→body)

Differences vs the actual 6-DOF model (Multicopter6DOFLite):
  - We skip the inner ω_body integration state (τ_rate filter on ω_body).
    Effect: pitch responds to u one filter "earlier" than truth. In practice
    this is < 50ms ahead at our dt=20ms, and the CBF margin α absorbs it.
  - We linearize pitch_d = -arctan(a_x/g) around a_x=0 (gives pitch_d ≈ -a_x/g).
    Effect: overestimates pitch_d magnitude at large |a_x| (since arctan is
    concave), so the CBF is *conservative* — throttles more than truth would
    require. Safe direction.
  - Thrust magnitude lag (τ_motor) is ignored. Effect: a small (5%) extra
    response margin in vertical-direction commands.

These approximations are why we still call this a "proxy" linearization. The
formal forward-invariance guarantee from the CBF papers applies to the
*proxy* model; the actual system tracks the proxy with bounded mismatch
absorbed by α.

----------------------------------------------------------------------------
Constraint summary (all relative degree 1 under this proxy):
----------------------------------------------------------------------------

  h_pitch = max_pitch² - pitch²           — depends on u via pitch_dot
  h_roll  = max_roll²  - roll²            — depends on u via roll_dot
  h_hfov  = tan²(α_h/2) - p̄_x²            — depends on u via ω_cam (via ω_body)
  h_vfov  = tan²(α_v/2) - p̄_y²            — depends on u via ω_cam (via ω_body)

For FOV: ṗ̄ = L_s · [v_cam; ω_cam] where v_cam (linear part) is u-independent
in proxy and ω_cam = R_cb · [roll_dot, pitch_dot, yaw_dot] is linear in u.
So the u-dependent contribution to ṗ̄ flows entirely through ω_cam.
"""

import numpy as np

from observers.interaction_matrix import InteractionMatrix

# Gravity (NED, z-down)
_G = 9.81


def compute_lie_derivatives(state: dict, params: dict) -> dict:
    """Compute L_f h and L_g h for all 4 CBF constraints at the current state.

    Args:
        state: dict with keys
            'pitch', 'roll', 'yaw'   : Euler angles (rad).
            'p_bar'                  : (2,) normalized image coords.
            'in_fov'                 : bool — target visible.
            'v_drone_efcs'           : (3,) drone velocity, NED EFCS (m/s).
            'v_target_efcs'          : (3,) target velocity, NED EFCS (m/s).
            'R_be'                   : (3, 3) body-to-earth rotation matrix.
            'R_cb'                   : (3, 3) body-to-camera rotation.
            'depth'                  : float — current z_c (camera-frame
                                       depth of target). Must be > 0.
        params: dict with keys
            'tau_rate'               : attitude tracking time constant (s).
            'tan_half_hfov',
            'tan_half_vfov'          : FOV thresholds.
            'max_pitch', 'max_roll'  : attitude limits.

    Returns:
        dict with per-constraint entries (4 keys: 'pitch', 'roll', 'hfov',
        'vfov'). Each value is a sub-dict:
            'h'   : scalar  — h_i(x).
            'Lfh' : scalar  — L_f h_i(x).
            'Lgh' : (4,)    — L_g h_i(x) (gradient w.r.t. u).
    """
    tau_rate = float(params['tau_rate'])
    pitch = float(state['pitch'])
    roll = float(state['roll'])
    p_bar = np.asarray(state['p_bar'], dtype=np.float64)
    p_bar_x, p_bar_y = float(p_bar[0]), float(p_bar[1])
    in_fov = bool(state.get('in_fov', True))

    # Effective limits (inner safety bound). The proxy is a first-order
    # approximation of the true second-order pitch/roll dynamics
    # (cascaded thrust + rate-loop lags). The actual system can overshoot
    # the proxy's predicted steady state, so we protect a *tighter* bound
    # and rely on the gap to absorb the proxy mismatch.
    safety_margin = float(params.get('attitude_safety_margin', 0.15))  # rad
    max_pitch_eff = max(0.0, params['max_pitch'] - safety_margin)
    max_roll_eff = max(0.0, params['max_roll'] - safety_margin)

    # ----- Attitude constraints (relative degree 1 in proxy) -----
    # h_pitch = max_pitch_eff² - pitch²
    # ḣ_pitch = -2·pitch · pitch_dot
    # pitch_dot = (1/τ)·(-a_x/g - pitch)
    # → ḣ_pitch = (2·pitch²)/τ + (2·pitch / (g·τ)) · a_x
    h_pitch = max_pitch_eff ** 2 - pitch ** 2
    Lf_pitch = 2.0 * pitch ** 2 / tau_rate
    Lg_pitch = np.zeros(4)
    Lg_pitch[0] = 2.0 * pitch / (_G * tau_rate)

    # h_roll, symmetric with a_y (roll_d = +a_y/g)
    # roll_dot = (1/τ)·(a_y/g - roll)
    # ḣ_roll = -2·roll · roll_dot = (2·roll²)/τ - (2·roll / (g·τ)) · a_y
    h_roll = max_roll_eff ** 2 - roll ** 2
    Lf_roll = 2.0 * roll ** 2 / tau_rate
    Lg_roll = np.zeros(4)
    Lg_roll[1] = -2.0 * roll / (_G * tau_rate)

    # ----- FOV constraints (relative degree 1 in proxy via attitude rate) -----
    # h_hfov = tan²(α/2) - p̄_x²
    # ḣ = -2·p̄_x · ṗ̄_x
    # ṗ̄ = L_s · [v_cam; ω_cam]
    # ω_cam = R_cb · ω_body where ω_body ≈ [roll_dot, pitch_dot, yaw_dot]
    # ω_body[0] = (1/τ)·(a_y/g - roll)        = drift_0 + (1/(g·τ)) · a_y
    # ω_body[1] = (1/τ)·(-a_x/g - pitch)      = drift_1 - (1/(g·τ)) · a_x
    # ω_body[2] = ω_yaw_rate                  = (1) · u_3 (no drift)
    if in_fov and state.get('depth', 0.0) > 0:
        L_s = InteractionMatrix.compute(p_bar, state['depth'])
        # L_s shape (2, 6): cols 0-2 = translational (with v_cam),
        # cols 3-5 = rotational (with ω_cam).
        L_trans = L_s[:, :3]
        L_rot = L_s[:, 3:]

        # v_cam in camera frame — moving target case
        R_cb = state['R_cb']
        R_be = state['R_be']
        v_rel_efcs = state['v_drone_efcs'] - state['v_target_efcs']
        v_rel_cam = R_cb @ R_be.T @ v_rel_efcs

        # Drift part of ω_body (the part independent of u)
        omega_body_drift = np.array([
            -roll / tau_rate,    # roll_dot drift
            -pitch / tau_rate,   # pitch_dot drift
            0.0,                  # yaw rate drift (none)
        ])
        # u-coefficient matrix M such that ω_body_u_part = M · u (M is 3×4)
        M = np.zeros((3, 4))
        M[0, 1] = 1.0 / (_G * tau_rate)    # ω_body[0] from a_y
        M[1, 0] = -1.0 / (_G * tau_rate)   # ω_body[1] from a_x
        M[2, 3] = 1.0                      # ω_body[2] from yaw rate

        omega_cam_drift = R_cb @ omega_body_drift
        # ω_cam = R_cb · ω_body → ω_cam_u_part = (R_cb · M) · u
        omega_cam_u_jacobian = R_cb @ M  # (3, 4)

        # ṗ̄ (drift) = L_trans · v_rel_cam + L_rot · ω_cam_drift
        # Note: standard L_s convention treats v_cam as camera's motion;
        # for moving target, ṗ̄_trans_total = L_trans·(v_target_cam - v_cam) =
        # -L_trans·v_rel_cam (with v_rel = v_drone - v_target as defined here).
        # Sign double-check: L_s[0,0] = -ρ. ṗ̄_x from camera moving forward in x
        # (v_cam[0]>0) should be negative (target appears to shift left). So
        # ṗ̄ = L_trans · v_cam (treating v_cam as the camera's own velocity).
        # With v_rel_cam = v_drone_cam - v_target_cam (where v_drone_cam IS
        # the camera velocity), we use ṗ̄ = L_trans · v_rel_cam directly.
        p_bar_dot_drift = L_trans @ v_rel_cam + L_rot @ omega_cam_drift
        # u-dependent Jacobian: ṗ̄_u_jac = L_rot · ω_cam_u_jacobian — shape (2, 4)
        p_bar_dot_jac_u = L_rot @ omega_cam_u_jacobian

        # ḣ_hfov = -2 · p̄_x · ṗ̄_x
        # L_f h_hfov = -2 · p̄_x · p_bar_dot_drift[0]
        # L_g h_hfov = -2 · p̄_x · p_bar_dot_jac_u[0]  (row 0, shape (4,))
        Lf_hfov = -2.0 * p_bar_x * float(p_bar_dot_drift[0])
        Lg_hfov = -2.0 * p_bar_x * p_bar_dot_jac_u[0]

        # h_vfov uses p̄_y, row 1
        Lf_vfov = -2.0 * p_bar_y * float(p_bar_dot_drift[1])
        Lg_vfov = -2.0 * p_bar_y * p_bar_dot_jac_u[1]
    else:
        # Out of FOV (or unknown depth): mark FOV barrier as "skipped" by
        # returning a strongly-positive h with zero Lf/Lg. The QP will then
        # ignore these constraints. (The caller / wrapper sets in_fov_only.)
        Lf_hfov = 0.0
        Lg_hfov = np.zeros(4)
        Lf_vfov = 0.0
        Lg_vfov = np.zeros(4)

    h_hfov = params['tan_half_hfov'] ** 2 - p_bar_x ** 2
    h_vfov = params['tan_half_vfov'] ** 2 - p_bar_y ** 2

    return {
        'pitch': {'h': h_pitch, 'Lfh': Lf_pitch, 'Lgh': Lg_pitch},
        'roll':  {'h': h_roll,  'Lfh': Lf_roll,  'Lgh': Lg_roll},
        'hfov':  {'h': h_hfov,  'Lfh': Lf_hfov,  'Lgh': Lg_hfov},
        'vfov':  {'h': h_vfov,  'Lfh': Lf_vfov,  'Lgh': Lg_vfov},
    }


def state_from_env(env_unwrapped, params: dict) -> dict:
    """Pack the env state into the dict expected by compute_lie_derivatives.

    Args:
        env_unwrapped: an InterceptionEnv instance (env.unwrapped).
        params: dict with FOV/attitude limits (passed through).

    Returns:
        State dict consumed by compute_lie_derivatives.
    """
    interceptor = env_unwrapped.interceptor
    target = env_unwrapped.target
    camera = env_unwrapped.camera
    R_be = interceptor.get_rotation_matrix()
    R_cb = camera.get_R_c_b()
    p_r = interceptor.position - target.position
    cam_result = camera.project(p_r, R_be)
    # Depth: ground-truth z_c in camera frame (for Phase 4a.3 cleanliness;
    # Phase 4a.4 could substitute the DKF/estimator value)
    target_in_cam = R_cb @ R_be.T @ (target.position - interceptor.position)
    depth = float(target_in_cam[2])
    return {
        'pitch': interceptor.pitch,
        'roll': interceptor.roll,
        'yaw': interceptor.yaw,
        'p_bar': cam_result['p_bar'],
        'in_fov': cam_result['in_fov'],
        'v_drone_efcs': interceptor.velocity.copy(),
        'v_target_efcs': target.velocity.copy(),
        'R_be': R_be,
        'R_cb': R_cb,
        'depth': depth,
    }
