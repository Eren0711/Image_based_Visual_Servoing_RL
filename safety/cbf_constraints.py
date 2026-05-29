"""
CBF Constraint Definitions — Stage 4a
======================================
Defines the four control barrier functions h_i(x) > 0 from the implementation
plan (per arXiv:2404.08296 §III-D and Ames et al. 2019):

  1. Horizontal FOV preservation: target stays inside camera's horizontal FOV
  2. Vertical FOV preservation:    target stays inside camera's vertical FOV
  3. Pitch limit:                  |pitch| < max_pitch
  4. Roll limit:                   |roll| < max_roll

All barriers use the **squared** form h = (limit)² − coord², which is smooth
through the boundary and has nonzero gradient there (the regularity condition
from Ames et al. Theorem 2 / Remark 5).

Note on FOV constraints: the camera's `in_fov` test uses arctan thresholds,
but we constrain the normalized image coords |p_bar_{x,y}| ≤ tan(α/2)
instead — these are equivalent at the boundary (arctan and tan are inverse)
and the tan form is smooth (no arctan/division-by-z singularity inside h).

State extraction is decoupled from h-evaluation: each function takes a state
dict (the schema used by `Multicopter6DOFLite.get_state()` augmented with
`p_bar` and target info, populated by the wrapper). This keeps the QP code
free of env knowledge.
"""

import numpy as np


def h_hfov(state: dict, tan_half_hfov: float, in_fov_only: bool = False) -> float:
    """Horizontal FOV barrier: h > 0 when target's image x-coord is inside FOV.

    h = tan(α_h/2)² − p_bar_x²

    Args:
        state: must contain 'p_bar' (2-vector, normalized image coords).
        tan_half_hfov: tan(α_hfov / 2), the horizontal half-FOV in tan space.
        in_fov_only: if True, return -1.0 when target is out of FOV (we
            don't try to recover via CBF once already lost; this avoids
            spurious negative-h states confusing the QP).

    Returns:
        h value (positive = safe, zero = on boundary, negative = violation).
    """
    if in_fov_only and not state.get('in_fov', True):
        return -1.0
    p_bar_x = float(state['p_bar'][0])
    return tan_half_hfov ** 2 - p_bar_x ** 2


def h_vfov(state: dict, tan_half_vfov: float, in_fov_only: bool = False) -> float:
    """Vertical FOV barrier: h > 0 when target's image y-coord is inside FOV.

    h = tan(α_v/2)² − p_bar_y²
    """
    if in_fov_only and not state.get('in_fov', True):
        return -1.0
    p_bar_y = float(state['p_bar'][1])
    return tan_half_vfov ** 2 - p_bar_y ** 2


def h_pitch(state: dict, max_pitch: float) -> float:
    """Pitch barrier: h > 0 when |pitch| < max_pitch.

    h = max_pitch² − pitch²
    """
    pitch = float(state['pitch'])
    return max_pitch ** 2 - pitch ** 2


def h_roll(state: dict, max_roll: float) -> float:
    """Roll barrier: h > 0 when |roll| < max_roll.

    h = max_roll² − roll²
    """
    roll = float(state['roll'])
    return max_roll ** 2 - roll ** 2


# ---------------------------------------------------------------------- #
# Aggregator: evaluate all four barriers from a single state dict        #
# ---------------------------------------------------------------------- #

def evaluate_all(state: dict, params: dict, in_fov_only: bool = False) -> np.ndarray:
    """Evaluate all 4 CBFs at a given state.

    Args:
        state: state dict with 'p_bar', 'in_fov', 'pitch', 'roll'.
        params: dict with 'tan_half_hfov', 'tan_half_vfov', 'max_pitch',
            'max_roll'.
        in_fov_only: see h_hfov/h_vfov.

    Returns:
        np.ndarray (4,) of [h_hfov, h_vfov, h_pitch, h_roll] values.
    """
    return np.array([
        h_hfov(state, params['tan_half_hfov'], in_fov_only),
        h_vfov(state, params['tan_half_vfov'], in_fov_only),
        h_pitch(state, params['max_pitch']),
        h_roll(state, params['max_roll']),
    ], dtype=np.float64)
