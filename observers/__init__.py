"""
Observers Module — State Estimation for IBVS
=============================================
Provides image Jacobian (interaction matrix) computation and
depth estimation tools for image-based visual servoing.

Components:
    - interaction_matrix: IBVS interaction matrix L_s and its decomposition
    - depth_estimator:    Recursive Jacobian-based inverse-depth estimator
"""

from observers.interaction_matrix import InteractionMatrix
from observers.depth_estimator import DepthEstimator
