"""
Wind Disturbance Model — Stage 4b
==================================
Generates a stochastic wind velocity (NED EFCS) and converts it to a drag-
coupled force on the interceptor. The wind itself is modeled as an
Ornstein-Uhlenbeck process: a stationary, exponentially-correlated random
process that captures the low-frequency variation of atmospheric wind
without the implementation cost of full Dryden turbulence.

Continuous-time OU process (per component):
    dV_wind = -θ · (V_wind − V_mean) · dt  +  σ · dW

where:
    V_mean  : mean wind vector (typically (0,0,0) for "no prevailing wind")
    θ       : mean-reversion rate (1/s); higher = wind returns to mean faster
    σ       : noise intensity (m/s · s^{−1/2})
    dW      : standard Wiener process increment

Euler-Maruyama discretization with timestep dt:
    V_wind[k+1] = V_wind[k] + (-θ · (V_wind[k] − V_mean)) · dt + σ · √dt · N(0,I_3)

Stationary distribution: V_wind ~ N(V_mean, σ² / (2θ) · I_3)
So the RMS gust magnitude is σ / √(2θ) per component.

The wind couples to the drone through a simple linear drag model:
    F_drag = -k_drag · m · (V_drone − V_wind)         (NED EFCS, in Newtons)
    a_drag = -k_drag · (V_drone − V_wind)              (m/s²)

This is added to the drone's EFCS acceleration each step. With k_drag ≈ 0.1/s,
a 5 m/s wind produces ~0.5 m/s² acceleration on a hovering drone — visible
but not catastrophic. Choose k_drag based on the size/drag of the platform.

What's NOT modeled here:
  - Wind gradient with altitude
  - Wind variation across the drone's wingspan (we treat the drone as a point)
  - Coupling to attitude (the drag should also produce a torque; ignored)
  - Dryden turbulence spectrum (we use OU which has a different power spectrum)

These approximations are appropriate for our point-mass + simplified-attitude
6-DOF model and let us study "does the policy still work with wind" without
committing to a high-fidelity atmosphere.
"""

import numpy as np


class WindModel:
    """Stochastic wind disturbance via OU process + linear drag."""

    def __init__(
        self,
        dt: float,
        sigma: float = 1.0,
        theta: float = 0.5,
        v_mean: np.ndarray = None,
        k_drag: float = 0.1,
        seed: int = None,
    ):
        """
        Args:
            dt:      Simulation timestep (s).
            sigma:   OU noise intensity. RMS gust magnitude per component is
                     σ / √(2θ). With σ=1.0, θ=0.5: ~1 m/s RMS.
            theta:   OU mean-reversion rate (1/s). Sets the wind's
                     correlation time τ_corr ≈ 1/θ. θ=0.5 → 2 s correlation.
            v_mean:  Mean wind vector in NED EFCS (3,). Default: (0,0,0).
            k_drag:  Linear drag coefficient (1/s). a_drag = -k_drag · (v_drone − v_wind).
            seed:    RNG seed for reproducibility. None = nondeterministic.
        """
        self.dt = float(dt)
        self.sigma = float(sigma)
        self.theta = float(theta)
        self.k_drag = float(k_drag)
        self.v_mean = (np.zeros(3) if v_mean is None
                       else np.asarray(v_mean, dtype=np.float64).copy())
        self.v_wind = self.v_mean.copy()
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int = None) -> None:
        """Re-seed and re-initialize wind to its mean.

        Args:
            seed: New RNG seed. None keeps the current generator state.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Initialize from stationary distribution rather than fixed mean to
        # avoid a transient at the start of each episode.
        stationary_std = self.sigma / np.sqrt(2.0 * self.theta)
        self.v_wind = self.v_mean + stationary_std * self._rng.standard_normal(3)

    def step(self) -> np.ndarray:
        """Advance the wind state by one timestep and return v_wind (EFCS).

        Returns:
            v_wind: (3,) wind velocity in NED EFCS at the new time (m/s).
        """
        drift = -self.theta * (self.v_wind - self.v_mean) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * self._rng.standard_normal(3)
        self.v_wind += drift + diffusion
        return self.v_wind.copy()

    def get_drag_acceleration(self, v_drone_efcs: np.ndarray) -> np.ndarray:
        """Compute the drag-coupled acceleration on the drone.

        a_drag = -k_drag · (v_drone − v_wind)

        Args:
            v_drone_efcs: (3,) drone velocity in NED EFCS (m/s).

        Returns:
            a_drag: (3,) drag acceleration to add to the drone's EFCS
                acceleration this step (m/s²).
        """
        return -self.k_drag * (np.asarray(v_drone_efcs, dtype=np.float64) - self.v_wind)
