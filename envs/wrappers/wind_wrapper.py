"""
Wind Disturbance Wrapper — Stage 4b
====================================
Gymnasium wrapper that applies a stochastic wind disturbance to the
interceptor's velocity each step, via the WindModel (OU process + linear
drag). This perturbs the dynamics, not the observations — the policy
discovers the wind only through its effect on subsequent state.

Stack position: outermost dynamics modifier, just outside the bare env:

    env → WindWrapper → IntermittentDetection → NoiseDelay → DKF → CBF → agent

The wrapper does its work between env.step() and returning the obs. It:
  1. Lets the env step normally (the policy commands an action that the
     interceptor executes under its own dynamics).
  2. Advances the wind state by one timestep.
  3. Computes the drag acceleration on the post-step velocity.
  4. Adds (drag acceleration · dt) directly to the interceptor's EFCS
     velocity. We perturb velocity rather than re-stepping the dynamics
     because the multicopter is a non-affine system and re-running step()
     would double-count motor/attitude lags.
  5. Recomputes the body-frame velocity cache (used by the reward's near-
     target brake term).

Notes on the velocity-perturbation approach:
  - Position is NOT corrected for the perturbation within this step. The
    error is O(dt²) — at dt=0.02s with realistic drag (~0.5 m/s² peaks),
    the per-step position error is ~10⁻⁴ m. Negligible.
  - This is the same approximation used by most multi-rotor wind models
    in research code (Gazebo, AirSim) at this fidelity level.

Wind state is exposed via info['wind'] for diagnostics / domain
randomization tracking.
"""

import numpy as np
import gymnasium as gym

from models.wind_model import WindModel


class WindWrapper(gym.Wrapper):
    """Apply OU-process wind disturbance to the interceptor's velocity.

    Args:
        env:        A Gymnasium env (or wrapped env). The unwrapped env
                    must expose `interceptor.velocity` and
                    `interceptor._body_vel` (Multicopter6DOFLite API).
        sigma:      OU noise intensity (m/s · s^{−1/2}).
        theta:      OU mean-reversion rate (1/s).
        v_mean:     Mean wind vector NED (3,). Default zero.
        k_drag:     Linear drag coefficient (1/s).
        seed:       Optional RNG seed.
        randomize_per_episode:
            If True, sample (sigma, theta, k_drag) from a band on each
            reset — used for domain randomization in Stage 4b training.
            Sampled ranges: sigma ∈ [0.5, 2.0], theta ∈ [0.3, 1.0],
            k_drag ∈ [0.05, 0.2]. Override via `randomization_ranges`.
        randomization_ranges:
            Dict overriding the default ranges, e.g.
            {'sigma': (0.5, 2.0), 'theta': (0.3, 1.0), 'k_drag': (0.05, 0.2)}.
    """

    _DEFAULT_RANGES = {
        'sigma': (0.5, 2.0),
        'theta': (0.3, 1.0),
        'k_drag': (0.05, 0.2),
    }

    def __init__(
        self,
        env: gym.Env,
        sigma: float = 1.0,
        theta: float = 0.5,
        v_mean: np.ndarray = None,
        k_drag: float = 0.1,
        seed: int = None,
        randomize_per_episode: bool = False,
        randomization_ranges: dict = None,
    ):
        super().__init__(env)
        base = self.env.unwrapped
        dt = float(base.dt)
        self.sigma = float(sigma)
        self.theta = float(theta)
        self.k_drag = float(k_drag)
        self.v_mean = (np.zeros(3) if v_mean is None
                       else np.asarray(v_mean, dtype=np.float64))
        self.randomize_per_episode = bool(randomize_per_episode)
        self.randomization_ranges = {
            **self._DEFAULT_RANGES,
            **(randomization_ranges or {}),
        }
        self._rand_rng = np.random.default_rng(seed)
        self.wind_model = WindModel(
            dt=dt, sigma=self.sigma, theta=self.theta,
            v_mean=self.v_mean, k_drag=self.k_drag, seed=seed,
        )
        self._base = base  # cached reference to unwrapped env
        self._dt = dt

        # Curriculum DR (Intervention B): when enabled, the per-episode
        # sampling band is linearly interpolated from an *easy* band (low
        # wind) to the configured full-hard band as `_curriculum_frac` goes
        # 0 → 1. A CurriculumCallback advances the frac during training.
        self._curriculum = False
        self._curriculum_frac = 1.0  # default = full-hard (no curriculum)
        self._easy_ranges = None

    def enable_curriculum(self, easy_ranges: dict = None) -> None:
        """Turn on curriculum annealing of the DR band.

        Args:
            easy_ranges: the band at frac=0 (episode start of training).
                Defaults to a light-wind band. The frac=1 band is the
                wrapper's configured `randomization_ranges` (full-hard).
        """
        self._curriculum = True
        # Easy = near-calm wind. theta/k_drag bands kept (they're benign).
        default_easy = {
            'sigma': (0.0, 0.5),
            'theta': self.randomization_ranges['theta'],
            'k_drag': self.randomization_ranges['k_drag'],
        }
        self._easy_ranges = {**default_easy, **(easy_ranges or {})}

    def set_curriculum_frac(self, frac: float) -> None:
        self._curriculum_frac = float(np.clip(frac, 0.0, 1.0))

    def __getattr__(self, name):
        """Forward attribute access to the wrapped env (for compatibility
        with downstream wrappers like NoiseDelayWrapper that expect to read
        env attributes directly)."""
        if name.startswith('_') or name in ('env', 'observation_space',
                                              'action_space', 'spec'):
            raise AttributeError(name)
        return getattr(self.env, name)

    def _current_band(self, key: str) -> tuple:
        """Return the (lo, hi) sampling band for `key`, applying curriculum
        interpolation between easy and full-hard if enabled."""
        hard = self.randomization_ranges[key]
        if not self._curriculum:
            return hard
        easy = self._easy_ranges[key]
        f = self._curriculum_frac
        lo = easy[0] + f * (hard[0] - easy[0])
        hi = easy[1] + f * (hard[1] - easy[1])
        return (lo, hi)

    def _maybe_randomize(self) -> None:
        if not self.randomize_per_episode:
            return
        new_sigma = float(self._rand_rng.uniform(*self._current_band('sigma')))
        new_theta = float(self._rand_rng.uniform(*self._current_band('theta')))
        new_kdrag = float(self._rand_rng.uniform(*self._current_band('k_drag')))
        self.wind_model.sigma = new_sigma
        self.wind_model.theta = new_theta
        self.wind_model.k_drag = new_kdrag

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._maybe_randomize()
        # Re-seed wind from kwargs if provided, else keep internal RNG state.
        seed = kwargs.get('seed', None)
        self.wind_model.reset(seed=seed)
        info = dict(info) if info is not None else {}
        info['wind'] = {
            'v_wind': self.wind_model.v_wind.tolist(),
            'sigma': self.wind_model.sigma,
            'theta': self.wind_model.theta,
            'k_drag': self.wind_model.k_drag,
        }
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Advance wind one step
        v_wind = self.wind_model.step()
        # Compute drag acceleration on the post-step velocity
        a_drag = self.wind_model.get_drag_acceleration(self._base.interceptor.velocity)
        # Apply as a velocity perturbation: Δv = a_drag · dt
        delta_v = a_drag * self._dt
        self._base.interceptor.velocity = self._base.interceptor.velocity + delta_v
        # Clamp to v_max (same safety clamp as the dynamics step)
        v_max = float(self._base.interceptor.v_max)
        speed = float(np.linalg.norm(self._base.interceptor.velocity))
        if speed > v_max:
            self._base.interceptor.velocity *= (v_max / speed)
        # Recompute body-frame velocity cache (used by reward's brake term
        # and the DKF wrapper's IMU feedforward computation)
        R_be = self._base.interceptor.get_rotation_matrix()
        self._base.interceptor._body_vel = R_be.T @ self._base.interceptor.velocity

        info = dict(info) if info is not None else {}
        info['wind'] = {
            'v_wind': v_wind.tolist(),
            'a_drag': a_drag.tolist(),
        }
        return obs, reward, terminated, truncated, info
