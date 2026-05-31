"""
Intermittent Detection Wrapper — Stage 4b
==========================================
Simulates a real-world object detector by stochastically dropping
detections based on distance and off-axis angle, and (optionally) adding
distance-dependent measurement noise on top of any homoscedastic noise
added by a downstream NoiseDelayWrapper.

This replaces the "perfect detector + Gaussian noise" model used through
Stage 3 with something closer to a YOLO-style detector's behavior:
  - Detection rate degrades with distance (target gets smaller in pixels)
  - Detection rate degrades near the FOV edge (cropped bounding box)
  - Measurement noise grows with distance (smaller pixels → larger relative error)

Stack position: sits between the bare env and NoiseDelayWrapper, so the
NoiseDelay's delay buffer sees the post-miss `in_fov` flag. The DKF then
naturally handles missed detections as predict-only steps (existing
behavior — no DKF change needed).

  env → WindWrapper → IntermittentDetectionWrapper → NoiseDelay → DKF → CBF → agent

Detection model:

  p_detect(d, m_fov) = sigmoid(β_1 / d  −  β_2 · (1 − m_fov)  +  β_3)

where:
  d        : ground-truth relative distance (m)
  m_fov    : fov_margin ∈ [0, 1], proxy for "how centered" the target is
             (already provided by the camera; 1 = centered, 0 = on the edge)
  β_1, β_2, β_3 : tunable; defaults give p_detect ≈ 0.95 at d=10m centered,
                  dropping to ~0.5 at d=30m centered, ~0.3 at d=30m edge.

Distance-dependent extra noise (added to p_bar):
  σ_dist(d) = σ_base + σ_slope · d

with default σ_base=0, σ_slope=0.0005 (negligible at close range, ~0.015
extra at d=30m).

Implementation note: we modify obs[4] (the in_fov flag) but NOT obs[0:2]
when a detection is missed. Why? The NoiseDelayWrapper downstream uses
the in_fov flag to gate downstream behavior and the DKF will predict-only
when in_fov=False, so leaving the (stale) p_bar in obs[0:2] is harmless.
This also means: detections are missed VISIBLY (the flag drops) rather
than silently (with bad numbers in obs[0:2]).
"""

import math
import numpy as np
import gymnasium as gym


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class IntermittentDetectionWrapper(gym.Wrapper):
    """Stochastically drop detections and add distance-scaled noise.

    Args:
        env:        Gymnasium env (typically just-out-of-WindWrapper).
        beta_1:     Distance term in p_detect logit (m). Default 8.0.
                    Higher = more robust at long range.
        beta_2:     FOV-edge penalty in p_detect logit. Default 4.0.
                    Higher = more drop-out near FOV edge.
        beta_3:     Bias in p_detect logit. Default 1.0.
                    Sets the overall detection rate.
        sigma_base: Distance-independent extra noise on p_bar.
                    Default 0 (rely on downstream NoiseDelay for the
                    homoscedastic part).
        sigma_slope: Linear growth of extra noise per metre.
                     Default 0.0005 (≈0.015 at d=30m).
        seed:       Optional RNG seed.
        randomize_per_episode:
            If True, jitter β_1, β_2, β_3 each reset for domain
            randomization (default: off; on for training).
        randomization_ranges:
            Dict of override ranges. Default: β_1 ∈ [6, 12], β_2 ∈ [2, 6],
            β_3 ∈ [0, 2].
        verbose:    If True, info['det'] gets the full p_detect each step.
    """

    _DEFAULT_RANGES = {
        'beta_1': (6.0, 12.0),
        'beta_2': (2.0, 6.0),
        'beta_3': (0.0, 2.0),
    }

    def __init__(
        self,
        env: gym.Env,
        beta_1: float = 8.0,
        beta_2: float = 4.0,
        beta_3: float = 1.0,
        sigma_base: float = 0.0,
        sigma_slope: float = 0.0005,
        seed: int = None,
        randomize_per_episode: bool = False,
        randomization_ranges: dict = None,
        verbose: bool = False,
    ):
        super().__init__(env)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.beta_3 = float(beta_3)
        self.sigma_base = float(sigma_base)
        self.sigma_slope = float(sigma_slope)
        self.randomize_per_episode = bool(randomize_per_episode)
        self.randomization_ranges = {
            **self._DEFAULT_RANGES,
            **(randomization_ranges or {}),
        }
        self._rng = np.random.default_rng(seed)
        self.verbose = bool(verbose)

        # Episode counters (reset each episode)
        self._n_steps = 0
        self._n_dropped = 0
        self._n_targets_in_fov = 0  # geometric in_fov, before drop

        # Curriculum DR (Intervention B). frac 0→1 interpolates the β bands
        # from an *easy* band (high detection rate) to the configured
        # full-hard band. Easy β₁ high (robust at range), β₃ high (high
        # baseline detection) → few drops; hard band allows heavy dropout.
        self._curriculum = False
        self._curriculum_frac = 1.0
        self._easy_ranges = None

    def enable_curriculum(self, easy_ranges: dict = None) -> None:
        """Turn on curriculum annealing of the detection-difficulty band."""
        self._curriculum = True
        default_easy = {
            'beta_1': (10.0, 14.0),   # strong distance term → robust at range
            'beta_2': self.randomization_ranges['beta_2'],
            'beta_3': (1.5, 2.0),     # high bias → high baseline detection
        }
        self._easy_ranges = {**default_easy, **(easy_ranges or {})}

    def set_curriculum_frac(self, frac: float) -> None:
        self._curriculum_frac = float(np.clip(frac, 0.0, 1.0))

    def __getattr__(self, name):
        """Forward attribute access to the wrapped env (compat with
        downstream wrappers that expect env.dt, env.np_random, etc.)."""
        if name.startswith('_') or name in ('env', 'observation_space',
                                              'action_space', 'spec'):
            raise AttributeError(name)
        return getattr(self.env, name)

    def _current_band(self, key: str) -> tuple:
        hard = self.randomization_ranges[key]
        if not self._curriculum:
            return hard
        easy = self._easy_ranges[key]
        f = self._curriculum_frac
        return (easy[0] + f * (hard[0] - easy[0]),
                easy[1] + f * (hard[1] - easy[1]))

    def _p_detect(self, distance: float, fov_margin: float) -> float:
        """Compute detection probability."""
        logit = (self.beta_1 / max(distance, 1e-3)
                 - self.beta_2 * (1.0 - fov_margin)
                 + self.beta_3)
        # Clip to avoid overflow in extreme cases
        logit = max(-30.0, min(30.0, logit))
        return _sigmoid(logit)

    def _maybe_randomize(self) -> None:
        if not self.randomize_per_episode:
            return
        self.beta_1 = float(self._rng.uniform(*self._current_band('beta_1')))
        self.beta_2 = float(self._rng.uniform(*self._current_band('beta_2')))
        self.beta_3 = float(self._rng.uniform(*self._current_band('beta_3')))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._maybe_randomize()
        self._n_steps = 0
        self._n_dropped = 0
        self._n_targets_in_fov = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._n_steps += 1

        # Pull current state from info (provided by InterceptionEnv)
        distance = float(info.get('relative_distance', 1.0))
        fov_margin = float(info.get('fov_margin', 0.0))
        geom_in_fov = bool(info.get('in_fov', False))

        det_valid = False
        p = 0.0
        if geom_in_fov:
            self._n_targets_in_fov += 1
            p = self._p_detect(distance, fov_margin)
            det_valid = bool(self._rng.random() < p)
            if not det_valid:
                self._n_dropped += 1

        # If detection invalid, drop the in_fov flag in the observation.
        # This is what the downstream NoiseDelay buffers and the DKF gates on.
        if not det_valid:
            obs = obs.copy()
            obs[4] = 0.0
        else:
            # Optionally add distance-scaled noise to the p_bar measurement.
            # (Homoscedastic noise is still handled by downstream NoiseDelay.)
            if self.sigma_base > 0 or self.sigma_slope > 0:
                extra_sigma = self.sigma_base + self.sigma_slope * distance
                obs = obs.copy()
                obs[0:2] = obs[0:2] + self._rng.normal(
                    0.0, extra_sigma, size=2
                ).astype(obs.dtype)

        info = dict(info) if info is not None else {}
        info['det'] = {
            'valid': det_valid,
            'p_detect': p,
            'geom_in_fov': geom_in_fov,
            'distance': distance,
            'fov_margin': fov_margin,
        }
        if terminated or truncated:
            info['det_episode'] = {
                'n_steps': self._n_steps,
                'n_targets_in_fov': self._n_targets_in_fov,
                'n_dropped': self._n_dropped,
                'drop_rate': (self._n_dropped / self._n_targets_in_fov
                              if self._n_targets_in_fov > 0 else 0.0),
                'beta_1': self.beta_1,
                'beta_2': self.beta_2,
                'beta_3': self.beta_3,
            }
        return obs, reward, terminated, truncated, info
