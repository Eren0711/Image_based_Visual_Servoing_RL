"""
Lead-Pursuit Observation Wrapper
================================
A pure *guidance transformation* applied to the observation before it reaches
the policy. No retraining, no reward/dynamics/target change.

Pure pursuit (the trained policy) drives the CURRENT image error
(p_bar_x, p_bar_y) to zero. Against a turning target the target has already
moved by the time the interceptor arrives, so reacting to where the target *is*
is always late. Lead pursuit instead drives the PREDICTED future image error to
zero:

    p_bar_lead = p_bar_current + p_bar_dot_current * T

where ``p_bar_dot`` is the image-plane velocity already estimated by the DKF and
exposed in the observation, and ``T`` is the lead/prediction horizon (seconds).

UNIT CORRECTNESS (critical)
---------------------------
The two image channels are normalized DIFFERENTLY in this project's observation
(see envs/interception_env.py::_build_observation and
envs/wrappers/dkf_wrapper.py::observation):

    obs[0:2] = p_bar / tan(half_fov)          # position, per-axis tan scaling
    obs[2:4] = (d p_bar / dt) / MAX_DP         # velocity, scaled by MAX_DP=10.0

So the naive ``obs[0:2] + obs[2:4]*T`` is dimensionally WRONG (it is off by a
factor of MAX_DP/tan(half_fov) ~ 14-19x and would barely lead at all). The
correct transform converts both channels to raw p_bar units, applies the lead,
and converts back:

    raw_pos = obs[0:2] * tan_half          # tan_half = [tan_h, tan_v]
    raw_vel = obs[2:4] * MAX_DP            # raw p_bar / s
    raw_lead = raw_pos + raw_vel * T
    obs_lead[0:2] = clip(raw_lead / tan_half, -1, 1)

equivalently  obs_lead[i] = clip(obs[i] + obs[i+2] * (MAX_DP / tan_half[i]) * T, -1, 1).

Only obs[0:2] (the image error the actor servos on) is modified; obs[2:4]
(velocity) and every other component are left untouched. T = 0 is the identity,
so it exactly reproduces the pure-pursuit baseline.

NOTE: the velocity used is the DKF *filtered* estimate (obs[2:4] is overwritten
by DKFWrapper), not the raw noisy measurement. MAX_DP=10.0 mirrors the constant
hardcoded in the env/DKF normalization; if that constant changes, update here.
"""

import numpy as np
import gymnasium as gym

# Must match the velocity normalization constant in
# envs/interception_env.py::_build_observation and dkf_wrapper.py::observation.
MAX_DP = 10.0


class LeadPursuitWrapper(gym.ObservationWrapper):
    """Replace obs[0:2] (current image error) with the lead-corrected error."""

    # observation indices (16-D base layout; HardNet appends context at >=16)
    IDX_PX, IDX_PY = 0, 1          # normalized image error
    IDX_VX, IDX_VY = 2, 3          # normalized image-plane velocity (DKF estimate)

    def __init__(self, env, lead_time: float):
        super().__init__(env)
        self.lead_time = float(lead_time)

        # Per-axis tan(half_fov) from the underlying camera.
        fov = env.unwrapped.camera.get_fov_params()
        self.tan_h = float(fov["tan_half_hfov"])
        self.tan_v = float(fov["tan_half_vfov"])

        # Conversion factors so that velocity (scaled by MAX_DP) and position
        # (scaled by tan_half) combine in consistent units.
        self.kx = MAX_DP / self.tan_h
        self.ky = MAX_DP / self.tan_v

    def observation(self, obs):
        if self.lead_time == 0.0:
            return obs  # identity -> exact pure-pursuit baseline
        obs = np.array(obs, dtype=np.float32, copy=True)
        obs[self.IDX_PX] = np.clip(
            obs[self.IDX_PX] + obs[self.IDX_VX] * self.kx * self.lead_time, -1.0, 1.0)
        obs[self.IDX_PY] = np.clip(
            obs[self.IDX_PY] + obs[self.IDX_VY] * self.ky * self.lead_time, -1.0, 1.0)
        return obs
