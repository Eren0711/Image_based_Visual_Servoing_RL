"""
Six-DOF Target Drone — Realistic Equal-Agility Adversary
=========================================================
A target that flies the SAME 6-DOF multicopter dynamics as the interceptor
(``Multicopter6DOFLite``) with the SAME agility limits (v_max, a_max,
yaw_rate_max, omega_max, attitude limits). This removes the interceptor/
target asymmetry of the point-mass ``TargetDrone``: the 6-DOF target must
bank to turn, its turn rate is bounded by its attitude loop, and turning
couples to speed exactly as the interceptor's does.

Because the underlying model has an SO(3) geometric attitude controller that
converts a *body-frame acceleration command* into a coordinated banked turn,
the evasion guidance law only needs to emit a 4-D command
``[a_x, a_y, a_z, yaw_rate]`` (identical semantics to the interceptor's
action) and the realistic flight transients (bank entry/exit, speed bleed,
attitude lag) emerge automatically.

Drop-in compatible with ``InterceptionEnv`` (matches the ``TargetDrone`` API):
exposes ``position``, ``velocity``, ``acceleration``, and
``reset(position, velocity, maneuver_mode, seed)`` / ``step(pursuer_pos)``.

----------------------------------------------------------------------------
Evasion curriculum (simplest -> hardest), selected by ``maneuver_mode``:
  L1  'cruise'        : hold heading + cruise speed (straight line).
  L2  'steady_turn'   : constant banked turn (predictable curved path).
  L3  'weave'         : periodic alternating banked turns (S-curves / jink).
  L4  'break_turn'    : REACTIVE — bank away from the pursuer's line of sight.
  L5  'random_evasive': reactive break-turns with randomized flip timing
                        (unpredictable, the hardest level).
All levels hold cruise speed via a forward-acceleration term, and all lateral
commands are bounded by the shared ``a_max`` — so the target can never out-
accelerate the interceptor.
----------------------------------------------------------------------------
"""

import numpy as np

from models.multicopter_6dof import Multicopter6DOFLite

# Curriculum levels, ordered simplest -> hardest.
EVASION_LEVELS = ['cruise', 'steady_turn', 'weave', 'break_turn', 'random_evasive']


class SixDOFTarget:
    """6-DOF target with an evasion guidance law and equal agility limits."""

    MODES = EVASION_LEVELS

    def __init__(self, config: dict):
        """Build the 6-DOF target.

        Args:
            config: must contain the SAME interceptor agility fields used for
                symmetry: v_max, a_max, yaw_rate_max, dt, max_pitch_deg,
                max_roll_deg, and a 'dynamics_6dof' block. By construction we
                pass the interceptor's own config here so the target is an
                exactly equal-capability adversary. Optional evasion params:
                  cruise_frac        (default 0.8) cruise speed = frac*v_max
                  turn_accel_frac    (default 0.9) lateral accel = frac*a_max
                  weave_period       (default 2.0 s) period of L3 S-curves
                  jink_period        (default 1.5 s) mean L5 flip interval
                  react_dist         (default 30 m) range for reactive turns
        """
        self.dt = float(config['dt'])
        self.v_max = float(config['v_max'])
        self.a_max = float(config['a_max'])
        self.yaw_rate_max = float(config['yaw_rate_max'])

        # The flight model — identical class + limits as the interceptor.
        self.model = Multicopter6DOFLite(config)

        # Evasion guidance parameters
        self.cruise_frac = float(config.get('cruise_frac', 0.8))
        self.turn_accel_frac = float(config.get('turn_accel_frac', 0.9))
        self.weave_period = float(config.get('weave_period', 2.0))
        self.jink_period = float(config.get('jink_period', 1.5))
        self.react_dist = float(config.get('react_dist', 30.0))

        # Public state (mirrors TargetDrone)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.maneuver_mode = 'cruise'

        self._time = 0.0
        self._step_count = 0
        self._rng = np.random.default_rng()
        self._jink_sign = 1.0
        self._next_jink_step = 0
        self._prev_velocity = np.zeros(3)
        self._cruise_speed = self.cruise_frac * self.v_max

    # ------------------------------------------------------------------ #
    def reset(self, position: np.ndarray, velocity: np.ndarray,
              maneuver_mode: str, seed: int = None) -> None:
        """Reset to a position/velocity and an evasion level.

        Args mirror TargetDrone.reset for drop-in compatibility. The
        maneuver_mode must be one of EVASION_LEVELS; legacy point-mass mode
        names are mapped to the nearest 6-DOF level for convenience.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mode = self._map_mode(maneuver_mode)
        self.maneuver_mode = mode

        position = np.array(position, dtype=np.float64)
        velocity = np.array(velocity, dtype=np.float64)
        speed = float(np.linalg.norm(velocity))

        # Initialize the flight model at the given position, heading aligned
        # with the initial velocity (yaw from horizontal velocity components).
        if speed > 1e-6:
            yaw = float(np.arctan2(velocity[1], velocity[0]))
        else:
            yaw = float(self._rng.uniform(-np.pi, np.pi))
            velocity = speed * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        self.model.reset(position, yaw)
        # Seed the model's velocity so the target starts at cruise, not hover.
        self.model.velocity = velocity.copy()
        self.model._body_vel = self.model.get_rotation_matrix().T @ velocity

        # Cruise speed = the SPAWNED initial speed, so the 6-DOF target matches
        # whatever speed regime the env samples (the training distribution is
        # uniform(0, 0.5*v_max)). This keeps the evasion comparison from
        # secretly also being a faster-target comparison. Falls back to
        # cruise_frac*v_max only when spawned essentially stationary.
        self._cruise_speed = speed if speed > 0.5 else self.cruise_frac * self.v_max

        self.position = self.model.position.copy()
        self.velocity = self.model.velocity.copy()
        self.acceleration = np.zeros(3)
        self._prev_velocity = self.velocity.copy()
        self._time = 0.0
        self._step_count = 0
        self._jink_sign = 1.0 if self._rng.random() < 0.5 else -1.0
        self._schedule_next_jink()

    @staticmethod
    def _map_mode(mode: str) -> str:
        """Map legacy / alias mode names onto the 6-DOF curriculum levels."""
        if mode in EVASION_LEVELS:
            return mode
        alias = {
            'constant_velocity': 'cruise',
            'sinusoidal': 'weave',
            'circular': 'steady_turn',
            'random_aggressive': 'random_evasive',
            'evasive': 'break_turn',
        }
        return alias.get(mode, 'cruise')

    def _schedule_next_jink(self) -> None:
        period_steps = max(1, int(self.jink_period / self.dt))
        jitter = self._rng.uniform(0.5, 1.5)
        self._next_jink_step = self._step_count + max(1, int(period_steps * jitter))

    # ------------------------------------------------------------------ #
    def step(self, pursuer_pos: np.ndarray = None) -> None:
        """Advance the 6-DOF target by one timestep under its evasion law."""
        self._step_count += 1
        self._time += self.dt

        action = self._guidance(pursuer_pos)   # [a_x, a_y, a_z, yaw_rate], physical
        self.model.step(action)

        self._prev_velocity = self.velocity.copy()
        self.position = self.model.position.copy()
        self.velocity = self.model.velocity.copy()
        self.acceleration = (self.velocity - self._prev_velocity) / self.dt

    # ------------------------------------------------------------------ #
    def _guidance(self, pursuer_pos: np.ndarray = None) -> np.ndarray:
        """Evasion guidance law -> body-frame command for the 6-DOF model.

        Returns a 4-vector [a_x_body, a_y_body, a_z_body, yaw_rate] in PHYSICAL
        units (the Multicopter6DOFLite.step expects already-scaled commands).

        a_x : forward accel to hold cruise speed.
        a_y : lateral accel (the turn command) — set per evasion level.
        a_z : zero (stay in the horizontal plane; gravity handled by the model).
        yaw_rate: zero — turning is achieved by lateral accel (bank-to-turn),
                  exactly as the interceptor does, not by direct yaw.
        """
        # --- forward speed-hold (proportional, NOT deadbeat) ---
        # A 1/dt gain would saturate a_x at a_max on every step (commanding a
        # full-throttle nose-down indefinitely). Use a proportional gain so the
        # command decays to ~0 as the speed approaches cruise.
        R_be = self.model.get_rotation_matrix()
        v_body = R_be.T @ self.velocity
        cruise = self._cruise_speed   # set at reset to the spawned speed
        fwd_speed = float(v_body[0])
        k_speed = 1.0  # 1/s: gentle speed regulation
        a_x = np.clip(k_speed * (cruise - fwd_speed), -self.a_max, self.a_max)

        # --- lateral turn command per level ---
        a_turn = self.turn_accel_frac * self.a_max
        mode = self.maneuver_mode
        if mode == 'cruise':
            a_y = 0.0
        elif mode == 'steady_turn':
            a_y = a_turn  # constant bank, one direction
        elif mode == 'weave':
            # periodic S-curve: alternate lateral direction sinusoidally
            a_y = a_turn * np.sign(np.sin(2 * np.pi * self._time / self.weave_period))
        elif mode == 'break_turn':
            a_y = a_turn * self._reactive_sign(pursuer_pos)
        elif mode == 'random_evasive':
            if self._step_count >= self._next_jink_step:
                self._jink_sign *= -1.0
                self._schedule_next_jink()
            # reactive when pursuer is close, else randomized jink
            sgn = self._reactive_sign(pursuer_pos, fallback=self._jink_sign)
            a_y = a_turn * sgn
        else:
            a_y = 0.0

        # Combined horizontal command magnitude must respect a_max.
        cmd = np.array([a_x, a_y, 0.0])
        n = np.linalg.norm(cmd)
        if n > self.a_max:
            cmd = cmd * (self.a_max / n)

        # --- Coordinating yaw-rate (turn coordination) ---
        # A pure body-lateral acceleration with zero yaw makes the bank angle
        # accumulate past 90° (the body keeps rolling as it turns). A real
        # coordinated turn yaws the body to follow the velocity heading at
        # rate omega = a_lat / v. Commanding this keeps the body aligned with
        # the flight path and the bank settles at a stable angle — exactly how
        # the interceptor policy flies (it commands yaw alongside lateral accel).
        speed = max(self.get_speed(), 1e-3)
        yaw_rate = np.clip(cmd[1] / speed, -self.yaw_rate_max, self.yaw_rate_max)
        return np.array([cmd[0], cmd[1], 0.0, yaw_rate])

    def _reactive_sign(self, pursuer_pos, fallback: float = 1.0) -> float:
        """Choose the lateral sign that banks the velocity AWAY from the
        pursuer line-of-sight. Falls back to `fallback` when no pursuer info
        or the pursuer is beyond react_dist."""
        if pursuer_pos is None:
            return fallback
        los = np.asarray(pursuer_pos) - self.position
        dist = float(np.linalg.norm(los))
        if dist < 1e-6 or dist > self.react_dist:
            return fallback
        v = self.velocity
        if np.linalg.norm(v) < 1e-6:
            return fallback
        v_hat = v / np.linalg.norm(v)
        lat = np.array([-v_hat[1], v_hat[0], 0.0])  # body-left in horizontal plane
        los_hat = los / dist
        # Turn so velocity rotates away from the LOS: pick sign opposing the
        # lateral projection of the LOS.
        s = -np.sign(np.dot(lat, los_hat))
        return float(s) if s != 0 else fallback

    # ------------------------------------------------------------------ #
    # Convenience getters mirroring the flight model (for diagnostics)
    def get_bank_angle(self) -> float:
        return float(self.model.roll)

    def get_speed(self) -> float:
        return float(np.linalg.norm(self.velocity))
