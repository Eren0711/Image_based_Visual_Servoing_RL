"""
Target Drone Point-Mass Model — Stage 1
========================================
Point-mass target model with selectable maneuver modes.
Implements Eq. (3) of the paper:
    ṗ_t = v_t
    v̇_t = a_t

Coordinate convention: NED (z-down)

Maneuver modes:
    - constant_velocity  : Straight line, zero acceleration
    - sinusoidal         : Lateral sinusoidal evasion
    - circular           : Circular orbit in a horizontal plane
    - random_aggressive  : Bounded random acceleration, resampled periodically

Reference: Section II-A.3 of arXiv:2404.08296
"""

import numpy as np


class TargetDrone:
    """Point-mass target drone with multiple maneuver modes.

    State:
        position     : np.ndarray (3,) — position in EFCS [x_e, y_e, z_e]
        velocity     : np.ndarray (3,) — velocity in EFCS [vx_e, vy_e, vz_e]
        acceleration : np.ndarray (3,) — current acceleration in EFCS
    """

    # Supported maneuver modes
    MODES = ['constant_velocity', 'sinusoidal', 'circular', 'random_aggressive',
             'evasive']

    def __init__(self, config: dict):
        """Initialize target drone model.

        Args:
            config: Target configuration plus interceptor timing values.
                Required: v_max, a_max, dt
                Optional: sin_amplitude, sin_frequency, circle_radius,
                          circle_omega, random_switch_interval
        """
        self.v_max = config.get('v_max', 10.0)
        self.a_max = config.get('a_max', 5.0)
        self.dt = config['dt']

        # Sinusoidal evasion parameters
        self.sin_amplitude = config.get('sin_amplitude', 3.0)
        self.sin_frequency = config.get('sin_frequency', 1.5)

        # Circular motion parameters
        self.circle_radius = config.get('circle_radius', 15.0)
        self.circle_omega = config.get('circle_omega', 0.5)

        # Random aggressive parameters
        self.random_switch_interval = config.get('random_switch_interval', 25)

        # Evasive parameters (reactive, turn-rate-limited high-g evasion).
        #   evasive_g          : peak lateral acceleration in g (default 2g).
        #   evasive_turn_rate  : max heading-change rate (rad/s), models the
        #                        physical turn-rate limit of a real vehicle so
        #                        the target cannot reverse instantaneously.
        #   evasive_jink_period: mean interval (s) between evasive direction
        #                        flips ("jinking"), randomized per flip.
        #   evasive_react_dist : pursuer range (m) within which evasion
        #                        becomes reactive (turn away from LOS).
        self.evasive_g = config.get('evasive_g', 2.0)
        self.evasive_turn_rate = config.get('evasive_turn_rate', 2.5)
        self.evasive_jink_period = config.get('evasive_jink_period', 1.0)
        self.evasive_react_dist = config.get('evasive_react_dist', 25.0)
        # Cruise speed the evader maintains while breaking (fraction of v_max).
        # Without an explicit speed-hold the perpendicular-only turn decays the
        # speed to near-zero, making the "evader" trivially easy to catch.
        self.evasive_cruise_frac = config.get('evasive_cruise_frac', 0.8)

        # State (initialized in reset)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.maneuver_mode = 'constant_velocity'
        self._time = 0.0
        self._step_count = 0
        self._rng = np.random.default_rng()

        # Internal state for specific modes
        self._sin_phase = np.zeros(3)       # Phase offsets for sinusoidal
        self._sin_axis = np.zeros((3, 3))   # Evasion axes
        self._circle_center = np.zeros(3)   # Center of circular orbit
        self._circle_normal = np.array([0.0, 0.0, 1.0])  # Orbit plane normal
        self._random_accel = np.zeros(3)    # Current random acceleration
        self._jink_sign = 1.0               # current evasive turn direction
        self._next_jink_step = 0            # step index of next jink flip

    def reset(self, position: np.ndarray, velocity: np.ndarray,
              maneuver_mode: str, seed: int = None) -> None:
        """Reset the target to initial conditions.

        Args:
            position: Initial position in EFCS [x_e, y_e, z_e].
            velocity: Initial velocity in EFCS [vx_e, vy_e, vz_e].
            maneuver_mode: One of 'constant_velocity', 'sinusoidal',
                           'circular', 'random_aggressive'.
            seed: Optional RNG seed for reproducibility.
        """
        if maneuver_mode not in self.MODES:
            raise ValueError(
                f"Unknown maneuver mode '{maneuver_mode}'. "
                f"Must be one of {self.MODES}"
            )

        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.acceleration = np.zeros(3)
        self.maneuver_mode = maneuver_mode
        self._time = 0.0
        self._step_count = 0

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Mode-specific initialization
        if maneuver_mode == 'sinusoidal':
            self._init_sinusoidal()
        elif maneuver_mode == 'circular':
            self._init_circular()
        elif maneuver_mode == 'random_aggressive':
            self._init_random_aggressive()
        elif maneuver_mode == 'evasive':
            self._init_evasive()

    def _init_sinusoidal(self) -> None:
        """Initialize sinusoidal evasion parameters.

        The target performs lateral sinusoidal evasion perpendicular to its
        initial velocity direction. Random phase offsets create varied evasion
        patterns.
        """
        # Random phase offsets for each axis
        self._sin_phase = self._rng.uniform(0, 2 * np.pi, size=3)

        # Build an evasion coordinate frame perpendicular to initial velocity
        v_norm = np.linalg.norm(self.velocity)
        if v_norm > 1e-6:
            forward = self.velocity / v_norm
        else:
            forward = np.array([1.0, 0.0, 0.0])

        # Find two vectors perpendicular to forward direction
        if abs(forward[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([1.0, 0.0, 0.0])
        lateral1 = np.cross(forward, up)
        lateral1 /= np.linalg.norm(lateral1)
        lateral2 = np.cross(forward, lateral1)
        lateral2 /= np.linalg.norm(lateral2)

        self._sin_axis = np.array([forward, lateral1, lateral2])

    def _init_circular(self) -> None:
        """Initialize circular orbit parameters.

        The target orbits around a center point in the horizontal plane (NED:
        x-y plane). The orbit center is computed from the initial position
        and the specified radius.
        """
        # Orbit in the x-y (horizontal) plane
        self._circle_normal = np.array([0.0, 0.0, 1.0])  # z-down normal

        # Compute initial angular position and velocity direction
        # Place center such that target starts on the circle
        speed = np.linalg.norm(self.velocity[:2])
        if speed < 1e-6:
            # Default: moving in +x direction
            self.velocity = np.array([
                self.circle_radius * self.circle_omega, 0.0, 0.0
            ])

        # Velocity direction in x-y plane
        v_dir = self.velocity[:2] / np.linalg.norm(self.velocity[:2])
        # Center is perpendicular to velocity (to the left in standard math)
        center_offset = np.array([-v_dir[1], v_dir[0]]) * self.circle_radius
        self._circle_center = self.position.copy()
        self._circle_center[:2] += center_offset

    def _init_random_aggressive(self) -> None:
        """Initialize random aggressive maneuver parameters."""
        self._resample_random_acceleration()

    def _resample_random_acceleration(self) -> None:
        """Sample a new random acceleration vector within bounds."""
        # Random direction, random magnitude up to a_max
        direction = self._rng.standard_normal(3)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0, 0.0])
        magnitude = self._rng.uniform(0, self.a_max)
        self._random_accel = direction * magnitude

    def step(self, pursuer_pos: np.ndarray = None) -> None:
        """Advance the target by one timestep.

        Computes the acceleration based on the current maneuver mode,
        then integrates velocity and position via Euler method.

        Args:
            pursuer_pos: Optional (3,) interceptor position in EFCS. Only the
                'evasive' mode uses it (to turn away from the pursuer's
                line-of-sight). All other modes ignore it, so existing
                callers that invoke step() with no argument are unaffected.
        """
        self._step_count += 1
        self._time += self.dt

        # Compute acceleration for current mode
        self.acceleration = self._compute_acceleration(pursuer_pos)

        # Euler integration: v_t ← v_t + a_t * dt
        self.velocity = self.velocity + self.acceleration * self.dt

        # Clip velocity to maximum speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.v_max:
            self.velocity = self.velocity * (self.v_max / speed)

        # Euler integration: p_t ← p_t + v_t * dt
        self.position = self.position + self.velocity * self.dt

    def _compute_acceleration(self, pursuer_pos: np.ndarray = None) -> np.ndarray:
        """Compute the target's acceleration based on the maneuver mode.

        Args:
            pursuer_pos: Optional (3,) pursuer position (evasive mode only).

        Returns:
            np.ndarray (3,) — acceleration in EFCS.
        """
        if self.maneuver_mode == 'constant_velocity':
            return np.zeros(3)

        elif self.maneuver_mode == 'sinusoidal':
            return self._accel_sinusoidal()

        elif self.maneuver_mode == 'circular':
            return self._accel_circular()

        elif self.maneuver_mode == 'random_aggressive':
            return self._accel_random_aggressive()

        elif self.maneuver_mode == 'evasive':
            return self._accel_evasive(pursuer_pos)

        else:
            return np.zeros(3)

    def _accel_sinusoidal(self) -> np.ndarray:
        """Sinusoidal lateral evasion acceleration.

        The acceleration is applied perpendicular to the initial velocity
        direction with sinusoidal time variation:
            a = A * sin(ω*t + φ)  along lateral axes
        """
        t = self._time
        omega = self.sin_frequency
        A = self.sin_amplitude

        # Evasion acceleration in the lateral directions (axes 1 and 2)
        a_lat1 = A * np.sin(omega * t + self._sin_phase[1])
        a_lat2 = A * 0.5 * np.sin(2 * omega * t + self._sin_phase[2])

        accel = (a_lat1 * self._sin_axis[1] + a_lat2 * self._sin_axis[2])

        # Clip to max acceleration
        a_norm = np.linalg.norm(accel)
        if a_norm > self.a_max:
            accel = accel * (self.a_max / a_norm)

        return accel

    def _accel_circular(self) -> np.ndarray:
        """Circular orbit centripetal acceleration.

        The target maintains a circular orbit around _circle_center in the
        horizontal plane. The centripetal acceleration is:
            a = -ω² * (p - center)   projected to the orbit plane
        """
        # Vector from center to current position (in horizontal plane)
        r_vec = self.position - self._circle_center
        r_vec[2] = 0  # Project to horizontal plane

        r_dist = np.linalg.norm(r_vec)
        if r_dist < 1e-6:
            return np.zeros(3)

        # Centripetal acceleration toward center
        omega = self.circle_omega
        accel = -(omega ** 2) * r_vec

        # Also add a tangential correction to maintain the target speed
        # on the circle (= R * omega)
        desired_speed = self.circle_radius * omega
        r_hat = r_vec / r_dist
        # Tangential direction (perpendicular to r_hat in horizontal plane)
        tangent = np.array([-r_hat[1], r_hat[0], 0.0])
        current_tangent_speed = np.dot(self.velocity[:2],
                                       tangent[:2])
        speed_error = desired_speed - current_tangent_speed
        accel += 2.0 * speed_error * tangent

        # Also correct radius drift
        radius_error = r_dist - self.circle_radius
        accel -= 1.0 * radius_error * r_hat

        # Clip total acceleration
        a_norm = np.linalg.norm(accel)
        if a_norm > self.a_max:
            accel = accel * (self.a_max / a_norm)

        return accel

    def _accel_random_aggressive(self) -> np.ndarray:
        """Random bounded acceleration, resampled periodically.

        Every `random_switch_interval` steps, a new random acceleration
        vector is sampled within the bounds [-a_max, a_max].
        """
        if self._step_count % self.random_switch_interval == 0:
            self._resample_random_acceleration()
        return self._random_accel.copy()

    def _init_evasive(self) -> None:
        """Initialize the reactive, turn-rate-limited evasive maneuver."""
        self._jink_sign = 1.0 if self._rng.random() < 0.5 else -1.0
        self._schedule_next_jink()

    def _schedule_next_jink(self) -> None:
        """Pick the step index of the next evasive direction flip.

        Jink intervals are randomized around `evasive_jink_period` so the
        timing is unpredictable to the pursuer (uniform in [0.5, 1.5]×period).
        """
        period_steps = max(1, int(self.evasive_jink_period / self.dt))
        jitter = self._rng.uniform(0.5, 1.5)
        self._next_jink_step = self._step_count + max(1, int(period_steps * jitter))

    def _accel_evasive(self, pursuer_pos: np.ndarray = None) -> np.ndarray:
        """Reactive, physically-constrained high-g evasion.

        Design goals:
          * High lateral g (default 2g) — genuinely aggressive.
          * Turn-rate-limited — acceleration is applied PERPENDICULAR to the
            current velocity (a coordinated turn), so heading changes at a
            bounded rate rather than the velocity vector teleporting. The
            magnitude is additionally capped so the per-step heading change
            does not exceed `evasive_turn_rate * dt`.
          * Reactive — when the pursuer is within `evasive_react_dist`, the
            turn direction is chosen to rotate the velocity AWAY from the
            line-of-sight to the pursuer (break-turn), which is what real
            evasion does. Beyond that range (or with no pursuer info), it
            falls back to unpredictable periodic "jinking".

        This stays a point-mass kinematic model (no attitude/rotor dynamics),
        but unlike `random_aggressive` it cannot reverse direction
        instantaneously and it actively responds to the pursuer.
        """
        v = self.velocity
        speed = np.linalg.norm(v)
        if speed < 1e-6:
            # Degenerate (hovering) — give a small kick to establish heading.
            return self.evasive_g * 9.81 * np.array([1.0, 0.0, 0.0])
        v_hat = v / speed

        # Lateral (turn) direction in the horizontal plane: perpendicular to v.
        lat = np.array([-v_hat[1], v_hat[0], 0.0])
        lat_norm = np.linalg.norm(lat)
        if lat_norm < 1e-6:
            # Velocity is near-vertical; pick an arbitrary horizontal normal.
            lat = np.array([1.0, 0.0, 0.0])
        else:
            lat = lat / lat_norm

        # --- Decide turn direction ---
        if pursuer_pos is not None:
            los = pursuer_pos - self.position
            dist = np.linalg.norm(los)
            if dist < self.evasive_react_dist and dist > 1e-6:
                # Break-turn: choose the lateral sign that turns velocity AWAY
                # from the pursuer (i.e. away from the LOS direction).
                los_hat = los / dist
                # Sign that makes the turn increase the v–LOS angle:
                self._jink_sign = -np.sign(np.dot(lat, los_hat)) or 1.0
            else:
                if self._step_count >= self._next_jink_step:
                    self._jink_sign *= -1.0
                    self._schedule_next_jink()
        else:
            if self._step_count >= self._next_jink_step:
                self._jink_sign *= -1.0
                self._schedule_next_jink()

        # --- Peak lateral acceleration (g), then turn-rate cap ---
        a_peak = self.evasive_g * 9.81
        # Centripetal a = v * dpsi/dt; cap a so dpsi/dt <= evasive_turn_rate.
        a_turn_cap = speed * self.evasive_turn_rate
        a_mag = min(a_peak, a_turn_cap)
        accel_turn = self._jink_sign * a_mag * lat

        # --- Speed-hold (cruise) term ALONG velocity ---
        # A pure perpendicular acceleration causes the integrated speed to
        # decay (the velocity vector curls and is re-clipped each step), so
        # without this the "evader" spirals to a near-stop and becomes trivial
        # to intercept. A real evader maintains cruise speed while breaking.
        # Drive speed toward a high cruise fraction of v_max.
        cruise = self.evasive_cruise_frac * self.v_max
        speed_err = cruise - speed
        accel_cruise = np.clip(speed_err / max(self.dt, 1e-3),
                               -a_peak, a_peak) * v_hat

        accel = accel_turn + accel_cruise

        # Respect the configured a_max ceiling as a final clamp (the turn term
        # is prioritized: if clipping is needed, preserve lateral authority).
        a_norm = np.linalg.norm(accel)
        if a_norm > self.a_max:
            accel = accel * (self.a_max / a_norm)
        return accel

    def get_state(self) -> dict:
        """Return the full target state.

        Returns:
            dict with keys:
                'position'     : np.ndarray (3,) — EFCS position
                'velocity'     : np.ndarray (3,) — EFCS velocity
                'acceleration' : np.ndarray (3,) — current EFCS acceleration
        """
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy(),
        }
