"""
IBVS Drone Interception Environment — Stage 3a
================================================
Gymnasium environment for RL-based image-guided drone interception.

Stage 2b → 3a changes:
  - Drone dynamics: first-order velocity lag (τ=0.2s) replaces instant response
  - Attitude coupling: pitch/roll derived from acceleration, tilts the camera
  - Reward: added attitude stability penalty to discourage excessive pitch/roll
  - Angular velocity for depth estimator: now uses actual pitch/roll rates
  - Observation: pitch/roll normalized by max_pitch/max_roll (better resolution)
  - Fresh training required (dynamics fundamentally changed)

The agent must now balance:
  (a) Aggressive pursuit → large acceleration → large pitch → target drifts in FOV
  (b) Gentle flight → small pitch → stable FOV → slow closure

Coordinate convention: NED (z-down)

Reference: arXiv:2404.08296
  "High-Speed Interception Multicopter Control by Image-based Visual Servoing"
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from models.drone_dynamics import InterceptorDrone
from models.target_model import TargetDrone
from models.camera_model import PinholeCamera
from observers.interaction_matrix import InteractionMatrix
from observers.depth_estimator import DepthEstimator


class InterceptionEnv(gym.Env):
    """Gymnasium environment for image-based visual servoing interception.

    Stage 3a: First-order inertia dynamics with attitude-camera coupling.

    Observation space (16-dim continuous):
        [0:2]   p̄_x, p̄_y          — normalized image-plane error
        [2:4]   dp̄_x/dt, dp̄_y/dt  — image-plane velocity (finite diff)
        [4]     in_fov              — target visible (1) or lost (0)
        [5]     ẑ_c (normalized)    — Jacobian-based depth ESTIMATE [0, 1]
        [6:9]   v_body              — interceptor velocity in body frame (3D)
        [9:12]  roll, pitch, yaw    — interceptor Euler angles (normalized)
        [12:16] prev_action         — previous action (4D)

    Stage 3a vs 2b:
        - pitch and roll are now NON-ZERO (coupled to acceleration)
        - obs[9:10] normalized by max_pitch/max_roll (better resolution)
        - Camera tilts with the drone, changing where target appears in image

    Action space (4-dim continuous, [-1, 1]):
        [0:3]   a_cmd       — body-frame acceleration commands (scaled by a_max)
        [3]     yaw_rate    — yaw rate command (scaled by yaw_rate_max)

    Reward: uses PRIVILEGED ground-truth distance for approach reward.
            Attitude stability penalty added (Stage 3a).

    Episode termination:
        - Success: relative distance < d_success
        - FOV loss: target lost for > fov_loss_limit consecutive steps
        - Timeout: step count >= max_steps (truncation)
    """

    metadata = {'render_modes': [], 'render_fps': 50}

    def __init__(self, config: dict = None):
        """Initialize the interception environment.

        Args:
            config: Full configuration dictionary (from config.yaml).
                    If None, loads default config from config.yaml.
        """
        super().__init__()

        # Load configuration
        if config is None:
            config = self._load_default_config()
        self.config = config

        # Extract sub-configs
        cam_cfg = config['camera']
        int_cfg = config['interceptor']
        tgt_cfg = config['target']
        env_cfg = config['env']
        rwd_cfg = config['reward']

        # Merge dt into target config for integration
        tgt_cfg_full = {**tgt_cfg, 'dt': int_cfg['dt']}

        # --- Create sub-models ---
        self.interceptor = InterceptorDrone(int_cfg)
        self.target = TargetDrone(tgt_cfg_full)
        self.camera = PinholeCamera(cam_cfg)

        # --- Depth estimator (Stage 2a: replaces ground-truth distance) ---
        depth_cfg = config.get('depth_estimator', {})
        self.depth_estimator = DepthEstimator(
            rho_init=depth_cfg.get('rho_init', 0.05),       # ~20m initial guess
            P_init=depth_cfg.get('P_init', 1.0),
            Q=depth_cfg.get('Q', 0.001),
            R_base=depth_cfg.get('R_base', 0.1),
            rho_min=depth_cfg.get('rho_min', 0.005),         # max 200m
            rho_max=depth_cfg.get('rho_max', 2.0),           # min 0.5m
            confidence_threshold=depth_cfg.get('confidence_threshold', 0.01),
        )

        # --- Environment parameters ---
        self.dt = int_cfg['dt']
        self.v_max = int_cfg['v_max']
        self.a_max = int_cfg['a_max']
        self.yaw_rate_max = int_cfg['yaw_rate_max']
        self.max_steps = env_cfg['max_steps']
        self.d_success = env_cfg['d_success']
        self.d_image_cutoff = env_cfg.get('d_image_cutoff', 25.0)
        self.fov_loss_limit = env_cfg['fov_loss_limit']
        self.init_dist_range = env_cfg['init_distance_range']
        self.norm_dist_max = env_cfg['norm_distance_max']
        self.norm_vel_max = env_cfg['norm_velocity_max']
        self.target_modes = tgt_cfg.get(
            'maneuver_modes',
            ['constant_velocity', 'sinusoidal']
        )
        # If maneuver_modes is missing or not a list, handle gracefully
        if not isinstance(self.target_modes, list):
            self.target_modes = ['constant_velocity', 'sinusoidal']

        # --- Reward parameters ---
        self.w_image    = rwd_cfg['w_image']
        self.w_approach = rwd_cfg['w_approach']
        self.w_fov_loss = rwd_cfg['w_fov_loss']
        self.w_boundary = rwd_cfg['w_boundary']
        self.w_effort   = rwd_cfg['w_effort']
        self.w_jerk     = rwd_cfg.get('w_jerk', -0.1)
        self.w_attitude = rwd_cfg.get('w_attitude', -0.05)
        self.w_dist_penalty = rwd_cfg.get('w_dist_penalty', 0.0)  # Stage 3a-v2
        self.w_near_brake = rwd_cfg.get('w_near_brake', 0.0)      # Stage 3a-noisy polish
        self.d_brake = rwd_cfg.get('d_brake', 5.0)
        self.w_intercept = rwd_cfg['w_intercept']
        self.w_timeout  = rwd_cfg['w_timeout']
        self.k1_image   = rwd_cfg['k1_image']

        # Stage 3a: attitude normalization constants (for obs and reward)
        self.max_pitch = self.interceptor.max_pitch  # rad
        self.max_roll  = self.interceptor.max_roll   # rad

        # --- Spaces ---
        # Observation: 16-dimensional (Stage 2a: vision-only + depth estimate)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )
        # Action: 4-dimensional, [-1, 1] (scaled internally)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # --- Internal state ---
        self._step_count = 0
        self._fov_loss_counter = 0
        self._prev_distance = 0.0
        self._prev_p_bar = np.zeros(2)
        self._current_p_bar = np.zeros(2)
        self._in_fov = True
        self._episode_outcome = 'running'
        self._prev_action = np.zeros(4)
        self._current_depth_est = 20.0     # Current depth estimate
        self._current_yaw_rate = 0.0       # Yaw rate applied this step
        self._prev_pitch = 0.0             # Stage 3a: for angular rate estimation
        self._prev_roll = 0.0

    @staticmethod
    def _load_default_config() -> dict:
        """Load the default configuration from config.yaml.

        Returns:
            dict: parsed YAML configuration.
        """
        import yaml
        import os
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config.yaml'
        )
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def reset(self, seed=None, options=None):
        """Reset the environment to a new episode.

        Randomizes:
          - Interceptor position at origin with random heading
          - Target position within [init_dist_min, init_dist_max] range,
            ensuring the target is initially within the camera FOV
          - Target maneuver mode (random selection)
          - Target initial velocity (random direction and magnitude)

        Args:
            seed: Optional integer seed for reproducibility.
            options: Optional dict (unused).

        Returns:
            observation: np.ndarray (16,) — initial observation.
            info: dict — initial info.
        """
        super().reset(seed=seed)

        # --- Reset counters ---
        self._step_count = 0
        self._fov_loss_counter = 0
        self._episode_outcome = 'running'
        self._prev_action = np.zeros(4)
        self._current_yaw_rate = 0.0
        self._prev_pitch = 0.0
        self._prev_roll  = 0.0

        # --- Reset interceptor ---
        interceptor_pos = np.zeros(3)
        interceptor_yaw = self.np_random.uniform(-np.pi, np.pi)
        self.interceptor.reset(interceptor_pos, interceptor_yaw)

        # --- Place target within FOV ---
        target_pos, target_vel = self._sample_target_initial_state(
            interceptor_pos, interceptor_yaw
        )

        # --- Select maneuver mode ---
        mode = self.np_random.choice(self.target_modes)

        # --- Reset target ---
        target_seed = int(self.np_random.integers(0, 2**31))
        self.target.reset(target_pos, target_vel, mode, seed=target_seed)

        # --- Compute initial camera projection ---
        p_r = self.interceptor.position - self.target.position
        R_be = self.interceptor.get_rotation_matrix()
        cam_result = self.camera.project(p_r, R_be)

        self._prev_p_bar = cam_result['p_bar'].copy()
        self._current_p_bar = cam_result['p_bar'].copy()
        self._in_fov = cam_result['in_fov']
        self._prev_distance = np.linalg.norm(p_r)

        # --- Reset depth estimator ---
        # Initial guess: use the ground-truth depth for initialization
        # (the agent doesn't see this — it's just to give the estimator
        # a reasonable starting point rather than always guessing 20m)
        init_depth = cam_result.get('depth', 20.0)
        if init_depth > 0.5:
            rho_init = 1.0 / init_depth
        else:
            rho_init = 0.05
        self.depth_estimator.reset(rho_init=rho_init, P_init=0.5)
        self._current_depth_est = init_depth

        # --- Build initial observation ---
        obs = self._build_observation(cam_result)
        info = self._build_info(cam_result)

        return obs, info

    def _sample_target_initial_state(
        self, interceptor_pos: np.ndarray, interceptor_yaw: float
    ) -> tuple:
        """Sample a target initial position and velocity that's within the FOV.

        The target is placed in front of the interceptor (along its heading)
        with some randomization, ensuring it projects within the camera FOV.

        Args:
            interceptor_pos: Interceptor position in EFCS.
            interceptor_yaw: Interceptor heading (rad).

        Returns:
            (target_pos, target_vel): Tuple of position and velocity arrays.
        """
        R_be = self.interceptor.get_rotation_matrix()
        max_attempts = 100

        for _ in range(max_attempts):
            dist = self.np_random.uniform(
                self.init_dist_range[0], self.init_dist_range[1]
            )

            body_dir = np.array([
                self.np_random.uniform(0.5, 1.0),        # mostly forward
                self.np_random.uniform(-0.3, 0.3),        # slight lateral
                self.np_random.uniform(-0.3, 0.3),        # slight vertical
            ])
            body_dir /= np.linalg.norm(body_dir)

            offset_efcs = R_be @ body_dir * dist
            target_pos = interceptor_pos + offset_efcs

            p_r = interceptor_pos - target_pos
            cam_result = self.camera.project(p_r, R_be)
            if cam_result['in_fov'] and cam_result['fov_margin'] > 0.2:
                break

        speed = self.np_random.uniform(0.0, self.target.v_max * 0.5)
        vel_dir = self.np_random.standard_normal(3)
        vel_norm = np.linalg.norm(vel_dir)
        if vel_norm > 1e-6:
            vel_dir /= vel_norm
        else:
            vel_dir = np.array([1.0, 0.0, 0.0])
        target_vel = vel_dir * speed

        return target_pos, target_vel

    def step(self, action):
        """Execute one environment step.

        Args:
            action: np.ndarray (4,) in [-1, 1]. Scaled internally:
                    action[0:3] * a_max → body acceleration commands
                    action[3] * yaw_rate_max → yaw rate command

        Returns:
            observation: np.ndarray (16,)
            reward: float
            terminated: bool (success or FOV loss)
            truncated: bool (timeout)
            info: dict
        """
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        # --- Scale action to physical units ---
        scaled_action = np.array([
            action[0] * self.a_max,
            action[1] * self.a_max,
            action[2] * self.a_max,
            action[3] * self.yaw_rate_max,
        ])
        self._current_yaw_rate = scaled_action[3]

        # --- Step interceptor ---
        self.interceptor.step(scaled_action)

        # --- Step target ---
        self.target.step()

        # --- Camera projection ---
        p_r = self.interceptor.position - self.target.position
        R_be = self.interceptor.get_rotation_matrix()
        cam_result = self.camera.project(p_r, R_be)

        # --- Update internal state ---
        self._step_count += 1
        self._prev_p_bar = self._current_p_bar.copy()
        self._current_p_bar = cam_result['p_bar'].copy()
        self._in_fov = cam_result['in_fov']

        current_distance = np.linalg.norm(p_r)

        # --- Stage 3a: store previous attitude for rate estimation ---
        prev_pitch = self.interceptor.pitch
        prev_roll  = self.interceptor.roll

        # --- Depth estimation via Jacobian ---
        self._update_depth_estimate(cam_result, R_be)

        # --- Update prev attitude after depth estimation ---
        self._prev_pitch = prev_pitch
        self._prev_roll  = prev_roll

        # --- FOV loss tracking ---
        if not self._in_fov:
            self._fov_loss_counter += 1
        else:
            self._fov_loss_counter = 0

        # --- Compute reward (uses PRIVILEGED ground-truth distance) ---
        reward = self._compute_reward(
            cam_result, current_distance, action
        )

        # --- Check termination ---
        terminated = False
        truncated = False

        if current_distance < self.d_success:
            terminated = True
            self._episode_outcome = 'success'
            reward += self.w_intercept

        elif self._fov_loss_counter >= self.fov_loss_limit:
            terminated = True
            self._episode_outcome = 'fov_loss'

        elif self._step_count >= self.max_steps:
            truncated = True
            self._episode_outcome = 'timeout'

        # --- Update previous state ---
        self._prev_distance = current_distance
        self._prev_action = action.copy()

        # --- Build observation and info ---
        obs = self._build_observation(cam_result)
        info = self._build_info(cam_result)

        return obs, reward, terminated, truncated, info

    def _update_depth_estimate(self, cam_result: dict,
                                R_be: np.ndarray) -> None:
        """Update the Jacobian-based depth estimate.

        Stage 3a change: angular velocity now includes pitch and roll rates
        (not just yaw) because the drone actually tilts with acceleration.
        This gives the depth estimator more accurate camera motion data.

        Args:
            cam_result: Camera projection results dict.
            R_be: Body-to-earth rotation matrix.
        """
        if not self._in_fov:
            return

        # Image velocity via finite differences
        p_bar     = self._current_p_bar
        p_bar_dot = (self._current_p_bar - self._prev_p_bar) / self.dt

        # Camera angular velocity in body frame
        # Stage 3a: include pitch and roll rates (from attitude coupling)
        pitch_rate = (self.interceptor.pitch - self._prev_pitch) / self.dt
        roll_rate  = (self.interceptor.roll  - self._prev_roll)  / self.dt
        # Body frame: [roll_rate, pitch_rate, yaw_rate] = [p, q, r]
        omega_body = np.array([roll_rate, pitch_rate, self._current_yaw_rate])

        v_cam, omega_cam = InteractionMatrix.compute_camera_velocity(
            v_interceptor_efcs=self.interceptor.velocity,
            omega_body=omega_body,
            R_b_e=R_be,
            R_c_b=self.camera.R_c_b,
        )

        est_result = self.depth_estimator.update(
            p_bar=p_bar,
            p_bar_dot=p_bar_dot,
            v_cam=v_cam,
            omega_cam=omega_cam,
            v_target_cam=None,
        )
        self._current_depth_est = est_result['z_hat']

    def _compute_reward(self, cam_result: dict, current_distance: float,
                        action: np.ndarray) -> float:
        """Compute the per-step reward.

        Stage 2a reward philosophy:
          Priority 1: Keep target in FOV (survival)
          Priority 2: Center target on image (tracking quality)
          Priority 3: Approach target (mission success)

        IMPORTANT: The approach reward uses PRIVILEGED ground-truth distance.
        This is standard RL practice — the reward is the "teacher" and is
        only used during training. The agent's policy only sees the observation.

        Args:
            cam_result: Camera projection results dict.
            current_distance: Current ||p_r|| (PRIVILEGED — not in obs).
            action: Raw action ([-1, 1] range).

        Returns:
            float: total reward.
        """
        reward = 0.0

        # --- 1. Image centering reward (distance-gated) ---
        # The raw form exp(-k * ||p̄||^2) pays the same at any distance as long
        # as the target is centered. That made loiter-at-distance the optimal
        # strategy under noisy perception (breakeven analysis: agent would need
        # 66% commit success to prefer closure over loitering at 35m).
        # The distance gate makes image reward decay linearly to zero at
        # d_image_cutoff, so the agent can only harvest image reward by getting
        # close. Tracking is still incentivized during legitimate approach
        # (factor > 0 for d < d_image_cutoff) but not while loitering at range.
        p_bar = cam_result['p_bar']
        image_error_sq = p_bar[0] ** 2 + p_bar[1] ** 2
        if self._in_fov:
            raw_image = np.exp(-self.k1_image * image_error_sq)
            distance_factor = max(
                0.0, 1.0 - current_distance / self.d_image_cutoff
            )
            r_image = raw_image * distance_factor
        else:
            r_image = 0.0
        reward += self.w_image * r_image

        # --- 2. Approach reward (PRIVILEGED — uses ground-truth distance) ---
        delta_dist = current_distance - self._prev_distance
        reward += self.w_approach * (-delta_dist)

        # --- 2b. Per-step distance penalty (Stage 3a-v2) ---
        # Without this, the optimal policy under v1 weights was "fly past the
        # target and orbit at constant distance" — image tracking paid forever
        # while delta_dist averaged to zero. The penalty makes loitering at
        # range strictly costly, so closure becomes mandatory for positive return.
        reward += self.w_dist_penalty * (current_distance / self.norm_dist_max)

        # --- 2c. Near-target braking penalty (Stage 3a-noisy polish) ---
        # 79% of FOV-loss failures occurred at d=2.5-5m, where the agent over-
        # committed terminally — high closure velocity caused aggressive pitch,
        # which slipped the target out of frame at the moment of interception.
        # This term pays nothing outside d_brake but progressively penalizes
        # body speed as the agent enters the terminal zone, encouraging it to
        # bleed velocity before the final approach.
        if current_distance < self.d_brake:
            proximity_speed = float(
                np.linalg.norm(self.interceptor.get_body_velocity())
            )
            proximity_factor = (self.d_brake - current_distance) / self.d_brake
            reward += self.w_near_brake * proximity_speed * proximity_factor

        # --- 3. FOV loss penalty ---
        if not self._in_fov:
            reward += self.w_fov_loss

        # --- 4. FOV boundary penalty ---
        if self._in_fov:
            fov_margin = cam_result['fov_margin']
            reward += self.w_boundary * max(0.0, 1.0 - fov_margin) ** 2

        # --- 5. Control effort penalty ---
        reward += self.w_effort * float(np.sum(action ** 2))

        # --- 6. Jerk penalty ---
        reward += self.w_jerk * float(np.sum((action - self._prev_action) ** 2))

        # --- 7. Attitude stability penalty (Stage 3a) ---
        # Penalize excessive pitch and roll — they degrade visual tracking
        # Normalized by max allowed angle → penalty ∈ [0, 1] each
        pitch_norm = abs(self.interceptor.pitch) / max(self.max_pitch, 1e-6)
        roll_norm  = abs(self.interceptor.roll)  / max(self.max_roll,  1e-6)
        r_attitude = pitch_norm ** 2 + roll_norm ** 2
        reward += self.w_attitude * r_attitude

        # --- 8. Time penalty ---
        reward += self.w_timeout

        return reward

    def _build_observation(self, cam_result: dict) -> np.ndarray:
        """Build the 16-dimensional observation vector.

        Stage 2a observation — vision-only with depth estimate:
            [0:2]   Image-plane error (normalized by FOV)
            [2:4]   Image-plane velocity
            [4]     Target visibility flag
            [5]     Depth ESTIMATE (Jacobian-based, normalized)
            [6:9]   Body velocity (ego-state, from IMU)
            [9:12]  Euler angles (ego-state, from IMU)
            [12:16] Previous action

        REMOVED from Stage 1b:
            - Ground-truth relative distance
            - LOS unit vector
            - Relative velocity

        Args:
            cam_result: Camera projection results dict.

        Returns:
            np.ndarray (16,) — observation vector.
        """
        obs = np.zeros(16, dtype=np.float32)

        # [0:2] Normalized image-plane error
        fov_params = self.camera.get_fov_params()
        tan_h = fov_params['tan_half_hfov']
        tan_v = fov_params['tan_half_vfov']
        if self._in_fov:
            obs[0] = np.clip(cam_result['p_bar'][0] / tan_h, -1.0, 1.0)
            obs[1] = np.clip(cam_result['p_bar'][1] / tan_v, -1.0, 1.0)
        else:
            obs[0] = np.clip(self._current_p_bar[0] / tan_h, -1.0, 1.0)
            obs[1] = np.clip(self._current_p_bar[1] / tan_v, -1.0, 1.0)

        # [2:4] Image-plane velocity (finite differences)
        dp_bar = (self._current_p_bar - self._prev_p_bar) / self.dt
        max_dp = 10.0
        obs[2] = np.clip(dp_bar[0] / max_dp, -1.0, 1.0)
        obs[3] = np.clip(dp_bar[1] / max_dp, -1.0, 1.0)

        # [4] Target visibility flag
        obs[4] = 1.0 if self._in_fov else 0.0

        # [5] Depth ESTIMATE (Jacobian-based, normalized to [0, 1])
        # This replaces the ground-truth distance that was in Stage 1b obs[5]
        obs[5] = np.clip(self._current_depth_est / self.norm_dist_max, 0.0, 1.0)

        # [6:9] Interceptor velocity in body frame (normalized, ego-state)
        body_vel = self.interceptor.get_body_velocity()
        obs[6:9] = np.clip(body_vel / self.v_max, -1.0, 1.0)

        # [9:12] Interceptor Euler angles — Stage 3a: normalize by max angle
        # Using max_pitch/max_roll gives better resolution than dividing by π
        # (pitch is typically <35°, so dividing by π wastes most of [-1,1])
        euler = self.interceptor.get_euler_angles()  # [roll, pitch, yaw]
        obs[9]  = np.clip(euler[0] / max(self.max_roll,  1e-6), -1.0, 1.0)   # roll
        obs[10] = np.clip(euler[1] / max(self.max_pitch, 1e-6), -1.0, 1.0)   # pitch
        obs[11] = np.clip(euler[2] / np.pi, -1.0, 1.0)                        # yaw

        # [12:16] Previous action (already in [-1, 1])
        obs[12:16] = self._prev_action.astype(np.float32)

        return obs

    def _build_info(self, cam_result: dict) -> dict:
        """Build the info dictionary.

        Includes ground-truth values for logging/evaluation (NOT in obs).

        Args:
            cam_result: Camera projection results dict.

        Returns:
            dict with diagnostic information.
        """
        p_r = self.interceptor.position - self.target.position
        return {
            'image_error':       float(np.linalg.norm(cam_result['p_bar'])),
            'relative_distance': float(np.linalg.norm(p_r)),
            'in_fov':            bool(cam_result['in_fov']),
            'fov_margin':        float(cam_result['fov_margin']),
            'episode_outcome':   self._episode_outcome,
            'step_count':        self._step_count,
            'interceptor_pos':   self.interceptor.position.copy(),
            'target_pos':        self.target.position.copy(),
            'p_bar':             cam_result['p_bar'].copy(),
            # Depth estimation diagnostics
            'depth_true':        float(cam_result.get('depth', 0.0)),
            'depth_est':         float(self._current_depth_est),
            'depth_error':       float(abs(
                self._current_depth_est - cam_result.get('depth', 0.0)
            )),
            # Stage 3a: attitude diagnostics
            'pitch_deg':         float(np.rad2deg(self.interceptor.pitch)),
            'roll_deg':          float(np.rad2deg(self.interceptor.roll)),
        }
