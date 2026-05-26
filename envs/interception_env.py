"""
IBVS Drone Interception Environment — Stage 1b
================================================
Gymnasium environment for RL-based image-guided drone interception.

Stage 1a → 1b changes:
  - Action semantics: velocity commands → acceleration commands
  - Observation: 18-dim → 22-dim (added previous action)
  - Reward: increased effort penalty, added jerk penalty
  - Drone dynamics: acceleration integration (no instant velocity reversal)

The interceptor drone observes the target's 2D projection on a pinhole camera
image and must simultaneously:
  1. Keep the target near the image center (minimize image-plane error)
  2. Close the 3D relative distance to achieve physical interception

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


class InterceptionEnv(gym.Env):
    """Gymnasium environment for image-based visual servoing interception.

    Observation space (22-dim continuous):
        [0:2]   p̄_x, p̄_y          — normalized image-plane error
        [2:4]   dp̄_x/dt, dp̄_y/dt  — image-plane velocity (finite diff)
        [4]     in_fov              — target visible (1) or lost (0)
        [5]     norm_p_r            — normalized relative distance [0, 1]
        [6:9]   v_body              — interceptor velocity in body frame (3D)
        [9:12]  roll, pitch, yaw    — interceptor Euler angles (normalized)
        [12:15] n_t                 — LOS unit vector in EFCS (3D)
        [15:18] v_r                 — relative velocity (normalized, 3D)
        [18:22] prev_action         — previous action (4D) for jerk awareness

    Action space (4-dim continuous, [-1, 1]):
        [0:3]   a_cmd       — body-frame acceleration commands (scaled by a_max)
        [3]     yaw_rate    — yaw rate command (scaled by yaw_rate_max)

    Reward: weighted sum of image centering, approach, FOV penalties,
            control effort, jerk penalty, and terminal bonuses/penalties.

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

        # --- Environment parameters ---
        self.dt = int_cfg['dt']
        self.v_max = int_cfg['v_max']
        self.a_max = int_cfg['a_max']
        self.yaw_rate_max = int_cfg['yaw_rate_max']
        self.max_steps = env_cfg['max_steps']
        self.d_success = env_cfg['d_success']
        self.fov_loss_limit = env_cfg['fov_loss_limit']
        self.init_dist_range = env_cfg['init_distance_range']
        self.norm_dist_max = env_cfg['norm_distance_max']
        self.norm_vel_max = env_cfg['norm_velocity_max']
        self.target_modes = tgt_cfg.get(
            'maneuver_modes',
            ['constant_velocity', 'sinusoidal', 'circular', 'random_aggressive']
        )
        # If maneuver_modes is missing or not a list, handle gracefully
        if not isinstance(self.target_modes, list):
            self.target_modes = list(TargetDrone.MODES)

        # --- Reward parameters ---
        self.w_image = rwd_cfg['w_image']
        self.w_approach = rwd_cfg['w_approach']
        self.w_fov_loss = rwd_cfg['w_fov_loss']
        self.w_boundary = rwd_cfg['w_boundary']
        self.w_effort = rwd_cfg['w_effort']
        self.w_jerk = rwd_cfg.get('w_jerk', -0.1)
        self.w_intercept = rwd_cfg['w_intercept']
        self.w_timeout = rwd_cfg['w_timeout']
        self.k1_image = rwd_cfg['k1_image']

        # --- Spaces ---
        # Observation: 22-dimensional (18 original + 4 prev action)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(22,), dtype=np.float32
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
        self._prev_action = np.zeros(4)  # Previous action for jerk penalty

        # Note: self.np_random is provided by gymnasium.Env and seeded
        # via super().reset(seed=...). We use it for all randomization.

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
            options: Optional dict (unused in Stage 1).

        Returns:
            observation: np.ndarray (22,) — initial observation.
            info: dict — initial info.
        """
        super().reset(seed=seed)
        # self.np_random is now seeded by Gymnasium's super().reset()

        # --- Reset counters ---
        self._step_count = 0
        self._fov_loss_counter = 0
        self._episode_outcome = 'running'
        self._prev_action = np.zeros(4)

        # --- Reset interceptor ---
        # Start at origin with random heading
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
        # Derive a deterministic seed for the target's internal RNG
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
            # Random distance within range
            dist = self.np_random.uniform(
                self.init_dist_range[0], self.init_dist_range[1]
            )

            # Random direction biased toward the camera's forward direction
            # In body frame, forward = x-axis → in EFCS rotated by R_b^e
            # Add some randomness but keep within FOV
            # Generate a random direction in body frame, mostly forward
            body_dir = np.array([
                self.np_random.uniform(0.5, 1.0),        # mostly forward
                self.np_random.uniform(-0.3, 0.3),        # slight lateral
                self.np_random.uniform(-0.3, 0.3),        # slight vertical
            ])
            body_dir /= np.linalg.norm(body_dir)

            # Transform to EFCS
            offset_efcs = R_be @ body_dir * dist

            target_pos = interceptor_pos + offset_efcs

            # Verify target is within FOV
            p_r = interceptor_pos - target_pos
            cam_result = self.camera.project(p_r, R_be)
            if cam_result['in_fov'] and cam_result['fov_margin'] > 0.2:
                break
        # If all attempts fail, the last position is used (may be at FOV edge)

        # Random target velocity
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
            observation: np.ndarray (22,)
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

        # --- FOV loss tracking ---
        if not self._in_fov:
            self._fov_loss_counter += 1
        else:
            self._fov_loss_counter = 0

        # --- Compute reward ---
        reward = self._compute_reward(
            cam_result, current_distance, action
        )

        # --- Check termination ---
        terminated = False
        truncated = False

        # Success: close enough to target
        if current_distance < self.d_success:
            terminated = True
            self._episode_outcome = 'success'
            reward += self.w_intercept  # Terminal bonus

        # FOV loss: target lost for too many consecutive steps
        elif self._fov_loss_counter >= self.fov_loss_limit:
            terminated = True
            self._episode_outcome = 'fov_loss'

        # Timeout
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

    def _compute_reward(self, cam_result: dict, current_distance: float,
                        action: np.ndarray) -> float:
        """Compute the per-step reward.

        Reward = w_image * r_image
               + w_approach * r_approach
               + w_fov_loss * (if lost)
               + w_boundary * r_boundary
               + w_effort * r_effort
               + w_jerk * r_jerk          ← NEW in Stage 1b
               + w_timeout

        Args:
            cam_result: Camera projection results dict.
            current_distance: Current ||p_r||.
            action: Raw action ([-1, 1] range) for effort/jerk penalty.

        Returns:
            float: total reward.
        """
        reward = 0.0

        # --- 1. Image centering reward ---
        # r_image = exp(-k1 * ||p̄||²)
        # Maximal (=1) when target is at image center, decays with error
        p_bar = cam_result['p_bar']
        image_error_sq = p_bar[0] ** 2 + p_bar[1] ** 2
        if self._in_fov:
            r_image = np.exp(-self.k1_image * image_error_sq)
        else:
            r_image = 0.0
        reward += self.w_image * r_image

        # --- 2. Approach reward ---
        # r_approach = -(current_distance - prev_distance)
        # Positive when distance decreases (approaching target)
        delta_dist = current_distance - self._prev_distance
        r_approach = -delta_dist
        reward += self.w_approach * r_approach

        # --- 3. FOV loss penalty ---
        if not self._in_fov:
            reward += self.w_fov_loss

        # --- 4. FOV boundary penalty ---
        # Penalize being close to FOV boundary (proportional to proximity)
        if self._in_fov:
            fov_margin = cam_result['fov_margin']
            r_boundary = max(0.0, 1.0 - fov_margin) ** 2
            reward += self.w_boundary * r_boundary

        # --- 5. Control effort penalty ---
        # Penalize large accelerations to encourage efficient control
        r_effort = np.sum(action ** 2)
        reward += self.w_effort * r_effort

        # --- 6. Jerk penalty (NEW in Stage 1b) ---
        # Penalize rapid changes in action to encourage smooth control
        action_delta = action - self._prev_action
        r_jerk = np.sum(action_delta ** 2)
        reward += self.w_jerk * r_jerk

        # --- 7. Time penalty ---
        # Encourages the agent to intercept quickly
        reward += self.w_timeout

        return reward

    def _build_observation(self, cam_result: dict) -> np.ndarray:
        """Build the 22-dimensional observation vector.

        All values are normalized to [-1, 1] or [0, 1].

        Stage 1b additions:
            [18:22] Previous action (4D) — allows the agent to be aware of
                    its current control state for smooth action transitions.

        Args:
            cam_result: Camera projection results dict.

        Returns:
            np.ndarray (22,) — observation vector.
        """
        obs = np.zeros(22, dtype=np.float32)

        # [0:2] Normalized image-plane error
        # Normalize by FOV half-angle tangent so that ±1 = FOV boundary
        fov_params = self.camera.get_fov_params()
        tan_h = fov_params['tan_half_hfov']
        tan_v = fov_params['tan_half_vfov']
        if self._in_fov:
            obs[0] = np.clip(cam_result['p_bar'][0] / tan_h, -1.0, 1.0)
            obs[1] = np.clip(cam_result['p_bar'][1] / tan_v, -1.0, 1.0)
        else:
            # If target is lost, set error to the last known direction (clipped)
            obs[0] = np.clip(self._current_p_bar[0] / tan_h, -1.0, 1.0)
            obs[1] = np.clip(self._current_p_bar[1] / tan_v, -1.0, 1.0)

        # [2:4] Image-plane velocity (finite differences)
        dp_bar = (self._current_p_bar - self._prev_p_bar) / self.dt
        # Normalize by a reasonable max velocity on image plane
        max_dp = 10.0  # Heuristic normalization constant
        obs[2] = np.clip(dp_bar[0] / max_dp, -1.0, 1.0)
        obs[3] = np.clip(dp_bar[1] / max_dp, -1.0, 1.0)

        # [4] Target visibility flag
        obs[4] = 1.0 if self._in_fov else 0.0

        # [5] Normalized relative distance
        p_r = self.interceptor.position - self.target.position
        rel_dist = np.linalg.norm(p_r)
        obs[5] = np.clip(rel_dist / self.norm_dist_max, 0.0, 1.0)

        # [6:9] Interceptor velocity in body frame (normalized)
        body_vel = self.interceptor.get_body_velocity()
        obs[6:9] = np.clip(body_vel / self.v_max, -1.0, 1.0)

        # [9:12] Interceptor Euler angles (normalized)
        euler = self.interceptor.get_euler_angles()  # [roll, pitch, yaw]
        obs[9] = np.clip(euler[0] / np.pi, -1.0, 1.0)   # roll / π
        obs[10] = np.clip(euler[1] / np.pi, -1.0, 1.0)   # pitch / π
        obs[11] = np.clip(euler[2] / np.pi, -1.0, 1.0)   # yaw / π

        # [12:15] LOS unit vector in EFCS (already unit, in [-1, 1])
        n_t = cam_result['n_t']
        obs[12:15] = np.clip(n_t, -1.0, 1.0)

        # [15:18] Relative velocity (normalized)
        v_r = self.interceptor.velocity - self.target.velocity
        obs[15:18] = np.clip(v_r / self.norm_vel_max, -1.0, 1.0)

        # [18:22] Previous action (already in [-1, 1])
        obs[18:22] = self._prev_action.astype(np.float32)

        return obs

    def _build_info(self, cam_result: dict) -> dict:
        """Build the info dictionary.

        Args:
            cam_result: Camera projection results dict.

        Returns:
            dict with diagnostic information.
        """
        p_r = self.interceptor.position - self.target.position
        return {
            'image_error': float(np.linalg.norm(cam_result['p_bar'])),
            'relative_distance': float(np.linalg.norm(p_r)),
            'in_fov': bool(cam_result['in_fov']),
            'fov_margin': float(cam_result['fov_margin']),
            'episode_outcome': self._episode_outcome,
            'step_count': self._step_count,
            'interceptor_pos': self.interceptor.position.copy(),
            'target_pos': self.target.position.copy(),
            'p_bar': cam_result['p_bar'].copy(),
        }
