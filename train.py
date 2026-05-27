"""PPO Training Script — Stages 1-2b
===================================
Trains a PPO agent on the IBVS Drone Interception environment using
Stable-Baselines3.

Usage:
    python train.py                          # default config
    python train.py --config custom.yaml     # custom config
    python train.py --timesteps 2000000      # override timesteps
    python train.py --noise-delay --dkf      # Stage 2b with wrappers

Features:
    - Vectorized environments for parallel training
    - TensorBoard logging of training metrics
    - Custom callback for episode outcome tracking
    - Periodic model checkpointing
    - Optional noise/delay + DKF wrappers (Stage 2b)
"""

import os
import argparse
import yaml
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.logger import configure

from envs.interception_env import InterceptionEnv
from experiment_paths import get_stage_paths, ensure_stage_dirs


class DeterministicEvalCallback(BaseCallback):
    """Run N deterministic eval episodes every M training steps.

    Why this exists: `InterceptionMetricsCallback` logs outcomes from the
    *stochastic* training rollouts, which over-report success because PPO's
    action noise occasionally rescues a policy that can't actually solve the
    task deterministically. Stage 3a-v1 showed a 15× gap (30% stochastic vs
    2% deterministic) that went unnoticed for 15M steps. This callback runs
    a real deterministic eval on a fresh env and logs `eval/det_*` so the
    gap is visible in TensorBoard.
    """

    def __init__(self, eval_env_fn, eval_freq: int = 1_000_000,
                 n_eval_episodes: int = 20, seed_base: int = 10_000,
                 verbose: int = 0):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed_base = seed_base
        self._last_eval_step = 0
        self._eval_env = None

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        if self._eval_env is None:
            self._eval_env = self.eval_env_fn()

        outcomes = []
        distances = []
        image_errors = []
        for i in range(self.n_eval_episodes):
            obs, info = self._eval_env.reset(seed=self.seed_base + i)
            ep_img_err = []
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _r, terminated, truncated, info = self._eval_env.step(action)
                ep_img_err.append(info.get('image_error', 0.0))
                done = terminated or truncated
            outcomes.append(info.get('episode_outcome', 'unknown'))
            distances.append(info.get('relative_distance', float('nan')))
            image_errors.append(float(np.mean(ep_img_err)) if ep_img_err else 0.0)

        n = len(outcomes)
        n_success = sum(1 for o in outcomes if o == 'success')
        n_fov = sum(1 for o in outcomes if o == 'fov_loss')
        n_timeout = sum(1 for o in outcomes if o == 'timeout')

        self.logger.record('eval/det_success_rate', n_success / n)
        self.logger.record('eval/det_fov_loss_rate', n_fov / n)
        self.logger.record('eval/det_timeout_rate', n_timeout / n)
        self.logger.record('eval/det_mean_distance', float(np.mean(distances)))
        self.logger.record('eval/det_mean_image_error', float(np.mean(image_errors)))

        if self.verbose:
            print(f"  [det-eval @ {self.num_timesteps:,} steps] "
                  f"success={n_success}/{n} ({100*n_success/n:.0f}%)  "
                  f"fov={n_fov}/{n}  timeout={n_timeout}/{n}  "
                  f"mean_dist={np.mean(distances):.1f}m")

        return True


class InterceptionMetricsCallback(BaseCallback):
    """Custom callback to log episode-level metrics to TensorBoard.

    Tracks:
        - Episode success rate (rolling window)
        - Mean image error at episode end
        - Mean relative distance at episode end
        - Episode length statistics
        - Episode outcome distribution
    """

    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_outcomes = []
        self.episode_image_errors = []
        self.episode_distances = []

    def _on_step(self) -> bool:
        """Called at each environment step.

        Checks for completed episodes in the vectorized env infos and
        logs aggregated metrics.
        """
        # Check for episode completions in vectorized environments
        infos = self.locals.get('infos', [])
        for info in infos:
            # SB3 wraps terminal info in 'terminal_info' or provides
            # episode stats in 'episode' key for Monitor wrapper.
            # We check for our custom keys:
            if 'episode_outcome' in info:
                outcome = info['episode_outcome']
                if outcome != 'running':
                    self.episode_outcomes.append(outcome)
                    self.episode_image_errors.append(
                        info.get('image_error', 0.0)
                    )
                    self.episode_distances.append(
                        info.get('relative_distance', 0.0)
                    )

        # Log metrics every N steps
        if self.n_calls % 2048 == 0 and len(self.episode_outcomes) > 0:
            recent = self.episode_outcomes[-self.window_size:]

            # Success rate
            n_success = sum(1 for o in recent if o == 'success')
            success_rate = n_success / len(recent)
            self.logger.record('custom/success_rate', success_rate)

            # FOV loss rate
            n_fov_loss = sum(1 for o in recent if o == 'fov_loss')
            fov_loss_rate = n_fov_loss / len(recent)
            self.logger.record('custom/fov_loss_rate', fov_loss_rate)

            # Timeout rate
            n_timeout = sum(1 for o in recent if o == 'timeout')
            timeout_rate = n_timeout / len(recent)
            self.logger.record('custom/timeout_rate', timeout_rate)

            # Mean image error (last episodes)
            recent_errors = self.episode_image_errors[-self.window_size:]
            self.logger.record(
                'custom/mean_image_error', np.mean(recent_errors)
            )

            # Mean final distance
            recent_dists = self.episode_distances[-self.window_size:]
            self.logger.record(
                'custom/mean_final_distance', np.mean(recent_dists)
            )

            # Total episodes
            self.logger.record(
                'custom/total_episodes', len(self.episode_outcomes)
            )

        return True


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        dict: parsed configuration.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(config: dict, use_noise_delay: bool = False, use_dkf: bool = False):
    """Factory function for creating InterceptionEnv instances.

    Args:
        config:          Full configuration dictionary.
        use_noise_delay: If True, wrap env with NoiseDelayWrapper (Stage 2b).
        use_dkf:         If True, wrap env with DKFWrapper (Stage 2b).
                         Requires use_noise_delay=True.

    Returns:
        Callable that creates a new environment instance.
    """
    def _init():
        env = InterceptionEnv(config=config)

        if use_noise_delay:
            from envs.wrappers.noise_delay_wrapper import NoiseDelayWrapper
            nd_cfg = config.get('noise_delay', {})
            env = NoiseDelayWrapper(
                env,
                delay=nd_cfg.get('delay', 3),
                sigma_noise=nd_cfg.get('sigma_noise', 0.03),
            )

        if use_dkf:
            from envs.wrappers.dkf_wrapper import DKFWrapper
            dkf_cfg = config.get('dkf', {})
            nd_cfg = config.get('noise_delay', {})
            env = DKFWrapper(
                env,
                delay=nd_cfg.get('delay', 3),
                dt=config['interceptor']['dt'],
                sigma_pos_process=dkf_cfg.get('sigma_pos_process', 0.01),
                sigma_vel_process=dkf_cfg.get('sigma_vel_process', 0.5),
                sigma_measurement=dkf_cfg.get('sigma_measurement', 0.03),
            )

        return env
    return _init


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description='Train PPO agent for IBVS drone interception'
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--stage', type=str, default=None,
        help='Experiment stage name for outputs (default: config experiment.stage)'
    )
    parser.add_argument(
        '--timesteps', type=int, default=None,
        help='Override total training timesteps'
    )
    parser.add_argument(
        '--n-envs', type=int, default=None,
        help='Override number of parallel environments'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to a saved model to resume training from'
    )
    parser.add_argument(
        '--noise-delay', action='store_true', default=False,
        help='Enable noise + delay wrapper (Stage 2b)'
    )
    parser.add_argument(
        '--dkf', action='store_true', default=False,
        help='Enable DKF wrapper on top of noise/delay (Stage 2b)'
    )
    args = parser.parse_args()

    # --- Load config ---
    config = load_config(args.config)
    train_cfg = config['training']
    paths = get_stage_paths(config, args.stage)

    total_timesteps = args.timesteps or train_cfg['total_timesteps']
    n_envs = args.n_envs or train_cfg['n_envs']
    log_dir = str(paths['tensorboard'])
    save_dir = str(paths['models'])
    checkpoint_freq = train_cfg.get('checkpoint_freq', 50000)

    # --- Create directories ---
    ensure_stage_dirs(paths, 'tensorboard', 'models')

    # --- Create vectorized environment ---
    use_noise_delay = args.noise_delay
    use_dkf = args.dkf

    if use_dkf and not use_noise_delay:
        print("  Note: --dkf implies --noise-delay. Enabling both.")
        use_noise_delay = True

    wrapper_str = ''
    if use_noise_delay:
        nd_cfg = config.get('noise_delay', {})
        wrapper_str += f"  + NoiseDelay (D={nd_cfg.get('delay', 3)}, σ={nd_cfg.get('sigma_noise', 0.03)})\n"
    if use_dkf:
        wrapper_str += '  + DKF\n'

    print(f"Creating {n_envs} parallel environments...")
    if wrapper_str:
        print(f"  Wrappers:\n{wrapper_str}")
    vec_env = make_vec_env(
        make_env(config, use_noise_delay=use_noise_delay, use_dkf=use_dkf),
        n_envs=n_envs,
    )

    # --- Create or load model ---
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = PPO.load(args.resume, env=vec_env)
    else:
        print("Creating new PPO model...")
        model = PPO(
            policy=train_cfg['policy'],
            env=vec_env,
            learning_rate=train_cfg['learning_rate'],
            n_steps=train_cfg['n_steps'],
            batch_size=train_cfg['batch_size'],
            n_epochs=train_cfg['n_epochs'],
            gamma=train_cfg['gamma'],
            gae_lambda=train_cfg['gae_lambda'],
            clip_range=train_cfg['clip_range'],
            ent_coef=train_cfg['ent_coef'],
            verbose=1,
            tensorboard_log=log_dir,
        )

    # --- Set up callbacks ---
    metrics_callback = InterceptionMetricsCallback(
        window_size=100, verbose=0
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=save_dir,
        name_prefix='ibvs_ppo',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    det_eval_callback = DeterministicEvalCallback(
        eval_env_fn=make_env(
            config, use_noise_delay=use_noise_delay, use_dkf=use_dkf,
        ),
        eval_freq=train_cfg.get('eval_freq', 1_000_000),
        n_eval_episodes=train_cfg.get('eval_n_episodes', 20),
        verbose=1,
    )
    callback = CallbackList([metrics_callback, checkpoint_callback, det_eval_callback])

    # --- Train ---
    print(f"\n{'='*60}")
    print(f"  IBVS Drone Interception — PPO Training")
    print(f"{'='*60}")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Stage           : {paths['stage']}")
    print(f"  Parallel envs   : {n_envs}")
    print(f"  Policy           : {train_cfg['policy']}")
    print(f"  Learning rate    : {train_cfg['learning_rate']}")
    print(f"  Batch size       : {train_cfg['batch_size']}")
    print(f"  Log directory    : {log_dir}")
    print(f"  Save directory   : {save_dir}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    # --- Save final model ---
    final_path = os.path.join(save_dir, 'ibvs_ppo_final')
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    # --- Print training summary ---
    outcomes = metrics_callback.episode_outcomes
    if outcomes:
        n_total = len(outcomes)
        n_success = sum(1 for o in outcomes if o == 'success')
        n_fov = sum(1 for o in outcomes if o == 'fov_loss')
        n_timeout = sum(1 for o in outcomes if o == 'timeout')

        print(f"\n{'='*60}")
        print(f"  Training Summary")
        print(f"{'='*60}")
        print(f"  Total episodes   : {n_total}")
        print(f"  Success          : {n_success} ({100*n_success/n_total:.1f}%)")
        print(f"  FOV loss         : {n_fov} ({100*n_fov/n_total:.1f}%)")
        print(f"  Timeout          : {n_timeout} ({100*n_timeout/n_total:.1f}%)")

        # Last 100 episodes
        last = outcomes[-100:]
        n_s = sum(1 for o in last if o == 'success')
        print(f"\n  Last 100 episodes:")
        print(f"    Success rate   : {100*n_s/len(last):.1f}%")
        print(f"{'='*60}")

    vec_env.close()


if __name__ == '__main__':
    main()
