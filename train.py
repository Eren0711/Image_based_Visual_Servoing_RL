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


class CurriculumCallback(BaseCallback):
    """Advance domain-randomization curriculum frac during training.

    Pushes frac = min(1, t / anneal_steps) into every sub-env each rollout
    (via env_method), so the WindWrapper / IntermittentDetectionWrapper
    interpolate their DR bands from easy → full-hard over the first
    `anneal_steps` timesteps, then hold full-hard for the remainder. This
    lets the final checkpoints train on the true target distribution while
    avoiding the early destabilization of full DR from step 0.

    Intervention B of the HardNet robustness study.
    """

    def __init__(self, anneal_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.anneal_steps = int(anneal_steps)
        self._last_logged_frac = -1.0

    def _enable_curriculum_on_envs(self) -> None:
        # Best-effort: call enable_curriculum on any wrapper that has it.
        for method in ('enable_curriculum',):
            try:
                self.training_env.env_method(method)
            except Exception:
                pass

    def _on_training_start(self) -> None:
        self._enable_curriculum_on_envs()
        self.training_env.env_method('set_curriculum_frac', 0.0)

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / max(1, self.anneal_steps))
        # Update every rollout (~n_steps); cheap broadcast.
        self.training_env.env_method('set_curriculum_frac', frac)
        if self.verbose and abs(frac - self._last_logged_frac) >= 0.1:
            self.logger.record('curriculum/frac', frac)
            self._last_logged_frac = frac
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


def make_env(config: dict, use_noise_delay: bool = False, use_dkf: bool = False,
             use_imu_dkf: bool = True, use_cbf: bool = False,
             cbf_method: str = 'hocbf',
             cbf_alpha_fov: float = 100.0, cbf_alpha_att: float = 100.0,
             cbf_horizon_fov: int = 1, cbf_horizon_att: int = 8,
             cbf_safety_margin: float = 0.10,
             use_wind: bool = False, use_intermittent_det: bool = False,
             domain_randomize: bool = False,
             use_cbf_context: bool = False):
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

        # Stage 4b: wind and intermittent detection sit between the env and
        # NoiseDelay so the existing delay buffer sees the post-miss in_fov
        # flag. Order: env → Wind → IntermittentDet → NoiseDelay → DKF → CBF.
        if use_wind:
            from envs.wrappers.wind_wrapper import WindWrapper
            wind_cfg = config.get('stage4b', {}).get('wind', {})
            env = WindWrapper(
                env,
                sigma=wind_cfg.get('sigma', 1.0),
                theta=wind_cfg.get('theta', 0.5),
                k_drag=wind_cfg.get('k_drag', 0.1),
                randomize_per_episode=domain_randomize,
                randomization_ranges=wind_cfg.get('randomization_ranges', None),
            )

        if use_intermittent_det:
            from envs.wrappers.intermittent_detection_wrapper import \
                IntermittentDetectionWrapper
            det_cfg = config.get('stage4b', {}).get('detection', {})
            env = IntermittentDetectionWrapper(
                env,
                beta_1=det_cfg.get('beta_1', 8.0),
                beta_2=det_cfg.get('beta_2', 4.0),
                beta_3=det_cfg.get('beta_3', 1.0),
                sigma_base=det_cfg.get('sigma_base', 0.0),
                sigma_slope=det_cfg.get('sigma_slope', 0.0005),
                randomize_per_episode=domain_randomize,
                randomization_ranges=det_cfg.get('randomization_ranges', None),
            )

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
                use_imu=use_imu_dkf,
            )

        if use_cbf:
            from envs.wrappers.cbf_wrapper import CBFWrapper
            env = CBFWrapper(
                env,
                method=cbf_method,
                alpha_fov=cbf_alpha_fov,
                alpha_attitude=cbf_alpha_att,
                horizon_fov=cbf_horizon_fov,
                horizon_attitude=cbf_horizon_att,
                attitude_safety_margin=cbf_safety_margin,
                in_fov_only=True,
            )

        # Stage 4a Phase 4 (HardNet): append CBF (A, b) coefficients to the
        # observation so the in-policy differentiable projection can consume
        # them. Mutually exclusive with the external CBF filter (use_cbf):
        # the projection replaces the filter. Must be the OUTERMOST wrapper.
        if use_cbf_context:
            from envs.wrappers.cbf_context_wrapper import CBFContextWrapper
            env = CBFContextWrapper(
                env,
                alpha_fov=cbf_alpha_fov,
                alpha_attitude=cbf_alpha_att,
                attitude_safety_margin=cbf_safety_margin,
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
    parser.add_argument(
        '--no-imu-dkf', action='store_true', default=False,
        help='Disable IMU-aware DKF prediction (B-minimal upgrade). '
             'Use this for ablation: --dkf alone uses the constant-velocity '
             'DKF; --dkf --no-imu-dkf forces the legacy behavior; '
             '--dkf with IMU prediction is the default new behavior.'
    )
    parser.add_argument(
        '--cbf', action='store_true', default=False,
        help='Enable CBF safety filter in training loop (Stage 4a Phase 2/3)'
    )
    parser.add_argument(
        '--cbf-method', type=str, default='hocbf',
        choices=['hocbf', 'bisection'],
        help='CBF solver method'
    )
    parser.add_argument(
        '--cbf-alpha-fov', type=float, default=100.0,
        help='CBF FOV margin (hocbf: 1/s; bisection: per-step decay)'
    )
    parser.add_argument(
        '--cbf-alpha-att', type=float, default=100.0,
        help='CBF attitude margin'
    )
    parser.add_argument(
        '--cbf-horizon-fov', type=int, default=1,
        help='Prediction horizon (bisection / hocbf fallback)'
    )
    parser.add_argument(
        '--cbf-horizon-att', type=int, default=8,
        help='Prediction horizon (bisection / hocbf fallback)'
    )
    parser.add_argument(
        '--cbf-safety-margin', type=float, default=0.10,
        help='HOCBF inner safety margin (rad) on attitude limits'
    )
    parser.add_argument(
        '--wind', action='store_true', default=False,
        help='Enable WindWrapper (Stage 4b — OU-process wind + drag)'
    )
    parser.add_argument(
        '--intermittent-det', action='store_true', default=False,
        help='Enable IntermittentDetectionWrapper (Stage 4b)'
    )
    parser.add_argument(
        '--domain-randomize', action='store_true', default=False,
        help='Sample wind/detection params per episode (Stage 4b training)'
    )
    parser.add_argument(
        '--stage4b', action='store_true', default=False,
        help='Shortcut: enable wind + intermittent-det + domain-randomize'
    )
    parser.add_argument(
        '--lr-decay', action='store_true', default=False,
        help='Linear LR decay from initial value to 0 over training'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Master seed for reproducible runs (PPO init + vec_env). '
             'Used for multi-seed variance studies; None = nondeterministic.'
    )
    parser.add_argument(
        '--hardnet', action='store_true', default=False,
        help='Stage 4a Phase 4: in-policy differentiable CBF projection '
             '(HardNet). Adds CBFContextWrapper and uses the custom policy. '
             'Replaces the external CBF filter (do not combine with --cbf).'
    )
    parser.add_argument(
        '--hardnet-proj-iters', type=int, default=20,
        help='Dykstra iterations in the HardNet projection layer'
    )
    parser.add_argument(
        '--max-log-std', type=float, default=None,
        help='Intervention C: hard cap on policy log_std (e.g. 0.0 → std≤1). '
             'Prevents the std runaway seen in baseline HardNet training.'
    )
    parser.add_argument(
        '--ent-coef', type=float, default=None,
        help='Override entropy coefficient (config default 0.01). '
             'Intervention C uses 0.001.'
    )
    parser.add_argument(
        '--curriculum', action='store_true', default=False,
        help='Intervention B: anneal DR band easy→full-hard over '
             '--curriculum-steps timesteps.'
    )
    parser.add_argument(
        '--curriculum-steps', type=int, default=2_000_000,
        help='Timesteps over which the DR curriculum anneals to full-hard '
             '(then holds). Default 2M of a 3M run.'
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
    use_imu_dkf = not args.no_imu_dkf  # default ON, --no-imu-dkf disables

    if use_dkf and not use_noise_delay:
        print("  Note: --dkf implies --noise-delay. Enabling both.")
        use_noise_delay = True

    use_cbf = args.cbf
    use_wind = args.wind or args.stage4b
    use_intermittent_det = args.intermittent_det or args.stage4b
    domain_randomize = args.domain_randomize or args.stage4b
    use_hardnet = args.hardnet
    if use_hardnet and use_cbf:
        raise SystemExit(
            "Error: --hardnet and --cbf are mutually exclusive. HardNet's "
            "in-policy projection replaces the external CBF filter."
        )

    wrapper_str = ''
    if use_noise_delay:
        nd_cfg = config.get('noise_delay', {})
        wrapper_str += f"  + NoiseDelay (D={nd_cfg.get('delay', 3)}, σ={nd_cfg.get('sigma_noise', 0.03)})\n"
    if use_dkf:
        imu_tag = ' (IMU-aware)' if use_imu_dkf else ' (constant-velocity)'
        wrapper_str += f'  + DKF{imu_tag}\n'
    if use_cbf:
        wrapper_str += (f'  + CBF (alpha_fov={args.cbf_alpha_fov} '
                        f'alpha_att={args.cbf_alpha_att} '
                        f'h_fov={args.cbf_horizon_fov} '
                        f'h_att={args.cbf_horizon_att})\n')
    if use_wind:
        dr_tag = ' [DR]' if domain_randomize else ''
        wrapper_str += f'  + Wind (OU + drag){dr_tag}\n'
    if use_intermittent_det:
        dr_tag = ' [DR]' if domain_randomize else ''
        wrapper_str += f'  + IntermittentDetection{dr_tag}\n'
    if use_hardnet:
        wrapper_str += (f'  + CBFContext (HardNet in-policy projection, '
                        f'alpha_fov={args.cbf_alpha_fov} '
                        f'alpha_att={args.cbf_alpha_att} '
                        f'margin={args.cbf_safety_margin})\n')

    print(f"Creating {n_envs} parallel environments...")
    if wrapper_str:
        print(f"  Wrappers:\n{wrapper_str}")
    env_kwargs = dict(
        use_noise_delay=use_noise_delay, use_dkf=use_dkf,
        use_imu_dkf=use_imu_dkf, use_cbf=use_cbf,
        cbf_method=args.cbf_method,
        cbf_alpha_fov=args.cbf_alpha_fov,
        cbf_alpha_att=args.cbf_alpha_att,
        cbf_horizon_fov=args.cbf_horizon_fov,
        cbf_horizon_att=args.cbf_horizon_att,
        cbf_safety_margin=args.cbf_safety_margin,
        use_wind=use_wind,
        use_intermittent_det=use_intermittent_det,
        domain_randomize=domain_randomize,
        use_cbf_context=use_hardnet,
    )
    vec_env = make_vec_env(
        make_env(config, **env_kwargs),
        n_envs=n_envs,
        seed=args.seed,
    )

    # --- Learning rate (constant or linear-decay schedule) ---
    base_lr = float(train_cfg['learning_rate'])
    if args.lr_decay:
        # SB3 calls the schedule with `progress_remaining` in [1.0 → 0.0].
        # Linear: lr(t) = base_lr * progress_remaining. Final lr = 0.
        # Addresses the persistent overtraining pattern seen across stages
        # (every fine-tune's peak ckpt is well before the final ckpt). With
        # decay, the late-training updates are smaller and the final ckpt
        # should match (or come close to) the peak.
        def lr_schedule(progress_remaining: float) -> float:
            return base_lr * float(progress_remaining)
        learning_rate = lr_schedule
        lr_desc = f'{base_lr} → 0 (linear decay)'
    else:
        learning_rate = base_lr
        lr_desc = f'{base_lr} (constant)'

    # Entropy coefficient (config default, overridable by --ent-coef for
    # Intervention C).
    ent_coef = (args.ent_coef if args.ent_coef is not None
                else train_cfg['ent_coef'])

    # --- Create or load model ---
    if use_hardnet:
        # HardNet uses a custom policy and a 36-D augmented observation, so a
        # plain PPO.load (which expects matching obs space + policy class)
        # won't work. Build a fresh PPO with the HardNet policy, then (if
        # resuming) copy the matching network weights from the base model.
        from safety.hardnet_policy import HardNetActorCriticPolicy
        print("Creating PPO with HardNet (in-policy CBF projection)...")
        policy_kwargs = dict(
            n_base=16,
            n_constraints=4,
            proj_iters=args.hardnet_proj_iters,
            max_log_std=args.max_log_std,
        )
        model = PPO(
            policy=HardNetActorCriticPolicy,
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=train_cfg['n_steps'],
            batch_size=train_cfg['batch_size'],
            n_epochs=train_cfg['n_epochs'],
            gamma=train_cfg['gamma'],
            gae_lambda=train_cfg['gae_lambda'],
            clip_range=train_cfg['clip_range'],
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            verbose=1,
            tensorboard_log=log_dir,
        )
        if args.resume:
            print(f"  Warm-starting HardNet from: {args.resume}")
            base_model = PPO.load(args.resume, device=model.device)
            # The MLP / action_net / value_net / log_std see only the 16-D
            # base obs (via the slicing extractor), so their shapes match the
            # vanilla policy exactly. features_extractor has no params in
            # either case. strict=False tolerates any key differences.
            src = base_model.policy.state_dict()
            tgt = model.policy.state_dict()
            copied, skipped = [], []
            for kkey, val in src.items():
                if kkey in tgt and tgt[kkey].shape == val.shape:
                    tgt[kkey] = val
                    copied.append(kkey)
                else:
                    skipped.append(kkey)
            model.policy.load_state_dict(tgt, strict=False)
            print(f"  Warm-start copied {len(copied)} param tensors, "
                  f"skipped {len(skipped)}: {skipped}")
    elif args.resume:
        print(f"Resuming training from: {args.resume}")
        # Override tensorboard_log explicitly — PPO.load restores the original
        # path from the saved model, which sends fine-tune runs to the parent
        # stage's TB folder. Without this, --stage on the CLI is ignored for TB.
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=log_dir)
        # Also override learning rate if --lr-decay is set (otherwise the
        # restored model keeps its original constant LR).
        if args.lr_decay:
            model.learning_rate = learning_rate
            model._setup_lr_schedule()
            print(f"  Resumed model LR overridden to: {lr_desc}")
    else:
        print("Creating new PPO model...")
        model = PPO(
            policy=train_cfg['policy'],
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=train_cfg['n_steps'],
            batch_size=train_cfg['batch_size'],
            n_epochs=train_cfg['n_epochs'],
            gamma=train_cfg['gamma'],
            gae_lambda=train_cfg['gae_lambda'],
            clip_range=train_cfg['clip_range'],
            ent_coef=ent_coef,
            seed=args.seed,
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
        eval_env_fn=make_env(config, **env_kwargs),
        eval_freq=train_cfg.get('eval_freq', 1_000_000),
        n_eval_episodes=train_cfg.get('eval_n_episodes', 20),
        verbose=1,
    )
    callbacks = [metrics_callback, checkpoint_callback, det_eval_callback]
    if args.curriculum:
        callbacks.append(CurriculumCallback(
            anneal_steps=args.curriculum_steps, verbose=1))
    callback = CallbackList(callbacks)

    # --- Train ---
    print(f"\n{'='*60}")
    print(f"  IBVS Drone Interception — PPO Training")
    print(f"{'='*60}")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Stage           : {paths['stage']}")
    print(f"  Parallel envs   : {n_envs}")
    print(f"  Policy           : {train_cfg['policy']}")
    print(f"  Learning rate    : {lr_desc}")
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
