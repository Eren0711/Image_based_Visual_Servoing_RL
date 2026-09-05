"""
Stage 4b / HardNet Evaluation Harness
======================================
HISTORICAL / NON-CANONICAL: this robustness/safety harness is outside the clean
fixed_camera_intercept_v1 MVP and is retained for prior-run reconstruction.

Runs a trained policy through the full realistic-perception stack
(wind + intermittent detection + noise/delay + DKF) under a chosen
difficulty condition, with one of three safety configurations:

  --mode none     : no safety filter, raw policy actions.
  --mode hocbf    : external HOCBF-QP filter (Stage 4a.3) on the actions.
  --mode hardnet  : in-policy differentiable CBF projection (Stage 4a.4).
                    The model must have been trained with --hardnet (its
                    policy is HardNetActorCriticPolicy and it expects the
                    36-D context-augmented observation).

Reports success / FOV-loss / timeout rates, detector drop rate, and ACTUAL
attitude safety (max |pitch|, max |roll|, and fraction of steps exceeding
the true limit) — the headline number for whether safety held.

Conditions (wind σ,θ,k_drag ; detector β₁,β₂,β₃):
  nominal : σ=1.0 θ=0.5 k=0.10 ; β₁=8 β₂=4 β₃=+1.0   (~50% drop)
  hard    : σ=1.5 θ=0.5 k=0.15 ; β₁=6 β₂=4 β₃= 0.0   (~75% drop)
  worst   : σ=1.5 θ=0.5 k=0.15 ; β₁=4 β₂=4 β₃=−1.0   (~90% drop)
  clean   : no wind, no detector drops (sanity / upper bound)

Usage:
  python scripts/legacy/eval_stage4b.py --model <path> --mode hardnet --condition nominal --episodes 200
"""

import argparse
import os
import sys
import yaml
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO

from envs.interception_env import InterceptionEnv
from envs.wrappers.wind_wrapper import WindWrapper
from envs.wrappers.intermittent_detection_wrapper import IntermittentDetectionWrapper
from envs.wrappers.noise_delay_wrapper import NoiseDelayWrapper
from envs.wrappers.dkf_wrapper import DKFWrapper

CONDITIONS = {
    #            σ     θ    k_drag  β₁   β₂   β₃
    'clean':   (0.001, 0.5, 0.10,   8.0, 4.0,  8.0),
    'nominal': (1.0,   0.5, 0.10,   8.0, 4.0,  1.0),
    'hard':    (1.5,   0.5, 0.15,   6.0, 4.0,  0.0),
    'worst':   (1.5,   0.5, 0.15,   4.0, 4.0, -1.0),
}


def build_env(cfg, seed, condition, mode,
              alpha_fov, alpha_att, safety_margin):
    s, th, kd, b1, b2, b3 = CONDITIONS[condition]
    env = InterceptionEnv(config=cfg)
    env = WindWrapper(env, sigma=s, theta=th, k_drag=kd, seed=seed)
    env = IntermittentDetectionWrapper(
        env, beta_1=b1, beta_2=b2, beta_3=b3, seed=seed)
    nd = cfg['noise_delay']
    dkf_c = cfg['dkf']
    env = NoiseDelayWrapper(env, delay=nd['delay'], sigma_noise=nd['sigma_noise'])
    env = DKFWrapper(
        env, delay=nd['delay'], dt=cfg['interceptor']['dt'],
        sigma_pos_process=dkf_c['sigma_pos_process'],
        sigma_vel_process=dkf_c['sigma_vel_process'],
        sigma_measurement=dkf_c['sigma_measurement'], use_imu=True)
    if mode == 'hocbf':
        from envs.wrappers.cbf_wrapper import CBFWrapper
        env = CBFWrapper(
            env, method='hocbf', alpha_fov=alpha_fov,
            alpha_attitude=alpha_att, attitude_safety_margin=safety_margin)
    elif mode == 'hardnet':
        from envs.wrappers.cbf_context_wrapper import CBFContextWrapper
        env = CBFContextWrapper(
            env, alpha_fov=alpha_fov, alpha_attitude=alpha_att,
            attitude_safety_margin=safety_margin)
    # mode == 'none': no safety layer
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Model path (no .zip)')
    ap.add_argument('--mode', choices=['none', 'hocbf', 'hardnet'],
                    default='hardnet')
    ap.add_argument('--condition', choices=list(CONDITIONS), default='nominal')
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument(
        '--config',
        default=os.path.join(ROOT, 'configs', 'legacy', 'stage3_stage4.yaml'))
    ap.add_argument('--alpha-fov', type=float, default=100.0)
    ap.add_argument('--alpha-att', type=float, default=100.0)
    ap.add_argument('--safety-margin', type=float, default=0.10)
    ap.add_argument('--max-pitch', type=float, default=0.611,
                    help='True attitude limit for exceedance counting (rad)')
    args = ap.parse_args()

    config_path = (args.config if os.path.isabs(args.config)
                   else os.path.join(ROOT, args.config))
    model_path = (args.model if os.path.isabs(args.model)
                  else os.path.join(ROOT, args.model))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.mode == 'hardnet':
        # Register the historical custom policy before deserializing it.
        from safety.hardnet_policy import HardNetActorCriticPolicy  # noqa: F401
    model = PPO.load(model_path)

    succ = fov = tmo = 0
    drop_rates = []
    max_pitch = max_roll = 0.0
    n_pitch_over = n_roll_over = total_steps = 0

    for ep in range(args.episodes):
        env = build_env(cfg, args.seed + ep, args.condition, args.mode,
                        args.alpha_fov, args.alpha_att, args.safety_margin)
        obs, _ = env.reset(seed=args.seed + ep)
        last = {}
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, t, tr, info = env.step(a)
            last = info
            p = abs(env.unwrapped.interceptor.pitch)
            rl = abs(env.unwrapped.interceptor.roll)
            max_pitch = max(max_pitch, p)
            max_roll = max(max_roll, rl)
            if p > args.max_pitch:
                n_pitch_over += 1
            if rl > args.max_pitch:
                n_roll_over += 1
            total_steps += 1
            if t or tr:
                break
        out = env.unwrapped._episode_outcome
        if out == 'success':
            succ += 1
        elif out == 'fov_loss':
            fov += 1
        else:
            tmo += 1
        drop_rates.append(last.get('det_episode', {}).get('drop_rate', 0.0))

    n = args.episodes
    print(f"\n{'='*64}")
    print(f"  Stage 4b eval — mode={args.mode}  condition={args.condition}  "
          f"n={n}")
    print(f"{'='*64}")
    print(f"  Success      : {100*succ/n:5.1f}%  ({succ}/{n})")
    print(f"  FOV loss     : {100*fov/n:5.1f}%  ({fov}/{n})")
    print(f"  Timeout      : {100*tmo/n:5.1f}%  ({tmo}/{n})")
    print(f"  Avg drop rate: {100*np.mean(drop_rates):5.1f}%")
    print(f"  --- attitude safety (true limit {args.max_pitch:.3f} rad) ---")
    print(f"  max|pitch|   : {max_pitch:.3f}  "
          f"exceed: {n_pitch_over} ({100*n_pitch_over/total_steps:.2f}% of steps)")
    print(f"  max|roll|    : {max_roll:.3f}  "
          f"exceed: {n_roll_over} ({100*n_roll_over/total_steps:.2f}% of steps)")
    print(f"{'='*64}")


if __name__ == '__main__':
    main()
