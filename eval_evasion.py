"""
Evasion Evaluation Harness — 6-DOF Equal-Agility Target
========================================================
Evaluates a trained policy against the 6-DOF symmetric-agility target across
the evasion curriculum, reporting interception success AND an explicit
FOV-retention metric (the IBVS-specific objective), separately from
success/FOV-loss outcomes.

Metrics reported per (policy, evasion level):
  success_rate       : fraction of episodes ending in interception (d<d_succ)
  fov_loss_rate      : fraction ending in terminal FOV loss
  timeout_rate       : fraction ending in timeout
  fov_retention      : MEAN fraction of in-FOV steps per episode (the headline
                       FOV metric — how well the target was kept in frame over
                       the whole episode, not just whether it was lost)
  mean_fov_margin    : mean FOV margin while in-FOV (1=centered, 0=edge)
  attitude_exceed    : fraction of steps exceeding the 0.611 rad attitude limit

Stack: matches the trained policy automatically (HardNet 36-D vs vanilla 16-D
detected from the model observation space). Noise + delay + DKF always on
(the realistic perception stack), CBF context added for HardNet policies.

Usage:
  python eval_evasion.py --model <path> --level cruise --episodes 100
  python eval_evasion.py --model <path> --all-levels --episodes 100
"""

import argparse
import yaml
import numpy as np
from stable_baselines3 import PPO

from envs.interception_env import InterceptionEnv
from envs.wrappers.noise_delay_wrapper import NoiseDelayWrapper
from envs.wrappers.dkf_wrapper import DKFWrapper
from envs.wrappers.cbf_context_wrapper import CBFContextWrapper
from safety.hardnet_policy import HardNetActorCriticPolicy  # noqa: register class
from models.target_6dof import EVASION_LEVELS


def build_env(cfg, level):
    """Full realistic-perception stack with a 6-DOF target at one level."""
    cfg = yaml.safe_load(yaml.safe_dump(cfg))  # deep copy
    cfg['target']['model'] = 'sixdof'
    cfg['target']['maneuver_modes'] = [level]
    env = InterceptionEnv(config=cfg)
    nd, dkf = cfg['noise_delay'], cfg['dkf']
    env = NoiseDelayWrapper(env, delay=nd['delay'], sigma_noise=nd['sigma_noise'])
    env = DKFWrapper(env, delay=nd['delay'], dt=cfg['interceptor']['dt'],
                     sigma_pos_process=dkf['sigma_pos_process'],
                     sigma_vel_process=dkf['sigma_vel_process'],
                     sigma_measurement=dkf['sigma_measurement'], use_imu=True)
    return env


def eval_level(model, cfg, level, n_eps, seed0, hardnet, max_pitch=0.611):
    succ = fov = tmo = 0
    fov_retentions = []
    fov_margins = []
    att_exceed_steps = 0
    total_steps = 0
    for i in range(n_eps):
        env = build_env(cfg, level)
        if hardnet:
            env = CBFContextWrapper(env, alpha_fov=100.0, alpha_attitude=100.0,
                                    attitude_safety_margin=0.10)
        obs, info = env.reset(seed=seed0 + i)
        base = env.unwrapped
        in_fov_steps = 0
        ep_steps = 0
        ep_margin_sum = 0.0
        ep_margin_n = 0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, t, tr, info = env.step(a)
            ep_steps += 1
            total_steps += 1
            if info.get('in_fov', False):
                in_fov_steps += 1
                ep_margin_sum += info.get('fov_margin', 0.0)
                ep_margin_n += 1
            if abs(base.interceptor.pitch) > max_pitch or \
               abs(base.interceptor.roll) > max_pitch:
                att_exceed_steps += 1
            if t or tr:
                break
        o = base._episode_outcome
        succ += o == 'success'; fov += o == 'fov_loss'; tmo += o == 'timeout'
        fov_retentions.append(in_fov_steps / max(ep_steps, 1))
        if ep_margin_n > 0:
            fov_margins.append(ep_margin_sum / ep_margin_n)
    return {
        'level': level, 'n': n_eps,
        'success': 100 * succ / n_eps,
        'fov_loss': 100 * fov / n_eps,
        'timeout': 100 * tmo / n_eps,
        'fov_retention': 100 * np.mean(fov_retentions),
        'mean_fov_margin': np.mean(fov_margins) if fov_margins else 0.0,
        'attitude_exceed': 100 * att_exceed_steps / max(total_steps, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--level', default='cruise', choices=EVASION_LEVELS)
    ap.add_argument('--all-levels', action='store_true')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=3000)
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = PPO.load(args.model)
    hardnet = (model.observation_space.shape[0] == 36)

    levels = EVASION_LEVELS if args.all_levels else [args.level]
    print(f"Model: {args.model}  (HardNet={hardnet})")
    print(f"{'level':<16}{'succ%':>7}{'fovloss%':>9}{'timeout%':>9}"
          f"{'FOVretain%':>11}{'fovMargin':>10}{'attExceed%':>11}")
    print('-' * 73)
    for lvl in levels:
        r = eval_level(model, cfg, lvl, args.episodes, args.seed, hardnet)
        print(f"{r['level']:<16}{r['success']:>7.1f}{r['fov_loss']:>9.1f}"
              f"{r['timeout']:>9.1f}{r['fov_retention']:>11.1f}"
              f"{r['mean_fov_margin']:>10.3f}{r['attitude_exceed']:>11.3f}")


if __name__ == '__main__':
    main()
