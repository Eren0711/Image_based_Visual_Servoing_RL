"""
Batch Video Generator — every stage and sub-experiment
=======================================================
Renders an agent-vs-target replay animation for each trained stage directory
under logs/stages/, saving the video INTO that stage's own folder
(logs/stages/<stage>/videos/replay_best.mp4). Fully automated: discovers all
stages, reconstructs the correct environment wrapper stack from the stage
name, loads the right policy class (vanilla MLP or HardNet), and animates the
best of N evaluation episodes via the existing visualize.create_animation.

Why this is not just `visualize.py` in a loop:
  * visualize.py builds a BARE env (no noise/DKF/CBF/HardNet) and a vanilla
    PPO.load. That is wrong for the 3b/4a/4b stages — it would neither apply
    the noise/safety the policy was trained with, nor be able to load the
    36-D HardNet policy at all.
  * This driver infers the stack from the stage name (see STACK_RULES) so the
    rendered behavior reflects how the policy actually operates.

Usage:
  python scripts/legacy/generate_all_videos.py                 # all stages
  python scripts/legacy/generate_all_videos.py --stages stage3b_noisy_mild stage4b_dr_finetune
  python scripts/legacy/generate_all_videos.py --evasive
  python scripts/legacy/generate_all_videos.py --episodes 5 --fps 20 --skip 2
  python scripts/legacy/generate_all_videos.py --format gif
"""

import os
import sys
import re
import glob
import argparse
import yaml
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO

from envs.interception_env import InterceptionEnv
from experiment_paths import get_stage_paths, ensure_stage_dirs
import visualize  # reuse collect_episode + create_animation

# ----------------------------------------------------------------------
# Per-stage stack inference from the directory name
# ----------------------------------------------------------------------
def infer_stack(stage: str) -> dict:
    """Infer which wrappers/policy a stage used, from its name.

    Returns a dict: {noise_delay, dkf, cbf, cbf_method, hardnet}.
    Conservative: when unsure, render the bare guidance behavior (no safety),
    which still produces a valid agent-vs-target video.
    """
    s = stage.lower()
    cfg = {'noise_delay': False, 'dkf': False, 'cbf': False,
           'cbf_method': 'hocbf', 'hardnet': False}
    # Noise/DKF: every noisy + 3b + 4a + 4b stage trained with them.
    if any(k in s for k in ['noisy', '3b', '4a', '4b', 'hardnet', 'hocbf',
                            'cbf', 'dkf']):
        cfg['noise_delay'] = True
        cfg['dkf'] = True
    if 'stage2b' in s:
        cfg['noise_delay'] = True
        cfg['dkf'] = True
    # Safety layer
    if 'hardnet' in s:
        cfg['hardnet'] = True
    elif 'hocbf' in s or ('cbf' in s and 'finetune' in s) or 'cbf_finetune' in s:
        cfg['cbf'] = True
        cfg['cbf_method'] = 'hocbf' if 'hocbf' in s else 'bisection'
    return cfg


def build_env(config, stack, evasive=False):
    """Construct the wrapped env matching a stage's training stack."""
    if evasive:
        # Swap the target maneuver set to the reactive high-g evasive mode and
        # lift the target a_max so the 2g turns are not clamped.
        config = yaml_deepcopy(config)
        config['target']['maneuver_modes'] = ['evasive']
        config['target']['a_max'] = max(config['target'].get('a_max', 5.0), 25.0)

    env = InterceptionEnv(config=config)

    if stack['hardnet']:
        # HardNet needs noise/delay + DKF + the CBF context wrapper (36-D obs).
        from envs.wrappers.noise_delay_wrapper import NoiseDelayWrapper
        from envs.wrappers.dkf_wrapper import DKFWrapper
        from envs.wrappers.cbf_context_wrapper import CBFContextWrapper
        nd = config['noise_delay']; dkf = config['dkf']
        env = NoiseDelayWrapper(env, delay=nd['delay'], sigma_noise=nd['sigma_noise'])
        env = DKFWrapper(env, delay=nd['delay'], dt=config['interceptor']['dt'],
                         sigma_pos_process=dkf['sigma_pos_process'],
                         sigma_vel_process=dkf['sigma_vel_process'],
                         sigma_measurement=dkf['sigma_measurement'], use_imu=True)
        env = CBFContextWrapper(env, alpha_fov=100.0, alpha_attitude=100.0,
                                attitude_safety_margin=0.10)
        return env

    if stack['noise_delay']:
        from envs.wrappers.noise_delay_wrapper import NoiseDelayWrapper
        nd = config['noise_delay']
        env = NoiseDelayWrapper(env, delay=nd['delay'], sigma_noise=nd['sigma_noise'])
    if stack['dkf']:
        from envs.wrappers.dkf_wrapper import DKFWrapper
        nd = config['noise_delay']; dkf = config['dkf']
        env = DKFWrapper(env, delay=nd['delay'], dt=config['interceptor']['dt'],
                         sigma_pos_process=dkf['sigma_pos_process'],
                         sigma_vel_process=dkf['sigma_vel_process'],
                         sigma_measurement=dkf['sigma_measurement'], use_imu=True)
    if stack['cbf']:
        from envs.wrappers.cbf_wrapper import CBFWrapper
        env = CBFWrapper(env, method=stack['cbf_method'], alpha_fov=100.0,
                         alpha_attitude=100.0, attitude_safety_margin=0.10,
                         in_fov_only=True)
    return env


def yaml_deepcopy(d):
    """Cheap deep copy via YAML round-trip (config is small & plain)."""
    return yaml.safe_load(yaml.safe_dump(d))


def load_model(model_path, stack):
    """Load the policy, registering the HardNet custom class if needed."""
    if stack['hardnet']:
        from safety.hardnet_policy import HardNetActorCriticPolicy  # noqa
    return PPO.load(model_path)


def pick_model(stage_dir):
    """Prefer ibvs_ppo_best.zip, else the highest-step checkpoint, else final."""
    mdir = os.path.join(stage_dir, 'models')
    best = os.path.join(mdir, 'ibvs_ppo_best.zip')
    if os.path.exists(best):
        return best[:-4]
    steps = glob.glob(os.path.join(mdir, 'ibvs_ppo_*_steps.zip'))
    if steps:
        steps.sort(key=lambda p: int(re.search(r'_(\d+)_steps', p).group(1)))
        return steps[-1][:-4]
    final = os.path.join(mdir, 'ibvs_ppo_final.zip')
    if os.path.exists(final):
        return final[:-4]
    return None


def render_stage(stage, config, args):
    """Render one stage's video into logs/stages/<stage>/videos/."""
    stage_dir = os.path.join(ROOT, 'logs/stages', stage)
    model_path = pick_model(stage_dir)
    if model_path is None:
        print(f"[skip] {stage}: no model checkpoint")
        return False

    stack = infer_stack(stage)
    tag = []
    if stack['hardnet']: tag.append('HardNet')
    elif stack['cbf']: tag.append(f"CBF:{stack['cbf_method']}")
    if stack['dkf']: tag.append('DKF')
    if args.evasive: tag.append('EVASIVE-TGT')
    tagstr = ('[' + ','.join(tag) + ']') if tag else '[bare]'

    try:
        env = build_env(config, stack, evasive=args.evasive)
        model = load_model(model_path, stack)
    except Exception as e:
        print(f"[FAIL] {stage} {tagstr}: build/load error: {e}")
        return False

    # Guard: legacy stages (1a/1b/2a) were trained on an 18-D privileged
    # observation that the current 16-D config cannot reproduce. Detect the
    # mismatch and skip cleanly rather than crash or render a wrong rollout.
    exp_obs = model.observation_space.shape
    env_obs = env.observation_space.shape
    if exp_obs != env_obs:
        print(f"[skip] {stage} {tagstr}: obs mismatch model{exp_obs} vs "
              f"env{env_obs} (legacy obs space not reproducible by current config)")
        return False

    # Collect N episodes, pick the best (closest final distance).
    data_list = []
    for i in range(args.episodes):
        try:
            d = visualize.collect_episode(model, env, deterministic=True,
                                          seed=args.seed + i)
            data_list.append(d)
        except Exception as e:
            print(f"[FAIL] {stage} {tagstr}: rollout error: {e}")
            return False
    idx = int(np.argmin([d['relative_distance'][-1] for d in data_list]))

    vid_dir = os.path.join(stage_dir, 'videos')
    os.makedirs(vid_dir, exist_ok=True)
    ext = 'gif' if args.format == 'gif' else 'mp4'
    suffix = '_evasive' if args.evasive else ''
    out = os.path.join(vid_dir, f'replay_best{suffix}.{ext}')

    try:
        visualize.create_animation(data_list[idx], config, fps=args.fps,
                                   save_path=out, skip=args.skip)
    except Exception as e:
        print(f"[FAIL] {stage} {tagstr}: animation error: {e}")
        return False

    outcomes = [d['outcome'] for d in data_list]
    n_succ = sum(o == 'success' for o in outcomes)
    print(f"[ok]  {stage:<38s} {tagstr:<22s} "
          f"best={data_list[idx]['outcome']:<8s} "
          f"({n_succ}/{args.episodes} succ) -> {os.path.relpath(out, ROOT)}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/legacy/stage3_stage4.yaml')
    ap.add_argument('--stages', nargs='*', default=None,
                    help='Specific stage names (default: all with a model)')
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--skip', type=int, default=2)
    ap.add_argument('--format', choices=['mp4', 'gif'], default='mp4')
    ap.add_argument('--evasive', action='store_true',
                    help='Override target to reactive high-g evasive mode')
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.config)) as f:
        config = yaml.safe_load(f)

    if args.stages:
        stages = args.stages
    else:
        stages = sorted(
            os.path.basename(os.path.dirname(p))
            for p in glob.glob(os.path.join(ROOT, 'logs/stages/*/models'))
        )
        # Keep only those with an actual checkpoint.
        stages = [s for s in stages
                  if pick_model(os.path.join(ROOT, 'logs/stages', s))]

    print(f"Rendering {len(stages)} stage(s); evasive={args.evasive}, "
          f"{args.episodes} ep each, format={args.format}\n")
    ok = 0
    for s in stages:
        ok += render_stage(s, config, args)
    print(f"\nDone: {ok}/{len(stages)} videos generated.")


if __name__ == '__main__':
    main()
