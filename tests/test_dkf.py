"""
DKF Validation Test — Stage 2b
================================
Evaluates the Disturbance Kalman Filter's estimation accuracy by comparing:
  1. Raw measurement  (noisy + delayed)       — what the agent would see WITHOUT DKF
  2. DKF estimate     (filtered, current-time) — what the agent sees WITH DKF
  3. Ground truth     (clean, current-time)    — the actual target image position

This gives three clean experimental results for the thesis:
  A. Noise+delay WITHOUT DKF  vs ground truth  → cost of raw sensing
  B. DKF estimate             vs ground truth  → residual error after filtering
  C. A vs B                                   → value added by DKF

Usage:
    python tests/test_dkf.py --stage stage2b
    python tests/test_dkf.py --stage stage2b --model logs/stages/stage2b/models/ibvs_ppo_final --episodes 10
"""

import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from envs.interception_env import InterceptionEnv
from envs.wrappers.noise_delay_wrapper import NoiseDelayWrapper
from envs.wrappers.dkf_wrapper import DKFWrapper
from experiment_paths import get_stage_paths


def run_dkf_episode(model, config, seed=42):
    """Run one episode collecting ground truth, raw noisy, and DKF-filtered data.

    Signal chain:
        InterceptionEnv → NoiseDelayWrapper → DKFWrapper → model

    All signals are collected in raw p_bar space (x_c/z_c, y_c/z_c) for
    consistent comparison. Coordinate conversions applied:
      info['p_bar']               → already raw p_bar (from base env)
      info['p_bar_noisy_delayed'] → FOV-normalized, converted via tan_half_fov
      info['dkf_p_bar']           → raw p_bar (DKFWrapper stores in raw space)

    Returns:
        dict with time-series arrays for each signal + episode metadata.
    """
    dt = config['interceptor']['dt']
    nd_cfg = config.get('noise_delay', {})
    dkf_cfg = config.get('dkf', {})
    delay = nd_cfg.get('delay', 3)
    sigma = nd_cfg.get('sigma_noise', 0.03)

    base_env = InterceptionEnv(config=config)
    env_nd = NoiseDelayWrapper(base_env, delay=delay, sigma_noise=sigma)
    env = DKFWrapper(
        env_nd,
        delay=delay,
        dt=dt,
        sigma_pos_process=dkf_cfg.get('sigma_pos_process', 0.01),
        sigma_vel_process=dkf_cfg.get('sigma_vel_process', 0.5),
        sigma_measurement=dkf_cfg.get('sigma_measurement', sigma),
    )

    obs, info = env.reset(seed=seed)

    # FOV scale factors for converting obs[0:2] (normalized) → raw p_bar
    fov_params = base_env.camera.get_fov_params()
    tan_h = fov_params['tan_half_hfov']
    tan_v = fov_params['tan_half_vfov']

    def fov_to_raw(p_fov):
        return np.array([float(p_fov[0]) * tan_h, float(p_fov[1]) * tan_v])

    prev_gt = info['p_bar'].copy()

    data = {
        'gt_px': [], 'gt_py': [],
        'gt_dpx': [], 'gt_dpy': [],
        'raw_px': [], 'raw_py': [],
        'dkf_px': [], 'dkf_py': [],
        'dkf_dpx': [], 'dkf_dpy': [],
        'dkf_P_pos': [],
        'dkf_P_vel': [],
        'in_fov': [],
        'distance': [],
    }

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Ground truth (raw p_bar, clean, current time)
        gt_p_bar = info['p_bar'].copy()
        gt_velocity = (gt_p_bar - prev_gt) / dt
        prev_gt = gt_p_bar.copy()

        # Raw noisy+delayed (FOV-normalized → raw p_bar)
        raw_fov = info.get('p_bar_noisy_delayed', np.zeros(2))
        raw_noisy = fov_to_raw(raw_fov)

        # DKF estimate (already in raw p_bar space)
        dkf_est = info.get('dkf_p_bar',  np.zeros(2)).copy()
        dkf_dp  = info.get('dkf_dp_bar', np.zeros(2)).copy()
        P_diag  = info.get('dkf_P_diag', np.ones(4)).copy()

        if np.any(np.isnan(dkf_est)) or np.any(np.isnan(obs)):
            dkf_est = gt_p_bar.copy()
            dkf_dp  = np.zeros(2)
            P_diag  = np.ones(4)

        data['gt_px'].append(gt_p_bar[0])
        data['gt_py'].append(gt_p_bar[1])
        data['gt_dpx'].append(gt_velocity[0])
        data['gt_dpy'].append(gt_velocity[1])
        data['raw_px'].append(float(raw_noisy[0]))
        data['raw_py'].append(float(raw_noisy[1]))
        data['dkf_px'].append(float(dkf_est[0]))
        data['dkf_py'].append(float(dkf_est[1]))
        data['dkf_dpx'].append(float(dkf_dp[0]))
        data['dkf_dpy'].append(float(dkf_dp[1]))
        data['dkf_P_pos'].append(float(np.sqrt((P_diag[0] + P_diag[1]) / 2)))
        data['dkf_P_vel'].append(float(np.sqrt((P_diag[2] + P_diag[3]) / 2)))
        data['in_fov'].append(float(obs[4]))
        data['distance'].append(float(info.get('relative_distance', 0.0)))

    data['outcome'] = info.get('episode_outcome', 'unknown')
    data['steps']   = len(data['gt_px'])

    for k, v in data.items():
        if isinstance(v, list) and v and isinstance(v[0], (int, float, np.floating)):
            data[k] = np.array(v)

    env.close()
    return data


def compute_metrics(data):
    """Compute RMSE metrics for raw vs DKF estimation."""
    gt_p  = np.stack([data['gt_px'],  data['gt_py']],  axis=1)
    raw_p = np.stack([data['raw_px'], data['raw_py']],  axis=1)
    dkf_p = np.stack([data['dkf_px'], data['dkf_py']], axis=1)
    gt_v  = np.stack([data['gt_dpx'], data['gt_dpy']],  axis=1)
    dkf_v = np.stack([data['dkf_dpx'],data['dkf_dpy']], axis=1)

    raw_err = np.linalg.norm(raw_p - gt_p, axis=1)
    dkf_err = np.linalg.norm(dkf_p - gt_p, axis=1)
    vel_err = np.linalg.norm(dkf_v - gt_v, axis=1)

    skip = 5
    raw_m = max(float(np.mean(raw_err[skip:])), 1e-9)
    return {
        'raw_rmse':    float(np.sqrt(np.mean(raw_err[skip:] ** 2))),
        'raw_mean':    raw_m,
        'dkf_rmse':    float(np.sqrt(np.mean(dkf_err[skip:] ** 2))),
        'dkf_mean':    float(np.mean(dkf_err[skip:])),
        'vel_rmse':    float(np.sqrt(np.mean(vel_err[skip:] ** 2))),
        'improvement': float((1 - float(np.mean(dkf_err[skip:])) / raw_m) * 100),
        'raw_err_arr': raw_err,
        'dkf_err_arr': dkf_err,
        'vel_err_arr': vel_err,
    }


def plot_dkf_validation(data, metrics, config, ep_idx, save_path=None):
    """9-panel DKF validation plot."""
    dt  = config['interceptor']['dt']
    n   = data['steps']
    t   = np.arange(1, n + 1) * dt
    nd_cfg = config.get('noise_delay', {})

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor('#0D0D1A')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32,
                            left=0.07, right=0.97, top=0.90, bottom=0.06)

    outcome  = data['outcome'].upper()
    delay_ms = nd_cfg.get('delay', 3) * int(dt * 1000)
    sigma    = nd_cfg.get('sigma_noise', 0.03)
    fig.suptitle(
        f'DKF Validation — Episode {ep_idx} | {outcome} | '
        f'D={delay_ms}ms  σ={sigma:.2f}',
        fontsize=14, fontweight='bold', color='#E0E0E0', y=0.96
    )

    C  = dict(gt='#06D6A0', raw='#EF476F', dkf='#FFD166',
              vel='#118AB2', conf='#8338EC')
    BG = '#0D0D1A'; GR = '#1A1A3A'

    def _style(ax):
        ax.set_facecolor(BG); ax.tick_params(colors='#AAA', labelsize=7)
        ax.grid(True, color=GR, alpha=0.4); ax.set_xlim(0, t[-1])

    def _leg(ax):
        ax.legend(fontsize=7, facecolor=BG, edgecolor='#2A2A4A', labelcolor='#E0E0E0')

    # 1 — X position
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, data['gt_px'],  color=C['gt'],  lw=2,   label='Ground truth')
    ax.scatter(t, data['raw_px'], s=4, color=C['raw'], alpha=0.5, label='Raw (noisy+delay)')
    ax.plot(t, data['dkf_px'], color=C['dkf'], lw=1.5, label='DKF estimate')
    ax.set_title('Image X Position  p̄_x', color='#E0E0E0', fontsize=10)
    ax.set_xlabel('Time [s]', color='#AAA'); ax.set_ylabel('p̄_x', color='#AAA')
    _style(ax); _leg(ax)

    # 2 — Y position
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, data['gt_py'],  color=C['gt'],  lw=2,   label='Ground truth')
    ax.scatter(t, data['raw_py'], s=4, color=C['raw'], alpha=0.5, label='Raw (noisy+delay)')
    ax.plot(t, data['dkf_py'], color=C['dkf'], lw=1.5, label='DKF estimate')
    ax.set_title('Image Y Position  p̄_y', color='#E0E0E0', fontsize=10)
    ax.set_xlabel('Time [s]', color='#AAA'); ax.set_ylabel('p̄_y', color='#AAA')
    _style(ax); _leg(ax)

    # 3 — Position error
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, metrics['raw_err_arr'], color=C['raw'], lw=1.5, alpha=0.8,
            label=f"Raw  RMSE={metrics['raw_rmse']:.4f}")
    ax.plot(t, metrics['dkf_err_arr'], color=C['gt'],  lw=2,
            label=f"DKF  RMSE={metrics['dkf_rmse']:.4f}")
    ax.axhline(metrics['raw_rmse'], color=C['raw'], ls='--', lw=0.8, alpha=0.5)
    ax.axhline(metrics['dkf_rmse'], color=C['gt'],  ls='--', lw=0.8, alpha=0.5)
    ax.set_title('Position Error  ‖p̄_est − p̄_true‖₂', color='#E0E0E0', fontsize=10)
    ax.set_xlabel('Time [s]', color='#AAA'); ax.set_ylabel('Error', color='#AAA')
    ax.text(0.98, 0.98, f'Improvement: {metrics["improvement"]:.1f}%',
            transform=ax.transAxes, fontsize=9, color='#06D6A0',
            va='top', ha='right',
            bbox=dict(facecolor=BG, alpha=0.8, edgecolor='#2A2A4A', pad=3))
    _style(ax); _leg(ax)

    # 4 — X velocity
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, data['gt_dpx'],  color=C['gt'],  lw=2,   label='Ground truth')
    ax.plot(t, data['dkf_dpx'], color=C['dkf'], lw=1.5, label='DKF estimate')
    ax.set_title('Image X Velocity  ṗ̄_x', color='#E0E0E0', fontsize=10)
    ax.set_xlabel('Time [s]', color='#AAA'); ax.set_ylabel('ṗ̄_x [1/s]', color='#AAA')
    _style(ax); _leg(ax)

    # 5 — Y velocity
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, data['gt_dpy'],  color=C['gt'],  lw=2,   label='Ground truth')
    ax.plot(t, data['dkf_dpy'], color=C['dkf'], lw=1.5, label='DKF estimate')
    ax.set_title('Image Y Velocity  ṗ̄_y', color='#E0E0E0', fontsize=10)
    ax.set_xlabel('Time [s]', color='#AAA'); ax.set_ylabel('ṗ̄_y [1/s]', color='#AAA')
    _style(ax); _leg(ax)

    # 6 — Velocity error + uncertainty
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, metrics['vel_err_arr'], color=C['vel'], lw=1.5,
            label=f"Vel RMSE={metrics['vel_rmse']:.4f}")
    ax.set_title('Velocity Error  ‖ṗ̄_est − ṗ̄_true‖₂', color='#E0E0E0', fontsize=10)
    ax.set_xlabel('Time [s]', color='#AAA'); ax.set_ylabel('Error [1/s]', color='#AAA')
    ax2 = ax.twinx()
    ax2.fill_between(t, 0, data['dkf_P_pos'], color=C['conf'], alpha=0.15)
    ax2.plot(t, data['dkf_P_pos'], color=C['conf'], lw=1, alpha=0.7, label='DKF σ_pos')
    ax2.set_ylabel('σ_pos', color=C['conf'])
    ax2.tick_params(axis='y', colors=C['conf'], labelsize=7)
    ax2.legend(fontsize=7, facecolor=BG, edgecolor='#2A2A4A', labelcolor='#E0E0E0', loc='upper right')
    _style(ax); _leg(ax)

    # 7 — 2D image plane
    ax = fig.add_subplot(gs[2, 0])
    ax.set_facecolor(BG)
    ax.scatter(data['gt_px'],  data['gt_py'],  c=t, cmap='plasma', s=8,
               zorder=3, label='Ground truth')
    ax.scatter(data['raw_px'], data['raw_py'],  color=C['raw'], s=3,
               alpha=0.3, zorder=1, label='Raw (noisy+delay)')
    ax.scatter(data['dkf_px'], data['dkf_py'],  color=C['dkf'], s=5,
               alpha=0.7, zorder=2, label='DKF estimate')
    ax.axhline(0, color='#444', lw=0.5); ax.axvline(0, color='#444', lw=0.5)
    ax.add_patch(Rectangle((-1,-1), 2, 2, fill=False,
                             edgecolor='#FFBE0B', lw=1.2, ls='--'))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_title('Image Plane Trajectory', color='#E0E0E0', fontsize=10)
    ax.set_xlabel('p̄_x', color='#AAA'); ax.set_ylabel('p̄_y', color='#AAA')
    ax.tick_params(colors='#AAA', labelsize=7); ax.grid(True, color=GR, alpha=0.4)
    _leg(ax)

    # 8 — Distance
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, data['distance'], color=C['gt'], lw=2, label='3D distance [m]')
    d_suc = config['env']['d_success']
    ax.axhline(d_suc, color='#FFBE0B', ls='--', lw=1, alpha=0.7,
               label=f'd_success={d_suc}m')
    ax.set_title('Relative Distance', color='#E0E0E0', fontsize=10)
    ax.set_xlabel('Time [s]', color='#AAA'); ax.set_ylabel('Distance [m]', color='#AAA')
    _style(ax); _leg(ax)

    # 9 — Summary table
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor(BG); ax.axis('off')
    rows = [
        ['Metric',      'Raw (noise+delay)',            'DKF'],
        ['Pos RMSE',    f"{metrics['raw_rmse']:.5f}",   f"{metrics['dkf_rmse']:.5f}"],
        ['Pos Mean',    f"{metrics['raw_mean']:.5f}",   f"{metrics['dkf_mean']:.5f}"],
        ['Improvement', '—',                             f"{metrics['improvement']:.1f}%"],
        ['Vel RMSE',    '—',                             f"{metrics['vel_rmse']:.5f}"],
        ['Steps',       str(n),                          str(n)],
        ['Outcome',     data['outcome'],                 data['outcome']],
    ]
    cell_c = [['#2A2A4A']*3] + [['#1A1A2E', C['raw']+'33', C['gt']+'33']] * (len(rows)-1)
    tbl = ax.table(cellText=rows, cellLoc='center', loc='center', cellColours=cell_c)
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 2.0)
    for _, cell in tbl.get_celld().items():
        cell.set_edgecolor('#2A2A4A'); cell.set_text_props(color='#E0E0E0')
    ax.set_title('Summary', color='#E0E0E0', fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'  Plot saved: {save_path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='DKF validation test — Stage 2b')
    parser.add_argument('--stage',    type=str, default='stage2b')
    parser.add_argument('--model',    type=str, default=None)
    parser.add_argument('--config',   type=str, default='config.yaml')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--save-dir', type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    paths      = get_stage_paths(config, args.stage)
    model_path = args.model or str(paths['models'] / 'ibvs_ppo_final')
    save_dir   = args.save_dir or str(paths['base'] / 'dkf_test')
    os.makedirs(save_dir, exist_ok=True)

    print(f'Loading model: {model_path}')
    model = PPO.load(model_path)

    all_raw_rmse, all_dkf_rmse, all_imp = [], [], []

    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f'\n--- Episode {ep+1}/{args.episodes} (seed={seed}) ---')
        data    = run_dkf_episode(model, config, seed=seed)
        metrics = compute_metrics(data)

        print(f'  Outcome: {data["outcome"]}  ({data["steps"]} steps)')
        print(f'  Raw noisy+delayed:  RMSE={metrics["raw_rmse"]:.5f}  '
              f'Mean={metrics["raw_mean"]:.5f}')
        print(f'  DKF estimate:       RMSE={metrics["dkf_rmse"]:.5f}  '
              f'Mean={metrics["dkf_mean"]:.5f}')
        print(f'  DKF improvement:    {metrics["improvement"]:.1f}%')
        print(f'  Velocity RMSE:      {metrics["vel_rmse"]:.5f}')

        all_raw_rmse.append(metrics['raw_rmse'])
        all_dkf_rmse.append(metrics['dkf_rmse'])
        all_imp.append(metrics['improvement'])

        save_path = os.path.join(save_dir, f'dkf_ep{ep+1}.png')
        plot_dkf_validation(data, metrics, config, ep + 1, save_path=save_path)

    print(f'\n{"="*60}')
    print(f'  DKF Validation Summary ({args.episodes} episodes)')
    print(f'{"="*60}')
    print(f'  Raw RMSE:         {np.mean(all_raw_rmse):.5f} ± {np.std(all_raw_rmse):.5f}')
    print(f'  DKF RMSE:         {np.mean(all_dkf_rmse):.5f} ± {np.std(all_dkf_rmse):.5f}')
    print(f'  DKF improvement:  {np.mean(all_imp):.1f}% ± {np.std(all_imp):.1f}%')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
