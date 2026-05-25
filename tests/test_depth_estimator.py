"""
Depth Estimator Validation Test
=================================
Runs the trained agent for several episodes while simultaneously running
the Jacobian-based depth estimator. Compares the estimated depth to the
ground-truth depth at every timestep.

Usage:
    python tests/test_depth_estimator.py --model logs/models/ibvs_ppo_final
    python tests/test_depth_estimator.py --model logs/models/ibvs_ppo_final --episodes 5

Generates:
    - Time-series plot comparing estimated vs. ground-truth depth
    - Estimation error statistics
    - Confidence / observability metric over time
"""

import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from envs.interception_env import InterceptionEnv
from observers.interaction_matrix import InteractionMatrix
from observers.depth_estimator import DepthEstimator
from experiment_paths import (
    default_model_path,
    ensure_stage_dirs,
    get_stage_paths,
)


def run_episode_with_depth_estimation(model, env, estimator, config,
                                       deterministic=True, seed=42):
    """Run one episode collecting both ground-truth and estimated depth.

    Args:
        model:       Trained SB3 model.
        env:         InterceptionEnv instance.
        estimator:   DepthEstimator instance.
        config:      Parsed config dict.
        deterministic: Use deterministic policy.
        seed:        Random seed.

    Returns:
        dict with ground-truth and estimated data arrays.
    """
    obs, info = env.reset(seed=seed)

    # Get camera model parameters
    R_c_b = env.camera.R_c_b

    # Reset estimator with a rough initial guess
    # Initial distance is unknown — guess 20m (typical mid-range)
    estimator.reset(rho_init=0.05, P_init=1.0)

    dt = config['interceptor']['dt']

    # Storage
    data = {
        'z_true': [],          # Ground-truth depth (z_c from camera model)
        'z_est': [],           # Estimated depth from Jacobian filter
        'rho_true': [],        # Ground-truth inverse depth
        'rho_est': [],         # Estimated inverse depth
        'rho_meas': [],        # Instantaneous Jacobian measurement
        'confidence': [],      # Observability metric
        'P': [],               # Estimation variance
        'd_rel': [],           # 3D relative distance (for reference)
        'image_error': [],     # Image error
        'p_bar': [],           # Image coordinates
        'v_cam_norm': [],      # Camera velocity magnitude (for observability)
        'updated': [],         # Whether measurement was used
    }

    # We need the previous p_bar for finite-difference velocity
    prev_p_bar = info['p_bar'].copy()

    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        # --- Compute ground-truth depth ---
        # p_r = interceptor - target (in EFCS)
        p_r = env.interceptor.position - env.target.position
        R_be = env.interceptor.get_rotation_matrix()

        # Target direction in EFCS → camera frame
        d_efcs = -p_r  # direction toward target
        R_e_b = R_be.T
        R_e_c = R_c_b @ R_e_b
        p_camera = R_e_c @ d_efcs
        z_true = p_camera[2]  # depth along optical axis

        # --- Compute camera velocity in camera frame ---
        # Angular velocity: in Stage 1, only yaw rate is nonzero
        # The yaw rate was the last action component
        yaw_rate = float(action[3]) * env.yaw_rate_max
        omega_body = np.array([0.0, 0.0, yaw_rate])  # [p, q, r] in body frame

        v_cam, omega_cam = InteractionMatrix.compute_camera_velocity(
            v_interceptor_efcs=env.interceptor.velocity,
            omega_body=omega_body,
            R_b_e=R_be,
            R_c_b=R_c_b,
        )

        # --- Compute image velocity (finite differences) ---
        current_p_bar = info['p_bar'].copy()
        p_bar_dot = (current_p_bar - prev_p_bar) / dt

        # --- Run depth estimator ---
        est_result = estimator.update(
            p_bar=current_p_bar,
            p_bar_dot=p_bar_dot,
            v_cam=v_cam,
            omega_cam=omega_cam,
            v_target_cam=None,  # Assume static target (worst case)
        )

        # --- Record ---
        data['z_true'].append(z_true)
        data['z_est'].append(est_result['z_hat'])
        data['rho_true'].append(1.0 / max(z_true, 1e-9) if z_true > 0 else 0.0)
        data['rho_est'].append(est_result['rho_hat'])
        data['rho_meas'].append(est_result['rho_meas'])
        data['confidence'].append(est_result['confidence'])
        data['P'].append(est_result['P'])
        data['d_rel'].append(info['relative_distance'])
        data['image_error'].append(info['image_error'])
        data['p_bar'].append(current_p_bar.copy())
        data['v_cam_norm'].append(np.linalg.norm(v_cam))
        data['updated'].append(est_result['updated'])

        prev_p_bar = current_p_bar.copy()

    data['outcome'] = info['episode_outcome']
    data['steps'] = step

    # Convert to numpy arrays
    for key in data:
        if isinstance(data[key], list) and len(data[key]) > 0:
            if isinstance(data[key][0], (int, float, bool, np.floating)):
                data[key] = np.array(data[key])
            elif isinstance(data[key][0], np.ndarray):
                data[key] = np.array(data[key])

    return data


def plot_depth_estimation(data, config, save_path=None):
    """Plot depth estimation results for one episode.

    Generates a 4-panel figure:
        (a) Depth: ground truth vs estimate
        (b) Inverse depth: ρ_true vs ρ̂ vs ρ_meas
        (c) Estimation error (absolute and relative)
        (d) Confidence / observability and camera speed

    Args:
        data:      Episode data dict.
        config:    Parsed config dict.
        save_path: If set, save figure to this path.
    """
    dt = config['interceptor']['dt']
    n_steps = data['steps']
    t = np.arange(1, n_steps + 1) * dt

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#1A1A2E')

    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.30,
                  left=0.07, right=0.96, top=0.92, bottom=0.07)

    outcome_str = data['outcome'].upper()
    fig.suptitle(
        f'Jacobian-Based Depth Estimation — {outcome_str} ({n_steps} steps)',
        fontsize=16, fontweight='bold', color='#E0E0E0', y=0.97
    )

    # === Panel 1: Depth comparison ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1A1A2E')
    ax1.plot(t, data['z_true'], color='#06D6A0', lw=2, label='Ground Truth $z_c$')
    ax1.plot(t, data['z_est'], color='#FF6B6B', lw=1.5, alpha=0.9,
             label='Estimated $\\hat{z}_c$')
    ax1.plot(t, data['d_rel'], color='#00B4D8', lw=1, alpha=0.5,
             ls='--', label='3D Distance $d_{rel}$')
    ax1.set_xlabel('Time [s]', color='#E0E0E0')
    ax1.set_ylabel('Depth [m]', color='#E0E0E0')
    ax1.set_title('Depth: Ground Truth vs Estimate', color='#E0E0E0',
                   fontsize=11, pad=8)
    ax1.legend(fontsize=8, facecolor='#1A1A2E', edgecolor='#2A2A4A',
               labelcolor='#E0E0E0')
    ax1.tick_params(colors='#E0E0E0', labelsize=7)
    ax1.grid(True, color='#2A2A4A', alpha=0.3)
    ax1.set_xlim(0, t[-1])
    ax1.set_ylim(0, max(max(data['z_true']), max(data['z_est'])) * 1.15)

    # === Panel 2: Inverse depth ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1A1A2E')
    ax2.plot(t, data['rho_true'], color='#06D6A0', lw=2, label='ρ true (1/$z_c$)')
    ax2.plot(t, data['rho_est'], color='#FF6B6B', lw=1.5, alpha=0.9,
             label='ρ̂ estimated')
    # Plot instantaneous measurements (scatter, transparent)
    valid = data['rho_meas'] > 0
    ax2.scatter(t[valid], data['rho_meas'][valid], color='#FFBE0B', s=5,
                alpha=0.3, zorder=1, label='ρ instant (Jacobian)')
    ax2.set_xlabel('Time [s]', color='#E0E0E0')
    ax2.set_ylabel('Inverse Depth ρ [1/m]', color='#E0E0E0')
    ax2.set_title('Inverse Depth: Kalman Filtered vs Instantaneous',
                   color='#E0E0E0', fontsize=11, pad=8)
    ax2.legend(fontsize=8, facecolor='#1A1A2E', edgecolor='#2A2A4A',
               labelcolor='#E0E0E0')
    ax2.tick_params(colors='#E0E0E0', labelsize=7)
    ax2.grid(True, color='#2A2A4A', alpha=0.3)
    ax2.set_xlim(0, t[-1])

    # === Panel 3: Estimation error ===
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#1A1A2E')

    # Absolute error
    z_err = np.abs(data['z_est'] - data['z_true'])
    ax3.plot(t, z_err, color='#FF6B6B', lw=1.5, label='|$\\hat{z}_c - z_c$| [m]')
    ax3.set_xlabel('Time [s]', color='#E0E0E0')
    ax3.set_ylabel('Absolute Error [m]', color='#FF6B6B')
    ax3.set_title('Depth Estimation Error', color='#E0E0E0', fontsize=11, pad=8)
    ax3.tick_params(colors='#E0E0E0', labelsize=7)
    ax3.grid(True, color='#2A2A4A', alpha=0.3)
    ax3.set_xlim(0, t[-1])

    # Relative error on twin axis
    ax3r = ax3.twinx()
    safe_z = np.maximum(data['z_true'], 0.1)
    rel_err = z_err / safe_z * 100
    ax3r.plot(t, rel_err, color='#FFBE0B', lw=1, alpha=0.7,
              label='Relative Error [%]')
    ax3r.set_ylabel('Relative Error [%]', color='#FFBE0B')
    ax3r.tick_params(axis='y', colors='#FFBE0B', labelsize=7)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               facecolor='#1A1A2E', edgecolor='#2A2A4A', labelcolor='#E0E0E0')

    # Summary statistics text
    mean_err = np.mean(z_err)
    median_err = np.median(z_err)
    # Skip first 10% for "converged" stats
    converged_start = max(1, n_steps // 10)
    mean_err_conv = np.mean(z_err[converged_start:])
    mean_rel_conv = np.mean(rel_err[converged_start:])
    stats_text = (
        f"Mean |err|: {mean_err:.2f}m (all), {mean_err_conv:.2f}m (conv.)\n"
        f"Median |err|: {median_err:.2f}m\n"
        f"Mean rel err (conv.): {mean_rel_conv:.1f}%"
    )
    ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes,
             fontsize=8, color='#E0E0E0', va='top', ha='right',
             fontfamily='monospace',
             bbox=dict(facecolor='#1A1A2E', alpha=0.8, edgecolor='#2A2A4A', pad=4))

    # === Panel 4: Confidence & Camera Speed ===
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#1A1A2E')

    ax4.plot(t, data['confidence'], color='#8338EC', lw=1.5,
             label='Confidence ||b||²')
    ax4.axhline(0.01, color='#EF476F', ls='--', lw=1, alpha=0.6,
                label='Confidence threshold')
    ax4.set_xlabel('Time [s]', color='#E0E0E0')
    ax4.set_ylabel('Confidence', color='#8338EC')
    ax4.set_title('Observability & Camera Motion', color='#E0E0E0',
                   fontsize=11, pad=8)
    ax4.tick_params(colors='#E0E0E0', labelsize=7)
    ax4.grid(True, color='#2A2A4A', alpha=0.3)
    ax4.set_xlim(0, t[-1])

    # Camera speed on twin axis
    ax4r = ax4.twinx()
    ax4r.plot(t, data['v_cam_norm'], color='#00B4D8', lw=1, alpha=0.7,
              label='Camera speed [m/s]')
    ax4r.set_ylabel('Camera Speed [m/s]', color='#00B4D8')
    ax4r.tick_params(axis='y', colors='#00B4D8', labelsize=7)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4r.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
               facecolor='#1A1A2E', edgecolor='#2A2A4A', labelcolor='#E0E0E0',
               loc='upper right')

    # Mark low-confidence regions
    low_conf = data['confidence'] < 0.01
    if np.any(low_conf):
        ax4.fill_between(t, 0, ax4.get_ylim()[1],
                          where=low_conf, alpha=0.1, color='#EF476F',
                          label='Low observability')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Plot saved to: {save_path}")

    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Validate Jacobian-based depth estimator against ground truth'
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML')
    parser.add_argument('--stage', type=str, default=None,
                        help='Experiment stage name for outputs and default model path')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to test')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    paths = get_stage_paths(config, args.stage)
    model_path = args.model or default_model_path(paths)
    save_dir = args.save_dir or str(paths['depth_test'])

    # Load model
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    # Create environment
    env = InterceptionEnv(config=config)

    # Create depth estimator
    estimator = DepthEstimator(
        rho_init=0.05,   # Initial guess: ~20m
        P_init=1.0,      # High initial uncertainty
        Q=0.001,         # Slow depth change (process noise)
        R_base=0.1,      # Base measurement noise
    )

    if args.save_dir is None:
        ensure_stage_dirs(paths, 'depth_test')
    else:
        os.makedirs(save_dir, exist_ok=True)

    # Run episodes
    all_errors = []
    all_rel_errors = []

    for ep in range(args.episodes):
        seed = args.seed + ep
        print(f"\n--- Episode {ep + 1}/{args.episodes} (seed={seed}) ---")

        data = run_episode_with_depth_estimation(
            model, env, estimator, config,
            deterministic=True, seed=seed
        )

        # Compute summary stats
        z_err = np.abs(data['z_est'] - data['z_true'])
        safe_z = np.maximum(data['z_true'], 0.1)
        rel_err = z_err / safe_z * 100

        # Skip first 10% (convergence transient)
        conv_start = max(1, data['steps'] // 10)

        print(f"  Outcome: {data['outcome']}")
        print(f"  Steps: {data['steps']}")
        print(f"  Depth error (all):      mean={np.mean(z_err):.2f}m, "
              f"median={np.median(z_err):.2f}m")
        print(f"  Depth error (converged): mean={np.mean(z_err[conv_start:]):.2f}m, "
              f"median={np.median(z_err[conv_start:]):.2f}m")
        print(f"  Relative error (conv.):  mean={np.mean(rel_err[conv_start:]):.1f}%")
        print(f"  Confidence (mean):       {np.mean(data['confidence']):.4f}")
        print(f"  Measurements used:       {np.sum(data['updated'])}/{data['steps']}")

        all_errors.append(np.mean(z_err[conv_start:]))
        all_rel_errors.append(np.mean(rel_err[conv_start:]))

        # Plot
        save_path = os.path.join(save_dir, f'depth_ep{ep + 1}.png')
        plot_depth_estimation(data, config, save_path=save_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Depth Estimation Summary ({args.episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Mean depth error (converged): {np.mean(all_errors):.2f} ± "
          f"{np.std(all_errors):.2f} m")
    print(f"  Mean rel. error (converged):  {np.mean(all_rel_errors):.1f} ± "
          f"{np.std(all_rel_errors):.1f} %")
    print(f"{'='*60}")

    env.close()


if __name__ == '__main__':
    main()
