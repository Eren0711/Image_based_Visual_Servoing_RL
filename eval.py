"""
Evaluation & Visualization Script — Stage 1
=============================================
Evaluates a trained PPO agent on the IBVS Drone Interception environment
and generates multi-panel diagnostic visualizations.

Usage:
    python eval.py --model logs/models/ibvs_ppo_final
    python eval.py --model logs/models/ibvs_ppo_final --episodes 50
    python eval.py --model logs/models/ibvs_ppo_final --deterministic

Visualizations generated:
    1. 3D trajectory plot (interceptor + target paths)
    2. Image-plane error over time with FOV boundary
    3. Relative distance over time
    4. Reward components breakdown
    5. Action history (velocity commands + yaw rate)
"""

import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from stable_baselines3 import PPO
from envs.interception_env import InterceptionEnv
from experiment_paths import (
    default_model_path,
    ensure_stage_dirs,
    get_stage_paths,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_episode(model, env, deterministic: bool = True,
                     seed: int = None) -> dict:
    """Run a single evaluation episode and collect trajectory data.

    Args:
        model: Trained SB3 model.
        env: InterceptionEnv instance.
        deterministic: If True, use deterministic actions.

    Returns:
        dict with trajectory data for visualization.
    """
    obs, info = env.reset(seed=seed)

    data = {
        'interceptor_pos': [info['interceptor_pos'].copy()],
        'target_pos': [info['target_pos'].copy()],
        'p_bar': [info['p_bar'].copy()],
        'image_error': [info['image_error']],
        'relative_distance': [info['relative_distance']],
        'in_fov': [info['in_fov']],
        'fov_margin': [info['fov_margin']],
        'actions': [],
        'rewards': [],
        'outcome': 'running',
        'cbf_corrections': 0,
        'cbf_correction_norms': [],
    }

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        data['interceptor_pos'].append(info['interceptor_pos'].copy())
        data['target_pos'].append(info['target_pos'].copy())
        data['p_bar'].append(info['p_bar'].copy())
        data['image_error'].append(info['image_error'])
        data['relative_distance'].append(info['relative_distance'])
        data['in_fov'].append(info['in_fov'])
        data['fov_margin'].append(info['fov_margin'])
        data['actions'].append(action.copy())
        data['rewards'].append(reward)
        if 'cbf' in info:
            if info['cbf']['corrected']:
                data['cbf_corrections'] += 1
            data['cbf_correction_norms'].append(info['cbf']['correction_norm'])

    data['outcome'] = info['episode_outcome']
    if 'cbf_episode' in info:
        data['cbf_episode'] = info['cbf_episode']

    # Convert lists to arrays
    for key in ['interceptor_pos', 'target_pos', 'p_bar', 'actions']:
        data[key] = np.array(data[key])
    for key in ['image_error', 'relative_distance', 'rewards',
                'in_fov', 'fov_margin']:
        data[key] = np.array(data[key])

    return data


def plot_single_episode(data: dict, config: dict, save_path: str = None):
    """Generate a comprehensive multi-panel visualization of one episode.

    Panels:
        (a) 3D trajectory
        (b) Image-plane error trajectory
        (c) Relative distance over time
        (d) Reward over time
        (e) Actions over time

    Args:
        data: Episode data dictionary from evaluate_episode().
        config: Configuration dictionary.
        save_path: If given, save the figure to this path.
    """
    dt = config['interceptor']['dt']
    n_steps = len(data['rewards'])
    time = np.arange(n_steps + 1) * dt  # +1 because we have initial state

    # Color palette
    c_interceptor = '#2196F3'  # Blue
    c_target = '#F44336'       # Red
    c_reward = '#4CAF50'       # Green
    c_fov = '#FF9800'          # Orange
    c_action = ['#9C27B0', '#00BCD4', '#FFEB3B', '#E91E63']

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f'IBVS Drone Interception — Episode Outcome: {data["outcome"].upper()}',
        fontsize=16, fontweight='bold', y=0.98
    )

    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ============================================================
    # (a) 3D Trajectory Plot
    # ============================================================
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')

    int_pos = data['interceptor_pos']
    tgt_pos = data['target_pos']

    # Plot in z-up for visualization (NED → flip z)
    ax3d.plot(int_pos[:, 0], int_pos[:, 1], -int_pos[:, 2],
              color=c_interceptor, linewidth=2, label='Interceptor')
    ax3d.plot(tgt_pos[:, 0], tgt_pos[:, 1], -tgt_pos[:, 2],
              color=c_target, linewidth=2, label='Target', linestyle='--')

    # Start and end markers
    ax3d.scatter(*int_pos[0, :2], -int_pos[0, 2],
                 color=c_interceptor, s=100, marker='o', zorder=5,
                 label='Interceptor Start')
    ax3d.scatter(*int_pos[-1, :2], -int_pos[-1, 2],
                 color=c_interceptor, s=100, marker='*', zorder=5)
    ax3d.scatter(*tgt_pos[0, :2], -tgt_pos[0, 2],
                 color=c_target, s=100, marker='o', zorder=5,
                 label='Target Start')
    ax3d.scatter(*tgt_pos[-1, :2], -tgt_pos[-1, 2],
                 color=c_target, s=100, marker='*', zorder=5)

    ax3d.set_xlabel('X (North) [m]')
    ax3d.set_ylabel('Y (East) [m]')
    ax3d.set_zlabel('Z (Up) [m]')
    ax3d.set_title('3D Trajectory')
    ax3d.legend(fontsize=8, loc='upper left')

    # ============================================================
    # (b) Image-Plane Error Trajectory
    # ============================================================
    ax_img = fig.add_subplot(gs[0, 1])

    p_bar = data['p_bar']
    cam_params = PinholeCamera(config['camera']).get_fov_params()
    tan_h = cam_params['tan_half_hfov']
    tan_v = cam_params['tan_half_vfov']

    # Normalize p_bar to FOV units
    p_bar_norm_x = p_bar[:, 0] / tan_h
    p_bar_norm_y = p_bar[:, 1] / tan_v

    # Plot trajectory on image plane
    scatter = ax_img.scatter(
        p_bar_norm_x, p_bar_norm_y,
        c=np.arange(len(p_bar_norm_x)), cmap='viridis',
        s=10, alpha=0.7, zorder=3
    )
    ax_img.plot(p_bar_norm_x, p_bar_norm_y,
                color='gray', alpha=0.3, linewidth=0.5, zorder=2)

    # FOV boundary rectangle
    fov_rect = plt.Rectangle((-1, -1), 2, 2, linewidth=2,
                              edgecolor=c_fov, facecolor='none',
                              linestyle='--', label='FOV Boundary')
    ax_img.add_patch(fov_rect)

    # Image center crosshair
    ax_img.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax_img.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax_img.plot(0, 0, '+', color='red', markersize=15, markeredgewidth=2)

    # Start and end markers
    ax_img.plot(p_bar_norm_x[0], p_bar_norm_y[0], 'go',
                markersize=10, label='Start', zorder=5)
    ax_img.plot(p_bar_norm_x[-1], p_bar_norm_y[-1], 'r*',
                markersize=12, label='End', zorder=5)

    ax_img.set_xlim(-1.5, 1.5)
    ax_img.set_ylim(-1.5, 1.5)
    ax_img.set_xlabel('Normalized Image X (±1 = FOV edge)')
    ax_img.set_ylabel('Normalized Image Y (±1 = FOV edge)')
    ax_img.set_title('Target Position on Image Plane')
    ax_img.set_aspect('equal')
    ax_img.legend(fontsize=8)
    plt.colorbar(scatter, ax=ax_img, label='Timestep')

    # ============================================================
    # (c) Relative Distance Over Time
    # ============================================================
    ax_dist = fig.add_subplot(gs[1, 0])

    ax_dist.plot(time, data['relative_distance'],
                 color=c_interceptor, linewidth=2)
    ax_dist.axhline(config['env']['d_success'], color=c_target,
                     linestyle='--', linewidth=1, alpha=0.7,
                     label=f"Success threshold ({config['env']['d_success']} m)")
    ax_dist.fill_between(time, 0, data['relative_distance'],
                          alpha=0.1, color=c_interceptor)

    # Mark FOV losses
    fov_lost = ~data['in_fov']
    if np.any(fov_lost):
        lost_times = time[fov_lost]
        ax_dist.scatter(lost_times,
                        data['relative_distance'][fov_lost],
                        color=c_fov, s=20, alpha=0.7, label='FOV Lost',
                        zorder=4)

    ax_dist.set_xlabel('Time [s]')
    ax_dist.set_ylabel('Relative Distance [m]')
    ax_dist.set_title('Relative Distance Over Time')
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3)

    # ============================================================
    # (d) Image Error Over Time
    # ============================================================
    ax_err = fig.add_subplot(gs[1, 1])

    ax_err.plot(time, data['image_error'],
                color='#9C27B0', linewidth=2, label='||p̄|| image error')
    ax_err.fill_between(time, 0, data['image_error'],
                         alpha=0.1, color='#9C27B0')

    ax_err.set_xlabel('Time [s]')
    ax_err.set_ylabel('Image Error ||p̄||')
    ax_err.set_title('Image-Plane Error Over Time')
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)

    # ============================================================
    # (e) Actions Over Time
    # ============================================================
    ax_act = fig.add_subplot(gs[2, 0])

    actions = data['actions']
    time_act = np.arange(len(actions)) * dt
    labels_act = ['v_x (forward)', 'v_y (lateral)', 'v_z (vertical)',
                  'ψ̇ (yaw rate)']

    for i in range(4):
        ax_act.plot(time_act, actions[:, i],
                    color=c_action[i], linewidth=1.5,
                    label=labels_act[i], alpha=0.8)

    ax_act.set_xlabel('Time [s]')
    ax_act.set_ylabel('Action (normalized [-1, 1])')
    ax_act.set_title('Agent Actions Over Time')
    ax_act.legend(fontsize=8, ncol=2)
    ax_act.grid(True, alpha=0.3)
    ax_act.set_ylim(-1.2, 1.2)

    # ============================================================
    # (f) Cumulative Reward
    # ============================================================
    ax_rew = fig.add_subplot(gs[2, 1])

    cum_reward = np.cumsum(data['rewards'])
    ax_rew.plot(time[1:], cum_reward, color=c_reward, linewidth=2)
    ax_rew.fill_between(time[1:], 0, cum_reward, alpha=0.1, color=c_reward)

    ax_rew.set_xlabel('Time [s]')
    ax_rew.set_ylabel('Cumulative Reward')
    ax_rew.set_title('Cumulative Reward Over Time')
    ax_rew.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")

    plt.show()


def print_summary(all_data: list):
    """Print aggregate evaluation statistics.

    Args:
        all_data: List of episode data dictionaries.
    """
    n = len(all_data)
    outcomes = [d['outcome'] for d in all_data]
    n_success = sum(1 for o in outcomes if o == 'success')
    n_fov = sum(1 for o in outcomes if o == 'fov_loss')
    n_timeout = sum(1 for o in outcomes if o == 'timeout')

    final_dists = [d['relative_distance'][-1] for d in all_data]
    final_errors = [d['image_error'][-1] for d in all_data]
    mean_errors = [np.mean(d['image_error']) for d in all_data]
    episode_lengths = [len(d['rewards']) for d in all_data]
    total_rewards = [np.sum(d['rewards']) for d in all_data]

    print(f"\n{'='*60}")
    print(f"  Evaluation Summary — {n} Episodes")
    print(f"{'='*60}")
    print(f"  Success rate     : {100*n_success/n:.1f}% ({n_success}/{n})")
    print(f"  FOV loss rate    : {100*n_fov/n:.1f}% ({n_fov}/{n})")
    print(f"  Timeout rate     : {100*n_timeout/n:.1f}% ({n_timeout}/{n})")
    print(f"  ---")
    print(f"  Final distance   : {np.mean(final_dists):.2f} ± "
          f"{np.std(final_dists):.2f} m")
    print(f"  Final img error  : {np.mean(final_errors):.4f} ± "
          f"{np.std(final_errors):.4f}")
    print(f"  Mean img error   : {np.mean(mean_errors):.4f} ± "
          f"{np.std(mean_errors):.4f}")
    print(f"  Episode length   : {np.mean(episode_lengths):.0f} ± "
          f"{np.std(episode_lengths):.0f} steps")
    print(f"  Total reward     : {np.mean(total_rewards):.2f} ± "
          f"{np.std(total_rewards):.2f}")
    print(f"{'='*60}")

    # CBF stats (if present)
    cbf_data = [d for d in all_data if 'cbf_episode' in d]
    if cbf_data:
        total_steps = sum(d['cbf_episode']['n_steps'] for d in cbf_data)
        total_corr = sum(d['cbf_episode']['n_corrections'] for d in cbf_data)
        total_infeas = sum(d['cbf_episode']['n_infeasible'] for d in cbf_data)
        avg_corr_norm = np.mean([
            d['cbf_episode']['avg_correction_norm']
            for d in cbf_data
            if d['cbf_episode']['n_corrections'] > 0
        ]) if any(d['cbf_episode']['n_corrections'] > 0 for d in cbf_data) else 0.0
        viol_counts = np.sum([
            d['cbf_episode']['violations_per_constraint']
            for d in cbf_data
        ], axis=0)
        print(f"\n  CBF safety filter:")
        print(f"    Correction rate  : {100*total_corr/total_steps:.1f}% "
              f"({total_corr}/{total_steps} steps)")
        print(f"    Avg correction norm: {avg_corr_norm:.3f}")
        print(f"    Infeasible solves: {total_infeas}")
        print(f"    Violations per constraint "
              f"(hfov,vfov,pitch,roll): {viol_counts.tolist()}")

    # Per-outcome stats
    if n_success > 0:
        success_dists = [d['relative_distance'][-1]
                         for d in all_data if d['outcome'] == 'success']
        success_lengths = [len(d['rewards'])
                           for d in all_data if d['outcome'] == 'success']
        print(f"\n  Successful interceptions:")
        print(f"    Final distance : {np.mean(success_dists):.3f} m")
        print(f"    Avg steps      : {np.mean(success_lengths):.0f}")


# Import here to avoid circular imports at module level
from models.camera_model import PinholeCamera


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained PPO agent for IBVS drone interception'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to saved model (without .zip extension)'
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--stage', type=str, default=None,
        help='Experiment stage name for outputs and default model path'
    )
    parser.add_argument(
        '--episodes', type=int, default=None,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--deterministic', action='store_true', default=None,
        help='Use deterministic actions'
    )
    parser.add_argument(
        '--plot-episode', type=int, default=0,
        help='Index of episode to generate detailed plots for (-1 = best)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for evaluation'
    )
    parser.add_argument(
        '--cbf', action='store_true',
        help='Wrap env with CBFWrapper (Stage 4a safety filter)'
    )
    parser.add_argument(
        '--cbf-alpha-fov', type=float, default=0.8,
        help='CBF margin for FOV constraints (1=tight, ~0.8 default)'
    )
    parser.add_argument(
        '--cbf-alpha-att', type=float, default=0.3,
        help='CBF margin for attitude constraints (0.3 default)'
    )
    parser.add_argument(
        '--cbf-horizon-fov', type=int, default=3,
        help='Prediction horizon (steps) for FOV constraints'
    )
    parser.add_argument(
        '--cbf-horizon-att', type=int, default=15,
        help='Prediction horizon (steps) for attitude constraints'
    )
    args = parser.parse_args()

    # --- Load config ---
    config = load_config(args.config)
    eval_cfg = config.get('evaluation', {})
    paths = get_stage_paths(config, args.stage)

    n_episodes = args.episodes or eval_cfg.get('n_episodes', 20)
    deterministic = args.deterministic if args.deterministic is not None \
        else eval_cfg.get('deterministic', True)
    save_dir = str(paths['eval'])
    ensure_stage_dirs(paths, 'eval')
    model_path = args.model or default_model_path(paths)

    # --- Load model ---
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # --- Create environment ---
    env = InterceptionEnv(config=config)
    if args.cbf:
        from envs.wrappers.cbf_wrapper import CBFWrapper
        env = CBFWrapper(
            env,
            alpha_fov=args.cbf_alpha_fov,
            alpha_attitude=args.cbf_alpha_att,
            horizon_fov=args.cbf_horizon_fov,
            horizon_attitude=args.cbf_horizon_att,
            in_fov_only=True,
        )
        print(f"CBF safety filter ENABLED: alpha_fov={args.cbf_alpha_fov} "
              f"alpha_att={args.cbf_alpha_att} "
              f"horizon_fov={args.cbf_horizon_fov} "
              f"horizon_att={args.cbf_horizon_att}")

    # --- Run evaluation ---
    print(f"Running {n_episodes} evaluation episodes "
          f"(deterministic={deterministic})...")
    all_data = []
    for i in range(n_episodes):
        data = evaluate_episode(
            model, env, deterministic=deterministic, seed=args.seed + i
        )
        outcome_symbol = {'success': '✓', 'fov_loss': '✗', 'timeout': '⏱'}
        symbol = outcome_symbol.get(data['outcome'], '?')
        print(f"  Episode {i+1:3d}/{n_episodes}: {symbol} {data['outcome']:<10s} "
              f"dist={data['relative_distance'][-1]:.2f}m  "
              f"img_err={np.mean(data['image_error']):.4f}  "
              f"steps={len(data['rewards'])}")
        all_data.append(data)

    # --- Print summary ---
    print_summary(all_data)

    # --- Generate plots ---
    # Select episode to plot
    if args.plot_episode == -1:
        # Best episode (lowest final distance)
        idx = np.argmin([d['relative_distance'][-1] for d in all_data])
    else:
        idx = min(args.plot_episode, len(all_data) - 1)

    print(f"\nGenerating detailed plot for episode {idx + 1}...")
    plot_path = os.path.join(save_dir, f'episode_{idx + 1}_analysis.png')
    plot_single_episode(all_data[idx], config, save_path=plot_path)

    env.close()
    print("\nEvaluation complete.")


if __name__ == '__main__':
    main()
