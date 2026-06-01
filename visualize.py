"""
Animated Visualization — IBVS Drone Interception
==================================================
Produces a synchronized 4-panel animated replay of an evaluation episode:

  Panel 1 (top-left):  3D Trajectory with FOV cone
  Panel 2 (top-right): Camera image plane (target dot + FOV boundary)
  Panel 3 (bot-left):  Relative distance & image error over time
  Panel 4 (bot-right): Action commands & cumulative reward

Usage:
    python visualize.py --model logs/models/ibvs_ppo_final
    python visualize.py --model logs/models/ibvs_ppo_final --save video.mp4
    python visualize.py --model logs/models/ibvs_ppo_final --episode best --fps 30

Requirements:
    pip install matplotlib numpy pyyaml stable-baselines3
    For MP4 export: ffmpeg must be installed (brew install ffmpeg)
"""

import os
import argparse
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from stable_baselines3 import PPO
from envs.interception_env import InterceptionEnv
from models.camera_model import PinholeCamera
from experiment_paths import (
    default_model_path,
    ensure_stage_dirs,
    get_stage_paths,
)


# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------
COLORS = {
    'interceptor': '#00B4D8',       # Cyan-blue
    'target': '#FF6B6B',            # Coral red
    'fov_cone': '#00B4D8',          # Matching interceptor
    'fov_boundary': '#FFB703',      # Amber
    'success': '#06D6A0',           # Mint green
    'fail': '#EF476F',              # Rose
    'action_vx': '#8338EC',         # Purple
    'action_vy': '#3A86FF',         # Blue
    'action_vz': '#06D6A0',         # Mint
    'action_yaw': '#FF006E',        # Magenta
    'reward': '#FFBE0B',            # Gold
    'distance': '#00B4D8',          # Cyan
    'image_err': '#FF6B6B',         # Coral
    'bg_dark': '#1A1A2E',           # Dark navy
    'grid': '#2A2A4A',              # Soft grid
    'text': '#E0E0E0',              # Light text
}


# ---------------------------------------------------------------------------
# Data Collection
# ---------------------------------------------------------------------------

def collect_episode(model, env, deterministic=True, seed=42):
    """Run one episode and collect all data needed for visualization.

    Args:
        model: Trained SB3 model.
        env: InterceptionEnv instance.
        deterministic: Whether to use deterministic policy.
        seed: Random seed.

    Returns:
        dict with numpy arrays for each recorded quantity.
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
        'yaw': [env.unwrapped.interceptor.yaw],
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
        data['yaw'].append(env.unwrapped.interceptor.yaw)

    data['outcome'] = info['episode_outcome']

    # Convert to arrays
    for key in ['interceptor_pos', 'target_pos', 'p_bar']:
        data[key] = np.array(data[key])
    for key in ['image_error', 'relative_distance', 'in_fov',
                'fov_margin', 'rewards', 'yaw']:
        data[key] = np.array(data[key])
    data['actions'] = np.array(data['actions']) if data['actions'] else np.zeros((0, 4))

    return data


# ---------------------------------------------------------------------------
# FOV Cone Helper
# ---------------------------------------------------------------------------

def compute_fov_cone_verts(position, yaw, half_hfov, half_vfov, cone_length=3.0):
    """Compute the 4 corner vertices of a FOV frustum in EFCS (NED, z-flipped for viz).

    Args:
        position: Interceptor position (3,) in EFCS.
        yaw: Heading angle (rad).
        half_hfov: Half horizontal FOV (rad).
        half_vfov: Half vertical FOV (rad).
        cone_length: Length of the FOV cone for display.

    Returns:
        np.ndarray (4, 3): Corner vertices in visualization coords (z-up).
    """
    # Body forward direction in NED
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    # The camera looks along body x-axis (forward-looking config).
    # Body frame axes in EFCS:
    fwd = np.array([cos_y, sin_y, 0.0])         # body x
    right = np.array([-sin_y, cos_y, 0.0])      # body y (NED: right)
    down = np.array([0.0, 0.0, 1.0])             # body z (NED: down)

    tan_h = np.tan(half_hfov)
    tan_v = np.tan(half_vfov)

    # Four corners of the FOV frustum at distance `cone_length`
    corners_body = [
        fwd * cone_length + right * ( tan_h * cone_length) + down * ( tan_v * cone_length),
        fwd * cone_length + right * (-tan_h * cone_length) + down * ( tan_v * cone_length),
        fwd * cone_length + right * (-tan_h * cone_length) + down * (-tan_v * cone_length),
        fwd * cone_length + right * ( tan_h * cone_length) + down * (-tan_v * cone_length),
    ]

    # Shift to interceptor position and flip z for visualization
    verts = []
    for c in corners_body:
        p = position + c
        verts.append([p[0], p[1], -p[2]])  # NED→z-up

    return np.array(verts)


# ---------------------------------------------------------------------------
# Animated Visualization
# ---------------------------------------------------------------------------

def create_animation(data, config, fps=20, save_path=None, skip=2):
    """Create a synchronized 4-panel animation.

    Args:
        data: Episode data dict from collect_episode().
        config: Parsed config.yaml dict.
        fps: Frames per second for the animation.
        save_path: If set, save as MP4 to this path.
        skip: Plot every `skip`-th frame (for speed).
    """
    dt = config['interceptor']['dt']
    n_total = len(data['rewards'])
    frame_indices = list(range(0, n_total + 1, skip))
    if frame_indices[-1] != n_total:
        frame_indices.append(n_total)
    n_frames = len(frame_indices)

    camera = PinholeCamera(config['camera'])
    fov_params = camera.get_fov_params()
    tan_h = fov_params['tan_half_hfov']
    tan_v = fov_params['tan_half_vfov']
    half_hfov = fov_params['alpha_hfov'] / 2.0
    half_vfov = fov_params['alpha_vfov'] / 2.0

    int_pos = data['interceptor_pos']
    tgt_pos = data['target_pos']
    outcome = data['outcome']

    # ---- Set up figure ----
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(COLORS['bg_dark'])

    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.28,
                  left=0.06, right=0.97, top=0.92, bottom=0.07)

    outcome_color = COLORS['success'] if outcome == 'success' else COLORS['fail']
    fig.suptitle(
        f'IBVS Drone Interception Replay',
        fontsize=18, fontweight='bold', color=COLORS['text'], y=0.97
    )

    # =====================================================================
    # Panel 1: 3D Trajectory + FOV Cone
    # =====================================================================
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax3d.set_facecolor(COLORS['bg_dark'])
    ax3d.set_xlabel('X (North) [m]', color=COLORS['text'], fontsize=8)
    ax3d.set_ylabel('Y (East) [m]', color=COLORS['text'], fontsize=8)
    ax3d.set_zlabel('Z (Up) [m]', color=COLORS['text'], fontsize=8)
    ax3d.set_title('3D Trajectory & FOV', color=COLORS['text'], fontsize=11, pad=10)
    ax3d.tick_params(colors=COLORS['text'], labelsize=7)

    # Pre-compute axis limits
    all_pos = np.vstack([int_pos, tgt_pos])
    margin = 5.0
    ax3d.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ax3d.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    ax3d.set_zlim(-all_pos[:, 2].max() - margin, -all_pos[:, 2].min() + margin)

    # Static trails (will be drawn up to current frame)
    int_trail, = ax3d.plot([], [], [], color=COLORS['interceptor'], lw=2,
                           alpha=0.6, label='Interceptor')
    tgt_trail, = ax3d.plot([], [], [], color=COLORS['target'], lw=2,
                           alpha=0.6, linestyle='--', label='Target')
    int_dot = ax3d.plot([], [], [], 'o', color=COLORS['interceptor'],
                        markersize=8, zorder=10)[0]
    tgt_dot = ax3d.plot([], [], [], 's', color=COLORS['target'],
                        markersize=8, zorder=10)[0]

    # Start markers
    ax3d.plot([int_pos[0, 0]], [int_pos[0, 1]], [-int_pos[0, 2]],
              'o', color=COLORS['interceptor'], markersize=10, alpha=0.3)
    ax3d.plot([tgt_pos[0, 0]], [tgt_pos[0, 1]], [-tgt_pos[0, 2]],
              's', color=COLORS['target'], markersize=10, alpha=0.3)

    ax3d.legend(fontsize=7, loc='upper left', facecolor=COLORS['bg_dark'],
                edgecolor=COLORS['grid'], labelcolor=COLORS['text'])

    # FOV cone (will be updated each frame)
    fov_polys = [None]  # mutable container for the Poly3DCollection

    # =====================================================================
    # Panel 2: Camera Image Plane
    # =====================================================================
    ax_cam = fig.add_subplot(gs[0, 1])
    ax_cam.set_facecolor('#0D1117')
    ax_cam.set_xlim(-1.4, 1.4)
    ax_cam.set_ylim(-1.4, 1.4)
    ax_cam.set_aspect('equal')
    ax_cam.set_xlabel('Image X (norm)', color=COLORS['text'], fontsize=9)
    ax_cam.set_ylabel('Image Y (norm)', color=COLORS['text'], fontsize=9)
    ax_cam.set_title('Camera Image Plane', color=COLORS['text'], fontsize=11, pad=10)
    ax_cam.tick_params(colors=COLORS['text'], labelsize=7)

    # FOV boundary rectangle
    fov_rect = patches.FancyBboxPatch(
        (-1, -1), 2, 2,
        boxstyle="round,pad=0.02",
        linewidth=2, edgecolor=COLORS['fov_boundary'],
        facecolor='none', linestyle='-', zorder=2
    )
    ax_cam.add_patch(fov_rect)

    # Crosshair at center
    ax_cam.axhline(0, color=COLORS['grid'], lw=0.5, alpha=0.4)
    ax_cam.axvline(0, color=COLORS['grid'], lw=0.5, alpha=0.4)
    ax_cam.plot(0, 0, '+', color=COLORS['fov_boundary'], markersize=18,
                markeredgewidth=1.5, alpha=0.6, zorder=3)

    # Concentric guide circles
    for r in [0.25, 0.5, 0.75]:
        circle = plt.Circle((0, 0), r, fill=False, color=COLORS['grid'],
                             lw=0.5, alpha=0.3, linestyle=':')
        ax_cam.add_patch(circle)

    # Target dot on image (will be updated)
    cam_trail, = ax_cam.plot([], [], '-', color=COLORS['target'],
                              alpha=0.3, lw=1, zorder=4)
    cam_dot, = ax_cam.plot([], [], 'o', color=COLORS['target'],
                            markersize=12, zorder=5)
    cam_dot_inner, = ax_cam.plot([], [], 'o', color='white',
                                  markersize=4, zorder=6)

    # Status text on image panel
    cam_status_text = ax_cam.text(
        0.02, 0.98, '', transform=ax_cam.transAxes,
        fontsize=9, color=COLORS['text'], va='top', ha='left',
        fontfamily='monospace',
        bbox=dict(facecolor=COLORS['bg_dark'], alpha=0.7, edgecolor='none', pad=4)
    )

    # =====================================================================
    # Panel 3: Distance & Image Error
    # =====================================================================
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_dist.set_facecolor(COLORS['bg_dark'])
    ax_dist.set_xlabel('Time [s]', color=COLORS['text'], fontsize=9)
    ax_dist.set_title('Distance & Image Error', color=COLORS['text'], fontsize=11, pad=10)
    ax_dist.tick_params(colors=COLORS['text'], labelsize=7)

    time_full = np.arange(n_total + 1) * dt

    # Distance axis (left)
    ax_dist.set_ylabel('Relative Distance [m]', color=COLORS['distance'], fontsize=9)
    ax_dist.set_xlim(0, time_full[-1])
    ax_dist.set_ylim(0, max(data['relative_distance']) * 1.1 + 1)
    ax_dist.axhline(config['env']['d_success'], color=COLORS['success'],
                     ls='--', lw=1, alpha=0.6, label=f"d_success = {config['env']['d_success']}m")

    dist_line, = ax_dist.plot([], [], color=COLORS['distance'], lw=2, label='Distance')
    dist_fill = None  # will be set in init

    # Image error axis (right)
    ax_err = ax_dist.twinx()
    ax_err.set_ylabel('Image Error ||p̄||', color=COLORS['image_err'], fontsize=9)
    ax_err.set_ylim(0, max(data['image_error']) * 1.1 + 0.05)
    ax_err.tick_params(axis='y', colors=COLORS['image_err'], labelsize=7)

    err_line, = ax_err.plot([], [], color=COLORS['image_err'], lw=1.5,
                             alpha=0.8, label='Image Error')

    # Vertical time cursor
    time_cursor_dist = ax_dist.axvline(0, color=COLORS['text'], lw=0.8,
                                        alpha=0.4, ls=':')

    ax_dist.legend(fontsize=7, loc='upper right', facecolor=COLORS['bg_dark'],
                   edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    ax_dist.grid(True, color=COLORS['grid'], alpha=0.2)

    # =====================================================================
    # Panel 4: Actions & Cumulative Reward
    # =====================================================================
    ax_act = fig.add_subplot(gs[1, 1])
    ax_act.set_facecolor(COLORS['bg_dark'])
    ax_act.set_xlabel('Time [s]', color=COLORS['text'], fontsize=9)
    ax_act.set_ylabel('Action [-1, 1]', color=COLORS['text'], fontsize=9)
    ax_act.set_title('Actions & Cumulative Reward', color=COLORS['text'],
                     fontsize=11, pad=10)
    ax_act.tick_params(colors=COLORS['text'], labelsize=7)
    ax_act.set_xlim(0, time_full[-1])
    ax_act.set_ylim(-1.3, 1.3)

    action_labels = ['v_x (fwd)', 'v_y (lat)', 'v_z (vert)', 'ψ̇ (yaw)']
    action_colors = [COLORS['action_vx'], COLORS['action_vy'],
                     COLORS['action_vz'], COLORS['action_yaw']]
    act_lines = []
    for i in range(4):
        line, = ax_act.plot([], [], color=action_colors[i], lw=1.5,
                             alpha=0.8, label=action_labels[i])
        act_lines.append(line)

    # Cumulative reward on secondary axis
    ax_rew = ax_act.twinx()
    cum_rew_full = np.cumsum(data['rewards'])
    rew_min = min(0, cum_rew_full.min()) - 5
    rew_max = cum_rew_full.max() + 10
    ax_rew.set_ylim(rew_min, rew_max)
    ax_rew.set_ylabel('Cumulative Reward', color=COLORS['reward'], fontsize=9)
    ax_rew.tick_params(axis='y', colors=COLORS['reward'], labelsize=7)

    rew_line, = ax_rew.plot([], [], color=COLORS['reward'], lw=2,
                             alpha=0.7, label='Σ Reward')

    # Time cursor
    time_cursor_act = ax_act.axvline(0, color=COLORS['text'], lw=0.8,
                                      alpha=0.4, ls=':')

    ax_act.legend(fontsize=7, loc='upper left', facecolor=COLORS['bg_dark'],
                  edgecolor=COLORS['grid'], labelcolor=COLORS['text'],
                  ncol=2)
    ax_act.grid(True, color=COLORS['grid'], alpha=0.2)

    # Global status text
    status_text = fig.text(
        0.5, 0.93, '', ha='center', va='bottom', fontsize=12,
        color=COLORS['text'], fontfamily='monospace',
        bbox=dict(facecolor=COLORS['bg_dark'], alpha=0.8,
                  edgecolor=COLORS['grid'], pad=6)
    )

    # =====================================================================
    # Animation Functions
    # =====================================================================

    def init():
        """Initialize all artists."""
        int_trail.set_data_3d([], [], [])
        tgt_trail.set_data_3d([], [], [])
        int_dot.set_data_3d([], [], [])
        tgt_dot.set_data_3d([], [], [])
        cam_trail.set_data([], [])
        cam_dot.set_data([], [])
        cam_dot_inner.set_data([], [])
        cam_status_text.set_text('')
        dist_line.set_data([], [])
        err_line.set_data([], [])
        rew_line.set_data([], [])
        for l in act_lines:
            l.set_data([], [])
        status_text.set_text('')
        return []

    def update(frame_num):
        """Update all panels for frame `frame_num`."""
        idx = frame_indices[frame_num]
        t_current = idx * dt

        # === Panel 1: 3D Trajectory ===
        s = max(0, idx - 200)  # trail window
        trail_x = int_pos[s:idx + 1, 0]
        trail_y = int_pos[s:idx + 1, 1]
        trail_z = -int_pos[s:idx + 1, 2]
        int_trail.set_data_3d(trail_x, trail_y, trail_z)

        trail_tx = tgt_pos[s:idx + 1, 0]
        trail_ty = tgt_pos[s:idx + 1, 1]
        trail_tz = -tgt_pos[s:idx + 1, 2]
        tgt_trail.set_data_3d(trail_tx, trail_ty, trail_tz)

        # Current positions
        int_dot.set_data_3d([int_pos[idx, 0]], [int_pos[idx, 1]],
                            [-int_pos[idx, 2]])
        tgt_dot.set_data_3d([tgt_pos[idx, 0]], [tgt_pos[idx, 1]],
                            [-tgt_pos[idx, 2]])

        # FOV cone
        if fov_polys[0] is not None:
            try:
                fov_polys[0].remove()
            except ValueError:
                pass

        cone_len = min(data['relative_distance'][idx] * 0.3, 5.0)
        if cone_len > 0.5:
            verts = compute_fov_cone_verts(
                int_pos[idx], data['yaw'][idx],
                half_hfov, half_vfov, cone_length=cone_len
            )
            apex = np.array([int_pos[idx, 0], int_pos[idx, 1], -int_pos[idx, 2]])
            faces = [
                [apex, verts[0], verts[1]],
                [apex, verts[1], verts[2]],
                [apex, verts[2], verts[3]],
                [apex, verts[3], verts[0]],
                [verts[0], verts[1], verts[2], verts[3]],  # base
            ]
            poly = Poly3DCollection(
                faces, alpha=0.08, facecolor=COLORS['fov_cone'],
                edgecolor=COLORS['fov_cone'], linewidth=0.5
            )
            ax3d.add_collection3d(poly)
            fov_polys[0] = poly

        # === Panel 2: Camera Image Plane ===
        p_bar = data['p_bar'][:idx + 1]
        p_bar_norm_x = p_bar[:, 0] / tan_h
        p_bar_norm_y = p_bar[:, 1] / tan_v

        # Trail (last 80 points)
        trail_start = max(0, len(p_bar_norm_x) - 80)
        cam_trail.set_data(p_bar_norm_x[trail_start:], p_bar_norm_y[trail_start:])

        # Current dot
        cx, cy = p_bar_norm_x[-1], p_bar_norm_y[-1]
        cam_dot.set_data([cx], [cy])
        cam_dot_inner.set_data([cx], [cy])

        # Color based on FOV status
        in_fov_now = data['in_fov'][idx]
        dot_color = COLORS['target'] if in_fov_now else COLORS['fail']
        cam_dot.set_color(dot_color)

        # Status text
        dist_now = data['relative_distance'][idx]
        img_err_now = data['image_error'][idx]
        fov_str = '● IN FOV' if in_fov_now else '✖ LOST'
        fov_color = COLORS['success'] if in_fov_now else COLORS['fail']
        cam_status_text.set_text(
            f"dist: {dist_now:6.2f} m\n"
            f"err:  {img_err_now:6.4f}\n"
            f"fov:  {fov_str}"
        )

        # === Panel 3: Distance & Image Error ===
        t_slice = time_full[:idx + 1]
        dist_line.set_data(t_slice, data['relative_distance'][:idx + 1])
        err_line.set_data(t_slice, data['image_error'][:idx + 1])
        time_cursor_dist.set_xdata([t_current])

        # === Panel 4: Actions & Reward ===
        if idx > 0:
            t_act = time_full[1:idx + 1]
            actions = data['actions'][:idx]
            for i in range(4):
                act_lines[i].set_data(t_act, actions[:, i])

            cum_rew = np.cumsum(data['rewards'][:idx])
            rew_line.set_data(t_act, cum_rew)

        time_cursor_act.set_xdata([t_current])

        # === Global Status ===
        step_str = f"Step {idx:4d}/{n_total}"
        time_str = f"t = {t_current:5.2f}s"
        dist_str = f"dist = {dist_now:6.2f}m"
        if idx == n_total:
            outcome_str = f"OUTCOME: {outcome.upper()}"
            o_color = outcome_color
        else:
            outcome_str = "RUNNING"
            o_color = COLORS['text']

        status_text.set_text(
            f"  {step_str}  │  {time_str}  │  {dist_str}  │  {outcome_str}  "
        )
        status_text.set_color(o_color)

        return []

    # =====================================================================
    # Build and Run Animation
    # =====================================================================
    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=n_frames, interval=1000 // fps,
        blit=False, repeat=False
    )

    if save_path:
        print(f"Saving animation to: {save_path}")
        ext = os.path.splitext(save_path)[1].lower()
        if ext == '.gif':
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(
                fps=fps, metadata={'title': 'IBVS Drone Interception'},
                bitrate=3000
            )
        anim.save(save_path, writer=writer, dpi=120)
        print(f"  ✓ Saved ({n_frames} frames @ {fps} fps)")
    else:
        plt.show()

    plt.close(fig)
    return anim


# ---------------------------------------------------------------------------
# Static Summary Dashboard
# ---------------------------------------------------------------------------

def create_summary_dashboard(all_data, config, save_path=None):
    """Create a static multi-episode summary dashboard.

    Shows aggregate statistics across multiple evaluation episodes.

    Args:
        all_data: List of episode data dicts.
        config: Parsed config dict.
        save_path: If set, save figure.
    """
    n_episodes = len(all_data)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(COLORS['bg_dark'])
    fig.suptitle(
        f'IBVS Evaluation Dashboard — {n_episodes} Episodes',
        fontsize=16, fontweight='bold', color=COLORS['text'], y=0.97
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                  left=0.06, right=0.97, top=0.90, bottom=0.08)

    dt = config['interceptor']['dt']

    # Categorize episodes
    outcomes = [d['outcome'] for d in all_data]
    n_success = sum(1 for o in outcomes if o == 'success')
    n_fov = sum(1 for o in outcomes if o == 'fov_loss')
    n_timeout = sum(1 for o in outcomes if o == 'timeout')

    # ---- Panel 1: Outcome Pie ----
    ax_pie = fig.add_subplot(gs[0, 0])
    ax_pie.set_facecolor(COLORS['bg_dark'])
    labels = ['Success', 'FOV Loss', 'Timeout']
    sizes = [n_success, n_fov, n_timeout]
    colors_pie = [COLORS['success'], COLORS['fail'], COLORS['fov_boundary']]
    # Filter out zero-sized slices
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors_pie) if s > 0]
    if non_zero:
        labels_f, sizes_f, colors_f = zip(*non_zero)
        wedges, texts, autotexts = ax_pie.pie(
            sizes_f, labels=labels_f, colors=colors_f, autopct='%1.0f%%',
            pctdistance=0.75, startangle=90,
            textprops={'color': COLORS['text'], 'fontsize': 10}
        )
        for at in autotexts:
            at.set_fontweight('bold')
    ax_pie.set_title('Episode Outcomes', color=COLORS['text'], fontsize=11, pad=10)

    # ---- Panel 2: Final Distance Distribution ----
    ax_fdist = fig.add_subplot(gs[0, 1])
    ax_fdist.set_facecolor(COLORS['bg_dark'])
    final_dists = [d['relative_distance'][-1] for d in all_data]
    ax_fdist.hist(final_dists, bins=20, color=COLORS['distance'],
                  alpha=0.7, edgecolor='white', linewidth=0.5)
    ax_fdist.axvline(config['env']['d_success'], color=COLORS['success'],
                     ls='--', lw=2, label=f"d_success = {config['env']['d_success']}m")
    ax_fdist.set_xlabel('Final Distance [m]', color=COLORS['text'], fontsize=9)
    ax_fdist.set_ylabel('Count', color=COLORS['text'], fontsize=9)
    ax_fdist.set_title('Final Distance Distribution', color=COLORS['text'],
                       fontsize=11, pad=10)
    ax_fdist.legend(fontsize=8, facecolor=COLORS['bg_dark'],
                    edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    ax_fdist.tick_params(colors=COLORS['text'], labelsize=7)
    ax_fdist.grid(True, color=COLORS['grid'], alpha=0.2)

    # ---- Panel 3: Episode Length Distribution ----
    ax_elen = fig.add_subplot(gs[0, 2])
    ax_elen.set_facecolor(COLORS['bg_dark'])
    ep_lens = [len(d['rewards']) for d in all_data]
    ax_elen.hist(ep_lens, bins=20, color=COLORS['action_vx'],
                 alpha=0.7, edgecolor='white', linewidth=0.5)
    ax_elen.set_xlabel('Episode Length [steps]', color=COLORS['text'], fontsize=9)
    ax_elen.set_ylabel('Count', color=COLORS['text'], fontsize=9)
    ax_elen.set_title('Episode Length Distribution', color=COLORS['text'],
                      fontsize=11, pad=10)
    ax_elen.tick_params(colors=COLORS['text'], labelsize=7)
    ax_elen.grid(True, color=COLORS['grid'], alpha=0.2)

    # ---- Panel 4: Overlaid Distance Curves ----
    ax_dcurves = fig.add_subplot(gs[1, 0])
    ax_dcurves.set_facecolor(COLORS['bg_dark'])
    for d in all_data:
        t = np.arange(len(d['relative_distance'])) * dt
        color = COLORS['success'] if d['outcome'] == 'success' else (
            COLORS['fail'] if d['outcome'] == 'fov_loss' else COLORS['fov_boundary']
        )
        ax_dcurves.plot(t, d['relative_distance'], color=color, alpha=0.4, lw=1)
    ax_dcurves.axhline(config['env']['d_success'], color=COLORS['success'],
                       ls='--', lw=1.5, alpha=0.7)
    ax_dcurves.set_xlabel('Time [s]', color=COLORS['text'], fontsize=9)
    ax_dcurves.set_ylabel('Distance [m]', color=COLORS['text'], fontsize=9)
    ax_dcurves.set_title('Distance Over Time (all episodes)',
                         color=COLORS['text'], fontsize=11, pad=10)
    ax_dcurves.tick_params(colors=COLORS['text'], labelsize=7)
    ax_dcurves.grid(True, color=COLORS['grid'], alpha=0.2)

    # ---- Panel 5: Overlaid Image Error ----
    ax_ecurves = fig.add_subplot(gs[1, 1])
    ax_ecurves.set_facecolor(COLORS['bg_dark'])
    for d in all_data:
        t = np.arange(len(d['image_error'])) * dt
        color = COLORS['success'] if d['outcome'] == 'success' else (
            COLORS['fail'] if d['outcome'] == 'fov_loss' else COLORS['fov_boundary']
        )
        ax_ecurves.plot(t, d['image_error'], color=color, alpha=0.4, lw=1)
    ax_ecurves.set_xlabel('Time [s]', color=COLORS['text'], fontsize=9)
    ax_ecurves.set_ylabel('Image Error ||p̄||', color=COLORS['text'], fontsize=9)
    ax_ecurves.set_title('Image Error Over Time (all episodes)',
                         color=COLORS['text'], fontsize=11, pad=10)
    ax_ecurves.tick_params(colors=COLORS['text'], labelsize=7)
    ax_ecurves.grid(True, color=COLORS['grid'], alpha=0.2)

    # ---- Panel 6: Summary Stats Table ----
    ax_table = fig.add_subplot(gs[1, 2])
    ax_table.set_facecolor(COLORS['bg_dark'])
    ax_table.axis('off')
    ax_table.set_title('Summary Statistics', color=COLORS['text'],
                       fontsize=11, pad=10)

    total_rewards = [np.sum(d['rewards']) for d in all_data]
    mean_errors = [np.mean(d['image_error']) for d in all_data]

    stats = [
        ['Metric', 'Value'],
        ['Episodes', f'{n_episodes}'],
        ['Success Rate', f'{100*n_success/n_episodes:.1f}%'],
        ['FOV Loss Rate', f'{100*n_fov/n_episodes:.1f}%'],
        ['Timeout Rate', f'{100*n_timeout/n_episodes:.1f}%'],
        ['', ''],
        ['Mean Final Dist', f'{np.mean(final_dists):.2f} ± {np.std(final_dists):.2f} m'],
        ['Mean Img Error', f'{np.mean(mean_errors):.4f}'],
        ['Mean Ep Length', f'{np.mean(ep_lens):.0f} steps'],
        ['Mean Reward', f'{np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}'],
    ]

    # Successful episodes stats
    if n_success > 0:
        success_dists = [d['relative_distance'][-1]
                         for d in all_data if d['outcome'] == 'success']
        success_lens = [len(d['rewards'])
                        for d in all_data if d['outcome'] == 'success']
        stats.append(['', ''])
        stats.append(['── Success ──', ''])
        stats.append(['  Final Dist', f'{np.mean(success_dists):.3f} m'])
        stats.append(['  Avg Steps', f'{np.mean(success_lens):.0f}'])

    y_start = 0.95
    for i, (metric, value) in enumerate(stats):
        y = y_start - i * 0.065
        if metric.startswith('──'):
            ax_table.text(0.05, y, metric, transform=ax_table.transAxes,
                          fontsize=9, color=COLORS['success'], fontweight='bold',
                          fontfamily='monospace')
        else:
            ax_table.text(0.05, y, metric, transform=ax_table.transAxes,
                          fontsize=9, color=COLORS['text'], fontfamily='monospace')
        ax_table.text(0.60, y, value, transform=ax_table.transAxes,
                      fontsize=9, color=COLORS['text'], fontweight='bold',
                      fontfamily='monospace')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Dashboard saved to: {save_path}")

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Animated visualization of trained IBVS agent'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to saved SB3 model (without .zip)'
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--stage', type=str, default=None,
        help='Experiment stage name for outputs and default model path'
    )
    parser.add_argument(
        '--save', type=str, default=None,
        help='Save animation to file (e.g., replay.mp4 or replay.gif)'
    )
    parser.add_argument(
        '--save-dashboard', type=str, default=None,
        help='Save summary dashboard to file (e.g., dashboard.png)'
    )
    parser.add_argument(
        '--episodes', type=int, default=1,
        help='Number of episodes to evaluate (animation shows the best one)'
    )
    parser.add_argument(
        '--episode', type=str, default='best',
        help="Which episode to animate: 'best', 'worst', or integer index"
    )
    parser.add_argument(
        '--fps', type=int, default=25,
        help='Animation frames per second'
    )
    parser.add_argument(
        '--skip', type=int, default=2,
        help='Show every N-th frame (higher = faster animation)'
    )
    parser.add_argument(
        '--deterministic', action='store_true', default=True,
        help='Use deterministic policy actions'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    paths = get_stage_paths(config, args.stage)
    model_path = args.model or default_model_path(paths)

    save_path = args.save
    if save_path is None and args.stage is not None:
        ensure_stage_dirs(paths, 'videos')
        save_path = str(paths['videos'] / 'replay_best.mp4')
    elif save_path is not None:
        save_parent = os.path.dirname(save_path)
        if save_parent:
            os.makedirs(save_parent, exist_ok=True)

    # --- Load model ---
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    # --- Create environment ---
    env = InterceptionEnv(config=config)

    # --- Collect episodes ---
    n_episodes = max(args.episodes, 1)
    print(f"Running {n_episodes} evaluation episode(s)...")
    all_data = []
    for i in range(n_episodes):
        data = collect_episode(
            model, env,
            deterministic=args.deterministic,
            seed=args.seed + i
        )
        outcome_sym = {'success': '✓', 'fov_loss': '✖', 'timeout': '⏱'}.get(
            data['outcome'], '?'
        )
        print(f"  Episode {i+1}/{n_episodes}: {outcome_sym} {data['outcome']:<10s}  "
              f"dist={data['relative_distance'][-1]:.2f}m  "
              f"steps={len(data['rewards'])}")
        all_data.append(data)

    # --- Select episode for animation ---
    if args.episode == 'best':
        idx = np.argmin([d['relative_distance'][-1] for d in all_data])
    elif args.episode == 'worst':
        idx = np.argmax([d['relative_distance'][-1] for d in all_data])
    else:
        idx = min(int(args.episode), len(all_data) - 1)

    print(f"\nAnimating episode {idx + 1} "
          f"(outcome={all_data[idx]['outcome']}, "
          f"final_dist={all_data[idx]['relative_distance'][-1]:.2f}m)")

    # --- Create animation ---
    create_animation(
        all_data[idx], config,
        fps=args.fps,
        save_path=save_path,
        skip=args.skip
    )

    # --- Summary dashboard ---
    if n_episodes > 1 or args.save_dashboard or args.stage is not None:
        save_dash = args.save_dashboard
        if save_dash is None:
            ensure_stage_dirs(paths, 'eval')
            save_dash = str(paths['eval'] / 'dashboard.png')
        else:
            save_parent = os.path.dirname(save_dash)
            if save_parent:
                os.makedirs(save_parent, exist_ok=True)

        print(f"\nGenerating summary dashboard...")
        create_summary_dashboard(all_data, config, save_path=save_dash)

    env.close()
    print("\n✓ Visualization complete.")


if __name__ == '__main__':
    main()
