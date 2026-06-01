# RL-Guided High-Speed Drone Interception via Image-Based Visual Servoing (IBVS)

This repository implements a high-fidelity **Gymnasium simulation environment** and **Reinforcement Learning (PPO)** agent for high-speed drone interception using **Image-Based Visual Servoing (IBVS)**. 

The project is inspired by the research paper:  
> **"High-Speed Interception Multicopter Control by Image-based Visual Servoing"**  
> *Yang, Bai, She, and Quan (arXiv:2404.08296)*

Instead of relying on classical geometric control laws, this framework trains a deep reinforcement learning agent (using **Stable-Baselines3**) to intercept a highly maneuverable target drone directly from 2D visual tracking errors on the onboard camera's image plane.

---

## 🗺️ Coordinate Systems & Architecture

The simulator utilizes a physical **NED (North-East-Down)** coordinate convention (where the $z$-axis points downwards). There are three primary reference frames:
1. **Earth-Fixed Coordinate System (EFCS)**: The global inertial reference frame.
2. **Body Coordinate System (BCS)**: Attached to the center of mass of the interceptor drone.
3. **Camera Coordinate System (CCS)**: Attached to the camera optical center. 
   - *Default Mounting Configuration*: A forward-looking camera aligned with the body $x$-axis. This maps the body frame to the camera frame (body $x$ $\rightarrow$ camera optical axis $z$, body $y$ $\rightarrow$ camera $x$ image right, body $z$ $\rightarrow$ camera $y$ image down).

```
          [ Body x-axis (Forward) ] ──> Camera optical axis (z-axis)
                        │
      [ Body y-axis (Right) ] ──> Camera x-axis (Image-plane Right)
                        │
      [ Body z-axis (Down) ] ──> Camera y-axis (Image-plane Down)
```

---

## 📁 Repository Structure

```filepath
├── config.yaml            # Centralized system and training configuration
├── requirements.txt       # Python package dependencies
├── train.py               # Vectorized PPO training script
├── eval.py                # Static evaluation & multi-panel metrics analyzer
├── visualize.py           # Animated 4-panel episode player & summary dashboard
├── envs/
│   ├── __init__.py
│   └── interception_env.py# Custom Gymnasium environment mapping dynamics & rewards
└── models/
    ├── __init__.py
    ├── drone_dynamics.py  # Kinematic model of the interceptor
    ├── camera_model.py    # Pinhole projection and FOV boundaries
    └── target_model.py    # Target drone with 4 distinct flight profiles
```

### Stage-Based Output Layout
Training artifacts are organized by experiment stage so that results from
Stage 1a, Stage 1b, Stage 2a, and later stages can be compared later:

```filepath
logs/stages/
├── stage1a/
│   ├── models/
│   ├── tensorboard/
│   ├── eval/
│   ├── videos/
│   └── depth_test/
└── stage1b/
    ├── models/
    ├── tensorboard/
    ├── eval/
    ├── videos/
    └── depth_test/
```

---

## ⚙️ Installation & Setup

### 1. Environment Creation (with Conda/Miniconda)
Create and activate a clean Python 3.10 environment:
```bash
conda create -n ibvs_rl python=3.10 -y
conda activate ibvs_rl
```

### 2. Dependency Installation
Install all core libraries, including Stable-Baselines3, PyYAML, Gymnasium, and visualization utilities:
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg (Optional but Recommended)
To save high-quality MP4/GIF animations of drone trajectories, make sure `ffmpeg` is installed:
```bash
# macOS (using Homebrew)
brew install ffmpeg

# Linux (Debian/Ubuntu)
sudo apt update && sudo apt install ffmpeg -y
```

---

## 🚀 Execution & Usage Guide

### 1. Training the Agent (`train.py`)
Trains a PPO agent across multiple vectorized environments using the hyperparameter profile specified in `config.yaml`.

```bash
# Train using the default config.yaml
python train.py

# Train and save outputs under logs/stages/stage1a/
python train.py --stage stage3a --timesteps 15000000 --n-envs 16

# Train with an overridden custom configuration
python train.py --config config.yaml

# Train for a specific number of timesteps and run 4 parallel environments
python train.py --timesteps 500000 --n-envs 4

# Resume training from a saved checkpoint
python train.py --stage stage1a --resume logs/stages/stage1a/models/ibvs_ppo_100000_steps.zip
```

#### CLI Parameters:
* `--config` *(str)*: Path to the configuration YAML file (default: `config.yaml`).
* `--stage` *(str)*: Experiment stage name used for outputs (default: `experiment.stage` from `config.yaml`).
* `--timesteps` *(int)*: Override the total training timesteps.
* `--n-envs` *(int)*: Override the number of parallel simulation environments.
* `--resume` *(str)*: Path to a saved `.zip` model checkpoint to resume training from.

> [!NOTE]
> **Output Assets**:
> - Models are saved under `logs/stages/<stage>/models/` (a final model `ibvs_ppo_final.zip` and periodic checkpoints are generated).
> - TensorBoard event files are saved to `logs/stages/<stage>/tensorboard/`. Run `tensorboard --logdir logs/stages/<stage>/tensorboard` to inspect live progress.

---

### 2. Evaluating the Agent (`eval.py`)
Performs multiple evaluation episodes to assess the agent's performance (success rate, intercept time, and tracking errors) and generates a detailed diagnostic figure.

```bash
# Evaluate the final model on 20 episodes
python eval.py --stage stage1a

# Run 50 deterministic evaluation episodes using a custom random seed
python eval.py --stage stage1a --episodes 50 --seed 123 --deterministic

# Plot the best-performing episode instead of the default first episode
python eval.py --stage stage1a --plot-episode -1
```

#### CLI Parameters:
* `--model` *(str)*: Path to the saved model (exclude the `.zip` extension). If omitted, uses `logs/stages/<stage>/models/ibvs_ppo_final`.
* `--config` *(str)*: Path to the configuration YAML file.
* `--stage` *(str)*: Experiment stage name used for outputs and the default model path.
* `--episodes` *(int)*: Number of episodes to run during evaluation.
* `--deterministic` *(flag)*: Force the policy to choose actions deterministically.
* `--plot-episode` *(int)*: Index of the episode to plot. Set to `0` for the first, or `-1` to automatically choose the "best" episode (lowest relative interception distance).
* `--seed` *(int)*: Base random seed for the evaluation batch.

> [!NOTE]
> **Output Assets**:
> - An episode breakdown diagram is saved under `logs/stages/<stage>/eval/episode_<id>_analysis.png`, plotting 3D coordinates, camera coordinate trajectories, distance histories, actions, and step rewards.

---

### 3. Rendering Animated Replays & Summary Dashboards (`visualize.py`)
Generates high-performance interactive or file-based synchronized animations showing the drone in real-time alongside a diagnostic summary dashboard.

```bash
# Save the default 4-panel animation of the best evaluation episode
python visualize.py --stage stage1a --episodes 10 --episode best

# Export a high-resolution animation to an MP4 video file at 30 FPS
python visualize.py --stage stage3a --save logs/stages/stage3a/videos/replay_best.mp4 --fps 10

# Export the animation as a GIF
python visualize.py --stage stage4a_hardnet_d_seed7 --save logs/stages/stage4a_hardnet_d_seed7/videos/replay_best.gif --skip 3

# Run a 100-episode evaluation and save a static aggregate performance dashboard
python visualize.py --stage stage4a_hardnet_d_seed7 --episodes 100 --save-dashboard logs/stages/stage4a_hardnet_d_seed7/eval/dashboard.png
```

#### CLI Parameters:
* `--model` *(str)*: Path to the saved model file (exclude the `.zip` extension). If omitted, uses `logs/stages/<stage>/models/ibvs_ppo_final`.
* `--config` *(str)*: Path to the configuration YAML file.
* `--stage` *(str)*: Experiment stage name used for outputs and the default model path.
* `--save` *(str)*: Output filename for the exported animation (supports `.mp4` and `.gif`). If omitted with `--stage`, saves to `logs/stages/<stage>/videos/replay_best.mp4`; otherwise launches an interactive GUI window.
* `--save-dashboard` *(str)*: Path to save the multi-episode dashboard analysis image (e.g. `dashboard.png`).
* `--episodes` *(int)*: Number of evaluation episodes to run before selecting one to animate (default: `1`).
* `--episode` *(str)*: Selection criterion for the animation: `'best'` (lowest final distance), `'worst'`, or an explicit integer index (e.g., `0` for the first episode).
* `--fps` *(int)*: Playback speed frame rate (default: `25`).
* `--skip` *(int)*: Plot every $N$-th simulation step to speed up rendering and compress video file size (default: `2`).
* `--deterministic` *(flag)*: Run the model with deterministic policy outputs (default: `True`).
* `--seed` *(int)*: Initial random seed.

#### 🎛️ The 4-Panel Animation Layout:
* **Panel 1 (Top-Left)**: Animated 3D Trajectory showing the interceptor path (cyan), target path (coral), starting points, and the active 3D camera Field of View (FOV) cone.
* **Panel 2 (Top-Right)**: The camera's **2D Image Plane**, rendering the projected target position relative to the center crosshair and the yellow HFOV/VFOV boundary box (updates color to red if the target escapes the camera frame).
* **Panel 3 (Bottom-Left)**: Synchronized time series curves for relative 3D distance and normalized image plane error.
* **Panel 4 (Bottom-Right)**: Active action control commands ($v_x, v_y, v_z$ velocities and $\dot{\psi}$ yaw rate) paired with the cumulative reward.

---

## 📊 Target Evasion Modes

The target model in `models/target_model.py` dynamically selects one of four aggressive flight modes during episode reset, ensuring a robust RL policy:
1. **Stationary/Hovering (`hover`)**: The target floats at a static coordinate, acting as a basic visual servoing baseline.
2. **Constant Velocity (`constant_velocity`)**: The target travels along a straight vector with a random heading and pitch at up to `target.v_max`.
3. **Sinusoidal Evasion (`sinusoidal`)**: The target executes high-frequency, aggressive snake-like maneuvers perpendicular to its forward axis.
4. **Aggressive Orbit (`circular`)**: The target performs tight, high-G circular banks at orbital speeds to break visual lock.


 