# Revised Implementation Plan — From Pipeline Validation to Realistic Visual Interception

> **Project**: RL-Based Image-Guided High-Speed Drone Interception via IBVS  
> **Reference**: Yang, Bai, She, Quan — arXiv:2404.08296  
> **Last updated**: 2026-05-24  

---

## Current State and Lessons Learned

### What we have built (Stage 1a — COMPLETE)

A fully working Gymnasium + Stable-Baselines3 PPO pipeline with:
- Kinematic interceptor model (instant velocity commands, yaw-only heading)
- Point-mass target with 4 maneuver modes
- Pinhole camera projection with FOV constraints
- 18-dimensional observation vector (includes ground-truth distance, velocity, LOS)
- 4-dimensional action space (body-frame velocities + yaw rate)
- 7-component reward function
- Vectorized training, TensorBoard logging, evaluation, and animated visualization

### Training results
- **100% success rate** after ~1M timesteps (converged by ~500K)
- **0% FOV loss** and **0% timeout**
- Mean image error ~0.67 (high — agent doesn't center the target)

### Problems identified

| Problem | Root Cause | Evidence |
|---|---|---|
| **Zigzag trajectory** | No inertia — velocity reverses instantly every 0.02s | Action plot shows v_y, v_z slamming ±1 at every timestep |
| **Agent doesn't do visual servoing** | Ground-truth distance and velocity given in observation — agent solves a 3D pursuit problem, not a 2D image-based one | Image error stays high (~0.67) even at 100% success |
| **Reward hacking (earlier)** | Initial reward weights made hovering more rewarding than intercepting | Fixed by retuning, but reveals fragility |
| **Unrealistic speed advantage** | Interceptor v_max=15 m/s vs target v_max=10 m/s with zero dynamics lag | Problem is trivially solvable — just fly faster than the target |

### Key insight

> The current agent has learned to solve a **3D pursuit problem with perfect information**, not a **2D visual servoing problem with partial observability**. Every subsequent stage must strip away one "cheat" and force the agent to develop more sophisticated strategies.

---

## Revised Stage Architecture

The overall pipeline we are building toward:

```
Delayed Noisy Image → DKF Observer → RL Guidance Policy → CBF-HardNet Safety Filter → 6-DOF Controller → Multicopter
```

We will build this incrementally. **Each stage introduces exactly ONE new challenge.** This is critical — if training fails after a change, we know precisely what caused it.

```mermaid
graph LR
    A["Stage 1a ✅<br/>Kinematic + Full Obs"] --> B["Stage 1b<br/>+ Action Smoothing"]
    B --> C["Stage 2a<br/>− Ground Truth Obs"]
    C --> D["Stage 2b<br/>+ Noise & Delay + DKF"]
    D --> E["Stage 3a<br/>+ Simplified Dynamics"]
    E --> F["Stage 3b<br/>+ Full 6-DOF Dynamics"]
    F --> G["Stage 4a<br/>+ CBF-HardNet Safety"]
    G --> H["Stage 4b<br/>+ Realistic Perception"]
```

---

## Stage 1b: Action Smoothing and Realistic Control Constraints

### Goal
Eliminate the zigzag behavior by making the agent's actions physically plausible, while keeping the kinematic model.

### What changes

| Aspect | Stage 1a (current) | Stage 1b |
|---|---|---|
| Action semantics | Velocity command (instant response) | **Acceleration command** (velocity integrates over time) |
| Action rate | Can reverse ±1 every step | **Rate-limited** — `|a_t - a_{t-1}| ≤ Δa_max` |
| Effort penalty | `w_effort = -0.01` (negligible) | **`w_effort = -0.05`**, plus new **jerk penalty** `w_jerk = -0.1` |
| Observation | 18-dim | 18-dim + **previous action** (4-dim) → **22-dim** |

### Technical details

**Velocity integration model** (replaces instant velocity):
```
v_body(t+1) = clip(v_body(t) + a_cmd * dt, -v_max, v_max)
```

The agent commands acceleration `a_cmd ∈ [-a_max, a_max]` instead of velocity directly. This means:
- To reach v_max from rest, it takes `v_max / a_max` seconds (not 1 timestep)
- Reversing direction requires decelerating to zero first
- Zigzag becomes physically costly

**New reward terms**:
```
r_jerk = -w_jerk * ||a_t - a_{t-1}||²    (penalizes rapid action changes)
r_effort = -w_effort * ||a_t||²           (increased from -0.01 to -0.05)
```

### Files to modify

| File | Change |
|---|---|
| [drone_dynamics.py](file:///Users/eren/MacbookAir/GIT/Image_based_Visual_Servoing_RL/models/drone_dynamics.py) | Replace `self.velocity = R_be @ v_cmd_body` with acceleration integration |
| [interception_env.py](file:///Users/eren/MacbookAir/GIT/Image_based_Visual_Servoing_RL/envs/interception_env.py) | Add previous action to observation, add jerk penalty to reward |
| [config.yaml](file:///Users/eren/MacbookAir/GIT/Image_based_Visual_Servoing_RL/config.yaml) | Add `a_max`, `w_jerk`, update `w_effort`, add `action_rate_limit` |

### Training strategy
- **Fresh training** (do NOT warm-start from Stage 1a — the zigzag policy is harmful)
- Start with `a_max = 10 m/s²` (generous but not instant)
- Train for 3-5M timesteps (harder problem, needs more time)

### Success criterion
- Success rate > 90%
- 3D trajectory shows **smooth curves**, no zigzag
- Action plot shows gradual transitions, not bang-bang oscillations

---

## Stage 2a: Vision-Only Observation (Remove Ground-Truth Cheats)

### Goal
Force the agent to solve the actual visual servoing problem by removing ground-truth 3D information from the observation.

### What changes

| Observation dim | Stage 1b (22-dim) | Stage 2a |
|---|---|---|
| `obs[0:2]` | Image-plane error (p̄_x, p̄_y) | ✅ **Keep** |
| `obs[2:4]` | Image-plane velocity (dp̄/dt) | ✅ **Keep** |
| `obs[4]` | FOV flag (in_fov) | ✅ **Keep** |
| `obs[5]` | **Normalized distance** (ground truth) | ❌ **REMOVE** |
| `obs[6:9]` | Body velocity (from dynamics) | ✅ **Keep** (agent knows its own velocity) |
| `obs[9:12]` | Euler angles | ✅ **Keep** (from IMU) |
| `obs[12:15]` | **LOS unit vector** (ground truth) | ❌ **REMOVE** (redundant with image coords) |
| `obs[15:18]` | **Relative velocity** (ground truth) | ❌ **REMOVE** |
| `obs[18:22]` | Previous action | ✅ **Keep** |

**New observation vector (13-dim)**:
```
o_t = [p̄_x, p̄_y, dp̄_x/dt, dp̄_y/dt, in_fov, v_body_x, v_body_y, v_body_z, roll, pitch, yaw, a_prev(4-dim)]
```

> [!IMPORTANT]
> **This is the hardest single transition in the entire plan.** The agent loses knowledge of how far away the target is and how fast it's moving relative to itself. It must now infer approach behavior purely from how the target moves on the image plane (e.g., image expansion = getting closer).

### Reward function change

> [!WARNING]
> The approach reward `r_approach = -Δd` uses ground-truth distance. We have two options:

**Option A (Recommended)**: Keep ground-truth distance **in the reward only** (not in the observation). This is standard practice in RL — the reward can use privileged information during training. The agent learns to approach without seeing the distance, but gets rewarded for closing it.

**Option B**: Replace the approach reward with an image-based proxy:
```
r_approach_image = w_expand * Δ(1/depth_estimate)   # image expansion rate
```
This is purer but much harder to tune and may not converge.

### Training strategy
- **Fresh training** (the old policy relied on obs[5] distance — it won't transfer)
- May need **curriculum learning**: start with target at 5-10m, gradually increase to 10-30m
- Train for 5-10M timesteps
- Consider using **LSTM or frame-stacking** instead of MLP, since the agent now needs temporal context to infer depth from image motion

### Success criterion
- Success rate > 70% (expect significant drop from 100%)
- Agent develops a clear "approach and center" strategy visible in trajectories
- Image error should be LOWER than Stage 1 (agent must center target to track it)

---

## Stage 2b: Measurement Noise, Delay, and DKF State Observer

### Goal
Add realistic sensor imperfections and a Disturbance Kalman Filter to handle them.

### What changes

**Measurement model with delay and noise**:
```
z_k = [p̄_{k-D,x}, p̄_{k-D,y}]ᵀ + n_img    where n_img ~ N(0, σ²_img I)
```
- `D = 2-5` steps of delay (40-100ms at 50Hz — realistic camera processing latency)
- `σ_img = 0.01-0.05` (pixel noise in normalized coordinates)

**DKF state vector** (constant-velocity image-plane model):
```
x_k = [p̄_x, p̄_y, dp̄_x/dt, dp̄_y/dt]ᵀ
```

**DKF prediction model**:
```
x_{k+1} = F x_k + w_k
F = [[1, 0, Δt, 0],
     [0, 1, 0,  Δt],
     [0, 0, 1,  0 ],
     [0, 0, 0,  1 ]]
```

**DKF delayed measurement update**:
```
z_k = H x_{k-D} + n_k
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
```

The DKF runs as a wrapper around the environment. The agent receives the **DKF's filtered estimate** instead of raw (delayed, noisy) measurements.

### Key comparison experiment
```
                                  Success Rate    Mean Image Error
Stage 2a (no noise, no delay)         X%              Y
Stage 2b (noise+delay, NO DKF)        ?               ?          ← expect large drop
Stage 2b (noise+delay, WITH DKF)      ?               ?          ← expect recovery
```

### Files to add/modify

| File | Change |
|---|---|
| **[NEW]** `observers/dkf.py` | Implement the Disturbance Kalman Filter |
| **[NEW]** `envs/wrappers/noise_delay_wrapper.py` | Gymnasium wrapper that adds noise and delay to observations |
| **[NEW]** `envs/wrappers/dkf_wrapper.py` | Gymnasium wrapper that filters observations through the DKF |
| [interception_env.py](file:///Users/eren/MacbookAir/GIT/Image_based_Visual_Servoing_RL/envs/interception_env.py) | No changes — wrappers handle everything |

### Training strategy
- **Warm-start from Stage 2a policy** (the observation structure is identical — DKF outputs the same 4 values the agent already expects)
- Fine-tune for 2-3M timesteps
- Train separately with and without DKF for comparison

### Success criterion
- With DKF: success rate within 10% of Stage 2a (noise-free) performance
- Without DKF: measurable degradation (proves DKF adds value)
- DKF estimation error (compared to ground truth) should be small

---

## Stage 3a: Simplified Dynamics with Inertia (First-Order Response)

### Goal
Introduce physical realism into the drone's motion without the full complexity of 6-DOF dynamics. The drone now has mass and cannot change velocity instantaneously.

### What changes

**First-order velocity response model**:
```
v̇_body = (1/τ) * (v_cmd - v_body)        τ = 0.1-0.3s (time constant)
```

This means:
- Commanding v_cmd = 15 m/s from rest takes ~0.3s to reach (not instant)
- Reversing direction takes ~0.6s (decelerate + accelerate)
- The drone has realistic "sluggishness"

**Simplified attitude coupling** (pitch tilts with acceleration):
```
pitch_approx = -arctan(a_forward / g)     # pitching forward when accelerating
roll_approx  = arctan(a_lateral / g)      # banking when turning
```

> [!IMPORTANT]
> This is where pitch-camera coupling first appears. When the drone accelerates forward, it pitches nose-down, which tilts the camera downward. The agent must learn to manage this tradeoff — aggressive acceleration improves closure rate but risks losing the target from the camera FOV.

### Observation addition
- Add `pitch` and `roll` (derived from acceleration) to the observation
- The agent needs to know its current tilt to predict where the camera is pointing

### Training strategy
- **Fresh training** (dynamics are fundamentally different)
- Start with τ = 0.3s (easier), reduce to τ = 0.15s after convergence
- May need to reduce interceptor v_max to 10-12 m/s (the speed advantage was compensating for lack of dynamics)
- Train for 10-20M timesteps (significantly harder)

### Success criterion
- Success rate > 50% (expect major difficulty increase)
- Trajectories show smooth, physically plausible curves
- Agent learns to moderate acceleration near the target (to keep camera stable)

---

## Stage 3b: Full 6-DOF Multicopter Dynamics

### Goal
Replace the simplified dynamics with the complete Newton-Euler equations of motion for a multicopter.

### What changes

**Full 6-DOF state**:
```
State = [p(3), v(3), R(9 or quaternion 4), ω(3)]     (15-dim or 13-dim)
```

**Equations of motion** (from the paper, Eq. 1):
```
ṗ = v
v̇ = g + (1/m)(f + f_drag)
Ṙ = R [ω]×
J ω̇ = -ω × (Jω) + G_a + τ
```

**Action space changes**:
The RL agent no longer commands velocities. Instead:

| Option | Action | Low-level controller |
|---|---|---|
| **A (Recommended)** | Desired acceleration vector (3D) + desired yaw rate | PD attitude controller converts to motor thrusts |
| B | Desired attitude (roll, pitch, yaw) + collective thrust | Direct motor mixing |

Option A is recommended because it provides a natural interface — the RL agent thinks in terms of "where do I want to accelerate" and a well-tuned inner-loop controller handles the attitude tracking.

### Architecture

```
RL Policy → [a_des(3), ψ̇_des] → Attitude Controller → [τ(3), T] → Motor Mixing → [ω₁², ω₂², ω₃², ω₄²]
```

### Files to add/modify

| File | Change |
|---|---|
| **[NEW]** `models/multicopter_6dof.py` | Full 6-DOF dynamics, motor model, drag model |
| **[NEW]** `models/attitude_controller.py` | PD attitude controller (inner loop) |
| [config.yaml](file:///Users/eren/MacbookAir/GIT/Image_based_Visual_Servoing_RL/config.yaml) | Add mass, inertia matrix, motor params, drag coefficients |
| [interception_env.py](file:///Users/eren/MacbookAir/GIT/Image_based_Visual_Servoing_RL/envs/interception_env.py) | Replace InterceptorDrone with Multicopter6DOF + AttitudeController |

### Training strategy
- **Warm-start from Stage 3a** (the observation and action semantics are similar — both command accelerations)
- Inner-loop attitude controller must be **well-tuned before RL training begins** (test it with step responses)
- Train for 20-50M timesteps
- Consider domain randomization on mass and inertia

### Success criterion
- Success rate > 40% (very challenging)
- Agent learns to balance aggressive pursuit with camera stability
- Pitch oscillations should be bounded (not spinning out of control)

---

## Stage 4a: CBF-HardNet Safety Filtering

### Goal
Add a safety layer that modifies the RL policy's actions to guarantee field-of-view preservation, attitude limits, and actuator constraints.

### Safety constraints (CBF functions)

1. **Horizontal FOV preservation**:
```
h_hfov(x) = α_hfov/2 - |arctan(x_c / z_c)|  > 0
```

2. **Vertical FOV preservation**:
```
h_vfov(x) = α_vfov/2 - |arctan(y_c / z_c)|  > 0
```

3. **Maximum pitch angle**:
```
h_pitch(x) = θ_max - |θ|  > 0     (e.g., θ_max = 45°)
```

4. **Maximum roll angle**:
```
h_roll(x) = φ_max - |φ|  > 0      (e.g., φ_max = 45°)
```

### CBF-QP formulation
```
u_safe = argmin_u ||u - u_RL||²
subject to: ḣ_i(x, u) + α_i h_i(x) ≥ 0    for all i
```

### HardNet (learned safety filter)
Instead of solving the QP online (which can be slow and requires an analytical model), HardNet is a neural network trained to approximate the QP solution:

```
u_safe = H_ψ(x̂, u_RL)
```

**HardNet training procedure**:
1. Run the trained RL policy from Stage 3b to collect trajectories
2. At each state-action pair, solve the CBF-QP to get the ground-truth safe action
3. Train HardNet supervised: minimize `||H_ψ(x, u_RL) - u_QP*||²`
4. Fine-tune the RL policy with HardNet in the loop

### Files to add

| File | Change |
|---|---|
| **[NEW]** `safety/cbf_constraints.py` | Define h_i(x) functions and their gradients |
| **[NEW]** `safety/cbf_qp_solver.py` | QP solver for ground-truth safe actions |
| **[NEW]** `safety/hardnet.py` | Neural network safety filter |
| **[NEW]** `safety/hardnet_trainer.py` | Supervised training script for HardNet |

### Training strategy
1. First train the CBF-QP solver (no learning — just optimization)
2. Collect a dataset of (x, u_RL, u_QP*) tuples (~100K samples)
3. Train HardNet supervised on this dataset
4. Fine-tune the RL policy with HardNet in the loop for 5-10M steps

### Success criterion
- Safety violation rate < 1% (FOV loss events)
- Success rate within 15% of Stage 3b (safety filter may reduce aggressiveness)
- HardNet inference time < 1ms per step

---

## Stage 4b: Realistic Perception and Environmental Disturbances

### Goal
Replace the perfect point-target detection with noisy, intermittent visual detection and add environmental disturbances.

### What changes

**Noisy detector output**:
```
z_k = [ū_k, v̄_k, w_box, h_box]ᵀ + n_det     with probability p_detect
z_k = ∅                                         with probability (1 - p_detect)
```

**Detection probability model** (decreases with distance and angle):
```
p_detect = sigmoid(β₁ / d_rel - β₂ * θ_off_axis - β₃)
```

**Wind disturbance model**:
```
v̇ = g + (1/m)(f + f_drag + f_wind)
f_wind = Dryden turbulence model or Ornstein-Uhlenbeck process
```

### Observer upgrade
The DKF from Stage 2b must be upgraded to handle:
- Missing measurements (prediction-only mode when z_k = ∅)
- Bounding box size as a weak depth cue
- IMU propagation during detection gaps

### Training strategy
- **Warm-start from Stage 4a**
- Use domain randomization on wind parameters, detection probability, noise levels
- Train for 10-20M timesteps with randomized conditions

### Success criterion
- Success rate > 30% under worst-case conditions
- Graceful degradation: performance decreases smoothly as conditions worsen
- System recovers from temporary detection loss (< 0.5s gaps)

---

## Summary: Complete Roadmap

| Stage | Challenge Introduced | Obs Dim | Action | Train From | Expected Success | Timesteps |
|---|---|---|---|---|---|---|
| **1a** ✅ | Pipeline validation | 18 | Velocity (4) | Fresh | 100% | 1M |
| **1b** | Action smoothing, inertia-lite | 22 | Acceleration (4) | Fresh | >90% | 3-5M |
| **2a** | Remove ground-truth obs | 13 | Acceleration (4) | Fresh | >70% | 5-10M |
| **2b** | Noise + delay + DKF | 13 | Acceleration (4) | Warm (2a) | >60% | 2-3M |
| **3a** | First-order dynamics, pitch coupling | 15 | Acceleration (4) | Fresh | >50% | 10-20M |
| **3b** | Full 6-DOF + attitude controller | 15+ | Desired accel (4) | Warm (3a) | >40% | 20-50M |
| **4a** | CBF-HardNet safety filter | 15+ | Safe accel (4) | Fine-tune (3b) | >35% | 5-10M |
| **4b** | Noisy detection, wind | 15+ | Safe accel (4) | Warm (4a) | >30% | 10-20M |

> [!IMPORTANT]
> **Total estimated training budget: ~70-120M timesteps** across all stages. At the current speed (~6000 fps with 16 envs on CPU), 10M steps takes ~28 minutes. The full project would require approximately 5-6 hours of total training time.

---

## Recommended Execution Order

### Phase 1: Fix the motion (Stages 1b)
**Start here.** The zigzag behavior is the most visible problem. Fixing it first gives us physically plausible trajectories that we can trust for all subsequent stages.

### Phase 2: Fix the observation (Stages 2a → 2b)
**This is the intellectual core of the project.** Removing ground-truth 3D information transforms the problem from trivial pursuit to genuine visual servoing. The DKF comparison (2b) provides a clean experimental result for the thesis.

### Phase 3: Fix the dynamics (Stages 3a → 3b)
**This is the engineering core.** Building a proper 6-DOF simulator and attitude controller requires careful implementation and testing of the inner-loop controller before RL training can begin.

### Phase 4: Add safety and realism (Stages 4a → 4b)
**This is the research contribution.** The CBF-HardNet safety filter is the novel component that differentiates this work from standard RL visual servoing.

---

## Open Questions

> [!WARNING]
> The following decisions need your input before we proceed:

1. **Should we start with Stage 1b (action smoothing) or jump to Stage 2a (remove ground-truth)?** Stage 1b fixes the zigzag but is a smaller step. Stage 2a is harder but more impactful for the thesis.

2. **Policy architecture for Stage 2a onwards**: Should we switch from `MlpPolicy` to an **LSTM-based recurrent policy** (`RecurrentPPO` from SB3-contrib)? The agent will need memory to infer depth from temporal image changes.

3. **Scope**: Are you targeting all stages for this project, or is there a deadline that limits how far we can go? This affects how much time we spend on each stage.
