# Research Spine: Fixed-Camera Agile Drone Interception

**Status:** Phase 1 reproducibility foundation complete; Phase 2 ready
**Last updated:** 2026-09-05

This document is the single source of truth for the research question, scope,
claims, experiments, and roadmap. Historical plans, reports, and exploratory
notes remain useful evidence, but they do not override this document.

## 0. Frozen canonical system (`fixed_camera_intercept_v1`)

The executable source of truth is
`configs/canonical/fixed_camera_intercept_v1.yaml`. New development must use
that file unless an experiment explicitly declares itself exploratory.

| Contract item | Frozen v1 decision |
|---|---|
| Interceptor | `multicopter_6dof` (the repository's 6-DOF-lite model) |
| Target | `sixdof`, fixed at `v_max=10 m/s` and `a_max=5 m/s²` in v1 |
| Camera | Rigid, forward-facing, monocular ideal pinhole point-feature camera |
| Target modes | Exactly `cruise`, `steady_turn`, `weave` |
| Base observation | Fixed 16-D ordered feature/state vector |
| Action | Normalized 4-D body-acceleration/yaw-rate command |
| Default pipeline | Clean feature-level PPO; no optional wrapper or safety layer |
| Success | Relative distance `<= 2.0 m` |
| Envelope failure | `|pitch| > 35 deg` or `|roll| > 35 deg` for one step |
| FOV failure | 15 consecutive geometrically invisible frames (`0.30 s`) |
| Timeout | 500 steps (`10.0 s`) |

Terminal precedence on the same simulation step is: success, flight-envelope
violation, FOV loss, then timeout. The target maneuver is sampled uniformly
from the three-mode list at reset. Initial target range is 10–30 m and initial
target speed is 20–50% of its configured maximum.

The envelope terminal has an explicit `-100` reward adjustment. This prevents
the policy from learning to violate the attitude envelope merely to terminate
early and avoid future per-step costs; it is part of the frozen v1 reward.

At reset, the interceptor is at the EFCS origin with yaw uniform on
`[-pi,pi]`. The target bearing is sampled from the existing normalized
body-frame box (`x~U(0.5,1)`, `y,z~U(-0.3,0.3)`) and accepted only with a
geometric FOV margin above `0.2`; its velocity direction is isotropic in 3-D.

The default target uses the same dynamics implementation as the interceptor
but not the same capability limits: interceptor `v_max=15 m/s`, `a_max=10
m/s^2`; target `v_max=10 m/s`, `a_max=5 m/s^2`. Equal-capability experiments
are outside canonical v1.

The action's first three limits are component-wise desired body-acceleration
limits, not a bound on the three-vector norm and not realized instantaneous
acceleration. The camera produces normalized ideal point features, not pixels,
bounding boxes, distortion, target size, or occlusion.

Ordered base observation contract:

| Indices | Signal | Normalization/source |
|---|---|---|
| `0:2` | image-plane error | divided by horizontal/vertical half-FOV tangent |
| `2:4` | image-plane velocity | finite difference divided by `10` and clipped |
| `4` | target visibility | geometric Boolean (`0` or `1`) |
| `5` | estimated optical-axis depth | divided by `50 m`, clipped to `[0,1]` |
| `6:9` | interceptor body velocity | divided by interceptor `v_max` |
| `9:12` | roll, pitch, yaw | roll/pitch by limits; yaw by `pi` |
| `12:16` | previous normalized action | already in `[-1,1]` |

The depth estimator is initialized from simulator depth in the current
implementation; therefore v1 is image-guided simulation, not a deployable
monocular-ranging claim. HardNet's additional privileged 20-D safety context
is outside the base MVP observation.

Ordered action contract:

| Indices | Command | Mapping |
|---|---|---|
| `0:3` | desired body-frame accelerations | each component times `10 m/s^2` |
| `3` | desired yaw rate | times `pi/4 rad/s` |

The retained camera extrinsic maps body `+x` to camera `+z` (optical axis),
body `+y` to camera `+y`, and body `+z` to camera `-x`.

Explicitly excluded from v1:

- raw-image processing or detector training;
- physical collision, capture, or damage modelling;
- an equal-agility impossibility theorem;
- a gimballed camera;
- LSTM/recurrent policies;
- adversarial multi-agent training;
- CBF/HardNet in the initial clean interception MVP.

## 1. North-star question

> Can the frozen 6-DOF-lite interceptor, using only the canonical 16-D
> image-guided observation, learn to reach within `2 m` of a `cruise`,
> `steady_turn`, or `weave` target while retaining visibility and respecting
> its flight envelope?

This is the immediate clean-MVP question. After it is answered reproducibly,
the broader thesis asks how the operating envelope changes with sensing delay,
missed detections, wind, target capability, and initial geometry. The project
does not claim that RL can catch every possible FPV drone; it must compare
learned guidance fairly with classical guidance and explain why and where each
method fails.

The canonical simulator defines an interception as relative distance at or
below `d_success = 2 m`. It does not model physical contact, collision mechanics, a
capture device, or damage.

## 2. Thesis and paper scopes

### Master's thesis

Working title:

**Safe Image-Guided Reinforcement Learning for Fixed-Camera Agile UAV
Interception: Method Comparison and Operating-Envelope Analysis**

The thesis may cover the complete system:

- fixed-camera partial observation;
- learned guidance;
- state/depth estimation;
- external and in-policy safety mechanisms;
- perception and wind disturbances;
- target-maneuver and agility operating envelopes;
- failure analysis and a simulation-to-hardware roadmap.

### First paper

Publication is optional and is not a development gate. This scope becomes
relevant only after the clean interception MVP and robustness work succeed.

Working title:

**Feasibility-Regularized Differentiable Safety Projection for Image-Guided
Agile Drone Interception**

The first paper should be narrower than the thesis. Its central question is:

> Does feasibility regularization mitigate the optimization difficulty caused
> by frequently active differentiable safety projections in PPO-based visual
> interception?

Equal-agility impossibility claims, gimballed cameras, recurrent policies,
raw-image detector training, multi-agent pursuit, and hardware capture are not
part of this first paper.

## 3. Precise current problem formulation

The task is a constrained, partially observed reach-avoid problem.

- **Goal set:** relative range is at most `2.0 m`.
- **Failure conditions:** the exact FOV, attitude and timeout conditions frozen
  in Section 0.
- **Base policy observation (16-D):** normalized image-plane position and
  velocity, visibility, estimated depth, interceptor body velocity, attitude,
  and previous action.
- **Future HardNet-only context (additional 20-D, outside MVP):** affine
  proxy-CBF coefficients computed from simulator state by the safety supervisor.
- **Action (4-D):** body-frame acceleration commands and yaw-rate command.
- **Dynamics:** only the frozen 6-DOF-lite interceptor and 6-DOF-lite target
  configurations are canonical v1. Kinematic and point-mass models are legacy
  tools.
- **Default sensing:** clean simulated point features. Noise/delay, missed
  detections and wind are later robustness profiles.

This is currently **image-guided state-based RL**, not end-to-end raw-pixel RL:

- the detector itself is not trained;
- proprioceptive states are included;
- the reward uses privileged ground-truth range;
- the depth estimator is initialized from simulator information;
- the safety supervisor uses privileged simulator state;
- all reported evidence is simulation-only.

These are acceptable thesis assumptions if they are stated explicitly. They
must not be hidden behind the phrase "vision-only".

## 4. Research questions and testable hypotheses

### RQ1 — Learned versus classical guidance

Can PPO improve interception performance over a classical controller when both
receive equivalent measurements and act through the same command interface?

**H1:** RL will be comparable on simple trajectories and more successful on
the trained maneuver distribution, but may generalize poorly outside it.

### RQ2 — Safety architecture

How do reward-only PPO, external CBF filtering, and differentiable in-policy
projection trade interception success against FOV and attitude violations?

**H2:** Safety layers will reduce violation severity and duration, with a
measurable success or control-effort cost.

### RQ3 — Projection and learnability

Does an active in-policy projection attenuate useful policy gradients, and can
feasibility regularization improve learning?

**H3:** Feasibility regularization will reduce raw-to-safe action distance and
projection activity and improve held-out performance. Jacobian and gradient
diagnostics are required before claiming a causal mechanism.

### RQ4 — Estimation and sensing

How much do the DKF/depth estimator help under noise, delay, and missed
detections?

**H4:** Estimation will matter more under degraded sensing than under clean
sensing.

### RQ5 — Operating envelope

How do target maneuver, target/interceptor capability ratio, sensing severity,
and initial geometry change the probability of interception?

**H5:** The system will exhibit a bounded empirical capture envelope. Failure
outside that measured envelope is not, by itself, a proof of geometric
impossibility.

## 5. Contribution hierarchy

Potential thesis contributions, in order of defensibility:

1. A reproducible constrained reach-avoid benchmark for fixed-camera aerial
   interception.
2. A controlled comparison of classical guidance, reward-only PPO, external
   CBF filtering, and differentiable in-policy safety projection.
3. An empirical study of projection activity, policy optimization, and
   feasibility regularization.
4. An operating-envelope and failure-mode map over maneuver, sensing, and
   capability ratios.

PPO, DKF, CBF, Dykstra projection, and SO(3) control are established tools.
Using them is not a contribution by itself. Unless a new result is proved, the
thesis should be described as model-informed and primarily empirical.

## 6. Future scientific method matrix

This matrix is not the immediate development target. First, the clean PPO
interception MVP must be reproducible on the frozen system. When method
comparison begins, every experiment must use the same dynamics, observations,
initialization bank, target realizations, evaluation budget, and metric
implementation.

| ID | Method | Purpose | Current state |
|---|---|---|---|
| `M0` | Classical IBVS + the same estimator | Essential non-learning baseline | Missing |
| `M1` | PPO without a safety layer | Learned-guidance baseline | Implemented |
| `M2` | PPO + external proxy-CBF/QP | Action-filter baseline | Implemented; setup and terminology need repair |
| `M3` | PPO + in-policy projection, `lambda_feas = 0` | Isolate projection effect | Implemented |
| `M4` | PPO + in-policy projection + feasibility loss | Proposed method | Implemented, preliminary evidence |

A proportional-navigation or privileged-state oracle may be added later to
separate perception limitations from guidance limitations. It is not a
substitute for `M0`.

Do not add new policy architectures until the clean PPO MVP works end to end.
After that, `M0`–`M4` must run through one canonical evaluation harness.

## 7. Canonical evaluation design

### Scenario axes

- **Target maneuver:** canonical v1 uses only cruise, steady turn and weave.
  Reactive/mixed maneuvers are deferred operating-envelope studies.
- **Sensing:** clean, nominal, severe.
- **Capability ratio:** vary target speed and acceleration relative to the
  interceptor; do not call a condition "equal agility" unless all compared
  limits are defined explicitly.
- **Initial geometry:** range, bearing, heading, and relative velocity.

The first publishable matrix should be deliberately small: `M0`–`M4` under
clean, nominal, and severe sensing on a fixed set of representative maneuvers.
The dense capability sweep belongs to the thesis after this matrix is stable.

### Primary outcomes

- interception success rate;
- FOV-loss rate;
- attitude-limit violation rate, severity, and duration;
- timeout rate.

### Secondary outcomes

- time to interception;
- minimum range and terminal closure rate;
- FOV retention;
- control effort and jerk;
- projection frequency and intervention norm;
- maximum constraint residual and infeasible-set rate;
- policy/projection inference time;
- projection Jacobian singular values or effective rank for the mechanism
  study.

### Statistical protocol

1. Use at least five independent training seeds for the core study; ten is
   preferable if compute permits.
2. Select exactly one checkpoint per training seed using a separate validation
   set and a rule fixed before test evaluation.
3. Evaluate all methods on the same episode seed bank and target trajectories.
4. Keep validation and test seeds separate.
5. Use 100–200 paired test episodes per condition and report 95% confidence
   intervals.
6. Treat training seeds, not checkpoints, as independent samples.
7. Report deterministic and stochastic policies separately.
8. Never combine the best nominal number from one policy with the best severe
   number from another policy as if they belonged to one model.

## 8. Safety claim boundary

The current implementation projects the Gaussian policy mean, not every
stochastic action sample. It also uses a finite-iteration projection and a
relative-degree-1 proxy model whose feasible set may be empty.

Therefore, the current code does **not** support the statement "every executed
action is guaranteed safe." Until the algorithm is corrected and verified,
use language such as:

> The deterministic policy mean is approximately projected onto the proxy-CBF
> action set when that set is nonempty, subject to numerical tolerance and
> model mismatch.

Required safety diagnostics:

- project and verify the actual executed sample, or explicitly restrict the
  claim to deterministic evaluation;
- log pre- and post-projection residuals;
- detect and report empty/inconsistent constraint sets;
- distinguish proxy-CBF margin violation from physical FOV/attitude violation;
- add unit tests for feasible, boundary, infeasible, and stochastic cases.

## 9. Current claim ledger

| Claim | Status | Action |
|---|---|---|
| PPO can intercept some simulated maneuvering targets with a fixed camera | Supported within tested simulation settings | Preserve and reproduce |
| Feasibility loss improves some HardNet training outcomes | Preliminary | Re-run with fair checkpoint selection and confidence intervals |
| In-policy projection restores full-rank gradients | Not established | Measure Jacobians/gradients; weaken wording |
| HardNet guarantees every executed action is safe | Not supported by current stochastic policy | Fix algorithm or narrow claim |
| The proposed policy beats external CBF filtering | Not established statistically | Paired core experiment |
| Fixed-camera equal-agility interception is geometrically impossible | Not established; current derivation is unsuitable | Keep exploratory and reformulate |
| The system is vision-only | False as currently implemented | Use "image-guided with proprioception and privileged safety context" |
| The method catches real FPV drones | Not tested | Reserve for future hardware validation |

## 10. Repository ownership map

The Phase-1 inventory was completed before source reorganization. No checkpoint
or result artifact was deleted or relocated. Core import paths were retained
because historical SB3 archives serialize names such as
`safety.hardnet_policy.HardNetActorCriticPolicy`.

### Canonical entry points

- `README.md` — concise project entry point and honest status.
- `PROJECT.md` — this research spine and decision record.
- `experiments/registry.yaml` — machine-readable status of the canonical
  method matrix and exploratory evidence.
- `configs/canonical/fixed_camera_intercept_v1.yaml` — frozen executable system
  and task definition for new work.
- `train.py` — manifest-backed training entry point.
- `evaluate.py` / `python -m evaluation` — canonical paired evaluation entry.
- `runtime/` — shared environment construction, seed and provenance utilities.
- `evaluation/suites/` — versioned validation and held-out test seed banks.
- `envs/`, `models/`, `observers/`, `safety/` — current implementation.

### Evidence and working notes

- `docs/legacy/studies/hardnet_*.md` — useful preliminary HardNet evidence,
  not final tables.
- `report/master_report.tex` — detailed development record.
- `report/journal_paper.tex` — paper draft requiring claim/statistics revision.
- `results/*/raw_results.json` — exploratory episode records where available.

### Exploratory branch

- `configs/legacy/equal_capability_evasion.yaml`;
- `experiments/legacy/agility_ablation.py`;
- `experiments/legacy/lead_pursuit_*.py`;
- `results/agility_ablation/`, `results/lead_pursuit*/`;
- `references/geometric_analysis*` and `results/geometric_analysis/`.

These files investigate operating-envelope limits. They are not evidence for
the first paper until their assumptions and protocol are repaired.

### Historical or transitional material

- `docs/archive/implementation_plan.md` — development history, not current status.
- `configs/legacy/stage3_stage4.yaml` — historical Stage 3/4 configuration.
- `configs/legacy/equal_capability_evasion.yaml` — exploratory equal-capability
  configuration.
- `scripts/legacy/training/run_hardnet_*.sh` — portable but non-canonical old
  recipes retained for reconstruction.
- `eval.py`, `visualize.py`, and `scripts/legacy/eval_*.py` — historical
  plotting/reconstruction tools; they do not define canonical metrics.
- `logs/stages/*` — local artifacts, not an experiment database.

## 11. Reproducibility contract

Every future training run must save:

- immutable resolved configuration;
- Git commit and dirty-state flag;
- dependency/environment manifest;
- training seed and every wrapper/subsystem seed;
- training budget and wall-clock/runtime information;
- checkpoint-selection rule;
- model hash.

Every validation/test run must use a versioned suite, link the training model
by hash, and save per-episode records plus a summary derived only from those
records.

Every evaluation record must contain at least:

- experiment, method, model, condition, and episode IDs;
- initialization/trajectory seed;
- terminal outcome and termination reason;
- success, minimum range, intercept time, and terminal closure rate;
- FOV and attitude violation counts, duration, and maximum severity;
- projection activity, intervention norm, residual, and infeasibility flag
  when a projection method is active; otherwise explicit not-applicable or
  unavailable fields;
- software/config/model provenance.

Plots and paper tables must be generated from these machine-readable records.
Numbers must not be copied manually into figure scripts.

## 12. Phased roadmap

### Phase 0 — Freeze target and scope (complete)

- freeze the exact system, task, input/output and outcome contract;
- make one canonical config the default for new work;
- label old configs and equal-agility/lead-pursuit work non-canonical;
- encode the scope contract in automated checks.

**Exit criterion:** `PROJECT.md`, the canonical config and runtime agree on
the decisions in Section 0. No optional method is silently part of the MVP.

### Phase 1 — Inventory and reproducibility foundation (complete)

- inventory historical runs and checkpoint provenance without deleting them;
- define one environment builder and episode-result schema;
- repair dependency installation and deterministic seed propagation;
- add run manifests with resolved config, seeds and hashes.

**Exit criterion met:** automated tests replay the base environment and the
full stochastic wrapper stack from the same config/scenario seed. New runs use
immutable directories, resolved configs, namespaced seeds and manifests; the
canonical evaluator writes versioned JSONL records for disjoint paired suites.
Acceptance evidence and the audited artifact counts are recorded in
`docs/current/phase1_completion.md`.

### Phase 2 — Correctness tests and clean PPO interception MVP (next)

- extend the existing infrastructure tests with dynamics, camera, reward,
  termination and oracle correctness tests;
- add an oracle controller for feasibility diagnosis;
- train PPO through `cruise -> steady_turn -> weave` without adding new scope;
- select checkpoints on `validation_v1`, then evaluate the frozen pipeline once
  on the untouched `test_v1` scenarios.

**Exit criterion:** clean PPO interception meets development thresholds fixed
before training and is reproducible.

### Phase 3 — Classical baseline and robustness

- implement classical IBVS with the same estimator and actuator interface;
- add noise, delay, dropout and wind one at a time;
- build a common evaluation harness and empirical operating envelope.

**Exit criterion:** failures can be attributed to guidance, sensing or physical
capability rather than inconsistent evaluation.

### Phase 4 — Safety methods

- correct and instrument the safety projection semantics;
- compare external CBF, in-policy projection and feasibility regularization;
- measure intervention, residual, infeasibility and gradient diagnostics.

**Exit criterion:** every safety/learning claim has a direct metric and a
control condition.

### Phase 5 — Optional publishable study

- train a pre-registered multi-seed method matrix with equal budgets;
- evaluate on held-out paired episodes with uncertainty;
- generate every table and figure from canonical raw records.

### Phase 6 — Thesis operating envelope and SITL/HITL path

- sweep target/interceptor capability ratios and initial geometry;
- use classical/oracle methods to diagnose failure sources;
- reformulate geometric analysis using valid pursuit and camera-body geometry;
- progress through SITL and HITL before controlled flight validation.

## 13. Suggested thesis structure

1. Motivation and research questions
2. Related work: IBVS, pursuit guidance, aerial RL, safe RL, and CBFs
3. Constrained partially observed reach-avoid formulation
4. Simulation, camera, estimator, disturbances, and target maneuvers
5. Classical and learned guidance baselines
6. External and differentiable in-policy safety mechanisms
7. Core experiments and component ablations
8. Operating-envelope and failure analysis
9. Limitations and simulation-to-real roadmap
10. Conclusions

## 14. Immediate decision

The project will not be restarted from zero. The existing simulator, PPO
pipeline, estimator, CBF code, HardNet implementation, and preliminary results
are retained. Canonical v1 prioritizes a clean, reproducible PPO interception
MVP. Safety-method comparisons and publication work follow only after that
milestone; the next work is correctness/oracle diagnosis and clean PPO
training, not another feature branch.
