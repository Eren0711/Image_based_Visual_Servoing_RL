# Safe RL for Fixed-Camera Agile Drone Interception

This repository studies whether a quadrotor with a rigid, forward-facing
monocular camera can learn image-guided interception of an aggressively
maneuvering drone while maintaining visual lock and respecting flight-envelope
constraints.

The project is a **simulation research prototype entering its clean-MVP phase**. It
already contains a working PPO/Gymnasium pipeline, camera and estimator models,
6-DOF-lite dynamics, synthetic sensing and wind disturbances, external CBF
filtering, differentiable in-policy projection, and multi-seed exploratory
experiments. Phase 1 now provides reproducible run/evaluation infrastructure,
but the project is not yet a publication-grade result set or a real-drone
capture system.

The canonical research question, claim boundaries, experiment matrix, and
roadmap live in **[PROJECT.md](PROJECT.md)**. Read that document before using
historical plans or result notes.
The completed consolidation audit is summarized in
**[docs/current/phase1_completion.md](docs/current/phase1_completion.md)**.

The frozen executable system definition is
**[`configs/canonical/fixed_camera_intercept_v1.yaml`](configs/canonical/fixed_camera_intercept_v1.yaml)**.
Historical configurations now live under `configs/legacy/`; neither defines
the current task.

## Research question

> Under what target maneuvers, sensing conditions, initial geometries, and
> relative vehicle capabilities can a fixed-camera interceptor learn safe
> image-guided interception, and how does the safety architecture affect both
> performance and learnability?

The broad question is intended for a master's thesis. A possible later paper
would be narrower: feasibility-regularized differentiable safety projection
for PPO interception.

## What the current system is

- A constrained, partially observed reach-avoid simulation.
- A fixed forward camera with image-plane target measurements.
- Image-guided state-based RL using visual features, estimated depth, and
  interceptor proprioception.
- Body-frame acceleration and yaw-rate guidance commands.
- Kinematic and 6-DOF-lite interceptor dynamics.
- Point-mass and 6-DOF-lite target models with multiple maneuver families.
- Optional noise, delay, missed detections, DKF filtering, and wind.
- Reward-only PPO, external proxy-CBF filtering, and in-policy differentiable
  projection with an optional feasibility loss.

Those alternatives remain in the repository for history and later studies.
Canonical v1 fixes both vehicles to the 6-DOF-lite model, uses only
`cruise`/`steady_turn`/`weave`, and starts with clean PPO without a safety
layer.

## What it is not yet

- End-to-end raw-image learning or detector training.
- A certified safety controller for the true 6-DOF dynamics.
- A hardware/HITL-validated interception system.
- A physical contact or capture model.
- Evidence that every FPV target can be intercepted.
- Proof that fixed-camera equal-agility interception is impossible.

## Repository superset architecture

```text
target + interceptor dynamics
          |
          v
fixed pinhole camera
          |
          v
noise / delay / missed detections
          |
          v
DKF + depth estimation
          |
          +--------------------+
          |                    |
          v                    v
      PPO policy         CBF context
          |                    |
          +------ safety ------+
                    |
                    v
       body acceleration + yaw-rate command
```

The HardNet path augments the 16-D base observation with 20 affine proxy-CBF
coefficients. Those coefficients are computed by a privileged safety
supervisor; this distinction matters when interpreting "vision-based" claims.
All disturbance, DKF and safety branches in this diagram are disabled in the
canonical clean MVP profile.

## Repository guide

| Path | Role |
|---|---|
| `PROJECT.md` | Canonical research plan and claim ledger |
| `configs/canonical/fixed_camera_intercept_v1.yaml` | Frozen v1 system/task contract |
| `experiments/registry.yaml` | Experiment status and future comparison protocol |
| `train.py` | Current PPO/HardNet training entry point |
| `evaluate.py`, `evaluation/` | Paired seed-locked evaluator, schemas and suites |
| `runtime/` | Shared environment builder, seed derivation and run manifests |
| `envs/` | Interception environment and sensing/safety wrappers |
| `models/` | Camera, target, interceptor, wind, and attitude models |
| `observers/` | Interaction matrix, depth estimator, and DKF |
| `safety/` | Proxy-CBF, QP filtering, Dykstra projection, and HardNet PPO |
| `experiments/legacy/` | Exploratory agility and lead-pursuit studies |
| `scripts/` | Diagnostics plus clearly separated historical tools |
| `docs/current/` | Current structure and audited legacy-artifact inventory |
| `docs/legacy/`, `docs/archive/` | Preliminary studies, raw text and old plans |
| `results/` | Exploratory episode-level outputs and figures |
| `report/` | Master report and journal-paper drafts |
| `configs/legacy/` | Preserved Stage 3/4 and equal-capability configurations |

## Current scientific status

| Component | State |
|---|---|
| Gymnasium/PPO/6-DOF-lite pipeline | Implemented |
| DKF, depth, noise, delay, dropout, and wind | Implemented |
| External proxy-CBF and in-policy projection | Implemented |
| Feasibility-loss and lambda ablations | Preliminary experiments completed |
| Classical IBVS controller under the same interface | Missing |
| Canonical evaluation runner and result schema | Implemented in Phase 1 |
| Deterministic seed propagation | Base env plus stochastic wrapper stack verified |
| Run manifests and immutable output directories | Implemented in Phase 1 |
| Historical checkpoint provenance | Audited; incomplete/unknown for all legacy checkpoints |
| Automated tests | Phase-0/1 contract, replay, manifest and evaluator tests added |
| Statistically clean, publication-grade rerun | Missing |
| HITL or physical flight validation | Not started |

## Safety caveat

The current HardNet policy projects the Gaussian action-distribution mean, not
every stochastic sample. The projection also uses finite iterations and a
proxy model. Consequently, the repository does not currently support the claim
that every executed action is guaranteed safe. See `PROJECT.md` for the exact
claim boundary and the required repair.

## Configuration and commands

The canonical config is the default for new training. Historical point-mass
Stage 3/4 reconstruction uses `configs/legacy/stage3_stage4.yaml`; the
equal-capability branch uses
`configs/legacy/equal_capability_evasion.yaml`.

Typical current commands are:

```bash
python train.py --help
python train.py --stage <run-name> --seed <seed>
python -m evaluation --help
python -m evaluation \
  --model artifacts/runs/<run-id>/models/ibvs_ppo_final.zip \
  --suite evaluation/suites/validation_v1.yaml \
  --output artifacts/evaluations/<evaluation-id>
python -m pytest -q
```

The canonical scope guard rejects options such as `--wind`, `--hardnet`, or
`--maneuver-curriculum` unless the run is explicitly marked as a scope
override. This prevents exploratory settings from silently becoming the MVP.

Install the core project and development checks with:

```bash
python -m pip install -e ".[dev]"
```

Use `.[safety]`, `.[reports]`, or `.[all]` for the corresponding historical
tools. MP4 rendering additionally needs the system `ffmpeg` executable. Every
new training/evaluation directory stores the exact installed dependency
versions, resolved config, Git state and hashes in `manifest.json`.

## Next milestone: clean Interception MVP v1

Phase 1 has made the execution and evidence pipeline reproducible. Phase 2 is
deliberately narrower than the future publication matrix:

1. add dynamics, camera, reward and reachability/oracle correctness checks;
2. fix the development success thresholds before starting substantive training;
3. train clean PPO only on the frozen `cruise`, `steady_turn`, and `weave`
   task with multiple independent seeds;
4. select checkpoints only on `validation_v1`;
5. run the untouched `test_v1` bank only after the pipeline and selection rule
   are frozen.

Classical IBVS, robustness profiles, CBF and HardNet follow this milestone. No
new policy architecture is added before it works end to end.

## Origins

The simulator was initially inspired by *High-Speed Interception Multicopter
Control by Image-based Visual Servoing* (Yang et al., arXiv:2404.08296). The
current project is not a reproduction of that controller; a fair classical
IBVS baseline remains a required next step.
