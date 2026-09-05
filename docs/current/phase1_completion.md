# Phase 1 completion record

**Completed:** 2026-09-05

**Scope:** repository inventory, organization and reproducibility foundation

**Scientific result produced:** no

## What is now canonical

- `configs/canonical/fixed_camera_intercept_v1.yaml` remains the frozen system
  contract.
- `runtime/environment.py` is the shared environment/wrapper builder. Its
  wrapper order is Wind → intermittent detection → noise/delay → DKF → external
  CBF or CBF context.
- `runtime/seeding.py` derives independent deterministic streams from one
  master seed.
- `train.py` creates a new immutable run directory and records the resolved
  config, Git state, dependency versions, worker/subsystem seed plan, source
  checkpoint hash, final model hash, runtime and checkpoint-selection rule.
- `python -m evaluation` is the canonical result producer. It writes validated
  episode JSONL, then reopens that file to calculate the summary.

## Fixed scenario banks

| Suite | Role | Seeds | Modes | Episodes |
|---|---|---:|---:|---:|
| `validation_v1` | development/checkpoint selection | 50 | 3 | 150 |
| `test_v1` | held-out final reporting | 200 | 3 | 600 |

Every seed is paired across `cruise`, `steady_turn` and `weave`. Validation
uses 10000–10049 and test uses 20000–20199; automated checks require the banks
to be disjoint. The held-out test suite was not rolled out during Phase 1.

## Historical artifact audit

The read-only inventory found 1,593 SB3 model ZIPs (264,995,914 bytes), 1,579
unique hashes and 11 exact-duplicate groups. Observation dimensions are 689 ×
16-D, 421 × 18-D, 101 × 22-D and 382 × 36-D. A training seed is absent from
1,229 checkpoints. None has enough adjacent manifest/config evidence to claim
full provenance, so every legacy checkpoint remains `unknown`; folder names
were not converted into guessed metadata.

No checkpoint/result artifact was deleted or relocated. The `envs`, `models`,
`observers` and `safety` import paths remain in place because old HardNet
checkpoints serialize `safety.hardnet_policy`.

## Acceptance evidence

- `pytest -q`: 20 tests passed.
- The base environment and complete stochastic wrapper stack replay exactly
  after `reset(seed=...)`; namespace streams are distinct.
- Two independent 2,048-step smoke trainings with seed 42 produced
  bit-identical policy tensors and the same episode statistics. ZIP bytes may
  differ because archive metadata is not the policy state.
- A real canonical validation smoke run produced 150/150 schema-valid records,
  balanced 50/50/50 across the three modes. Repeating it produced byte-identical
  `episodes.jsonl` and `summary.json` (SHA-256 values matched).
- Training, canonical evaluation, legacy-help entry points, Python compilation,
  shell syntax, YAML parsing, inventory freshness and `git diff --check` passed.

The smoke PPO received only one SB3 rollout and achieved 0% success. That is an
expected untrained baseline and is not evidence about the interception task.

## Repository organization

- old configs: `configs/legacy/`;
- old experiments: `experiments/legacy/`;
- manual diagnostics: `scripts/diagnostics/`;
- old evaluators, launchers and figure scripts: `scripts/legacy/`;
- preliminary studies/raw text: `docs/legacy/`;
- superseded plan: `docs/archive/`;
- generated new runs: ignored `artifacts/`;
- audited legacy models/results: unchanged under `logs/` and `results/`.

## Phase 2 boundary

Phase 1 makes future evidence reproducible; it does not establish that PPO can
solve the frozen three-mode task. Phase 2 must add correctness/oracle checks,
fix development thresholds before training, train clean PPO with multiple
seeds, select checkpoints on `validation_v1`, and only then run `test_v1` once.
