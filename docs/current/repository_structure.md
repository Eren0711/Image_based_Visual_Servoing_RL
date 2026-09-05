# Repository structure

The repository is organized by role so current work, reusable implementation,
and historical evidence are not confused.

```text
configs/
  canonical/          frozen MVP configuration
  legacy/             configurations used by historical runs
artifacts/             ignored manifest-backed runs/evaluations and local media
docs/
  current/            current engineering and research documentation
  legacy/studies/     preliminary study write-ups
  legacy/raw/         preserved raw evaluation text
  archive/            superseded plans and development history
envs/                  environment and observation wrappers
models/                interceptor and target dynamics
observers/             estimation components
safety/                historical safety-layer implementations
runtime/               shared run construction, seeding, and provenance
evaluation/            canonical evaluation protocol and schemas
experiments/
  registry.yaml       experiment index
  legacy/             exploratory one-off experiment drivers
scripts/
  diagnostics/        manual engineering diagnostics (not pytest tests)
  legacy/             historical launch, evaluation, video, and report tools
tests/                  automated regression and contract tests only
logs/                   preserved checkpoints and training artifacts
results/                preserved/generated experiment outputs
report/                 historical manuscript sources and figures
references/             technical notes and theoretical analysis
```

## Placement rules

New reusable behavior belongs in the core packages (`envs`, `models`,
`observers`, `runtime`, or `evaluation`) with an automated test. A manual tool
belongs in `scripts/diagnostics`. A one-off comparison belongs in
`experiments/` and must be registered. Superseded code that is retained only to
reconstruct an old result belongs under `legacy/`.

Checkpoints and existing result artifacts were deliberately left in place.
Moving or renaming the importable core packages can break Stable-Baselines3
checkpoint deserialization because custom policy class paths are stored in the
archive.
