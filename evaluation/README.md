# Canonical evaluation

This package is the only evaluation path intended for new quantitative claims.
It runs the same seed once for each canonical target mode, writes one validated
JSON object per terminal episode, and derives `summary.json` only by reopening
that JSONL file.

The committed suites have different roles:

- `suites/validation_v1.yaml`: 50 paired seeds (150 episodes) for development
  and checkpoint selection.
- `suites/test_v1.yaml`: 200 held-out paired seeds (600 episodes) for one final
  evaluation after the method and selection rule are frozen.

Do not tune on `test_v1`. The suite loader rejects duplicate seeds, and the
test suite verifies that validation and test banks are disjoint.

Example:

```bash
python -m evaluation \
  --model artifacts/runs/<run-id>/models/ibvs_ppo_final.zip \
  --suite evaluation/suites/validation_v1.yaml \
  --output artifacts/evaluations/<evaluation-id>
```

The output directory must not already exist. It contains:

- `manifest.json`: command, Git state, dependencies, config/model hashes,
  method identity, wrapper profile, suite hash, lifecycle status and duration;
- `resolved_config.yaml` and `resolved_suite.yaml`: immutable snapshots;
- `episodes.jsonl`: versioned episode records with identities, seeds, outcome,
  interception, FOV, attitude and projection-availability metrics;
- `summary.json`: aggregate and per-target-mode statistics calculated from the
  JSONL records.

`eval.py` and files under `scripts/legacy/` are historical visualization or
reconstruction tools. Their metrics are not interchangeable with this schema.
