# Configuration Directory

Configuration files are separated by evidence level rather than by filename age.

- `canonical/` contains the frozen, executable project contract. New baseline
  training and evaluation should start from
  `canonical/fixed_camera_intercept_v1.yaml`.
- `legacy/` contains historical and exploratory configurations retained for
  interpreting old checkpoints and reports. They are not canonical evidence and
  may rely on older command-line behavior or wrapper combinations.
  - `legacy/stage3_stage4.yaml` is the historical point-mass/6-DOF Stage 3–4
    superset used by old HardNet and robustness scripts.
  - `legacy/equal_capability_evasion.yaml` belongs to the exploratory
    equal-capability target branch.

Every new run should copy its fully resolved configuration into its own run
directory and record the source configuration hash in a manifest. A checkpoint
without that snapshot must be reported as having unknown provenance; do not
reconstruct its configuration from a directory name.
