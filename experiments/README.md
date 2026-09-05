# Experiments

`registry.yaml` is the index of planned, canonical, and historical studies.
The canonical experiment configuration lives under `configs/canonical/`.

`legacy/` contains exploratory one-off studies that were previously at the
top of this directory:

- `agility_ablation.py`
- `lead_pursuit_eval.py`
- `lead_pursuit_retrain.py`

These scripts are preserved for reconstruction and now resolve repository
paths independently of the caller's working directory. Their outputs under
`results/` remain historical evidence candidates; they are not automatically
promoted to canonical thesis or publication results.
