# Scripts

This directory contains runnable support tools, separated by evidential role.
None of the files under `legacy/` defines the canonical experiment protocol.

## Layout

- `diagnostics/` contains focused engineering checks. They generate plots and
  measurements, but they are not pytest tests and are not publication results.
- `legacy/` contains historical evaluation, video, reporting, and training
  launchers retained to reconstruct earlier work.
- `legacy/interactive_eval.py` is the implementation behind the root `eval.py`
  compatibility shim; it produces plots, not canonical JSONL evidence.
- `legacy/training/` contains the old HardNet launch recipes. They now use
  `${PYTHON:-python3}`, resolve the repository root from their own location,
  and explicitly select `configs/legacy/stage3_stage4.yaml`.
- `legacy/reporting/` regenerates figures for historical drafts.

Run a Python tool from any working directory by giving its path, for example:

```bash
python /path/to/repo/scripts/diagnostics/dkf_diagnostic.py --help
python /path/to/repo/scripts/legacy/eval_evasion.py --help
```

Historical tools may require optional safety or reporting dependencies and a
compatible legacy checkpoint. Use the canonical evaluation entry point for new
claims; do not combine legacy metrics with canonical result records.
