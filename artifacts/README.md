# Generated artifacts

New manifest-backed training runs are written under `runs/`; canonical
evaluation commands should use `evaluations/`. These generated directories are
ignored by Git because they may contain checkpoints, TensorBoard data, JSONL
records and media.

`local_media/` holds untracked local outputs moved out of the repository root.
Historical checkpoint trees remain under `logs/` so existing paths and custom
policy imports continue to work.

Directories named `phase1-smoke-*` are infrastructure acceptance runs made
with a deliberately tiny, untrained PPO budget. Their success rate is not a
scientific result and they must never be used in a thesis table.
