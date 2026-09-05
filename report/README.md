# Report Directory

This directory contains historical LaTeX reports, generated figures, auxiliary
LaTeX files, and compiled PDFs. It is an archive of the project's written
development, not the source of truth for the current executable protocol.

Use `PROJECT.md`, the canonical configuration, run manifests, and canonical
evaluation records when checking a claim. Report text may describe experiments
whose exact command, seed, wrapper stack, or checkpoint provenance was not
recorded. In particular, Stage 4b/HOCBF wording must remain marked ambiguous
until primary run metadata resolves it.

The optional report dependencies are installable with:

```bash
python -m pip install -e '.[reports]'
```
