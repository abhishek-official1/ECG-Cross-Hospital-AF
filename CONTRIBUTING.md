# Contributing to the ECG study

This repository publishes the analysis pipeline and dashboard source. The ECG recordings and trained weights stay local.

For dashboard changes, install the development requirements and run:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/test_dashboard.py
```

Tests that require saved experiment bundles skip on a fresh clone. Use synthetic 12-lead arrays for new public tests. Do not commit raw ECGs, record-level tables, prediction files, or checkpoints. If you change a metric or cohort definition, show the source run and the matching denominator in the pull request; keep exploratory Georgia results labeled as exploratory.

Report ordinary bugs with a small reproduction. See [SECURITY.md](SECURITY.md) for a private security report.
