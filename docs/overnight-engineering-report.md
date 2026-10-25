# Overnight engineering report

## Commits

- Starting baseline: `bc2bcf54b6cfb31e629a65f99af23d2fe7d15670` (`bc2bcf5`).
- Engineering changes: `28243b04d8b80d52f391e2048d777ee9e41ef285` (`feat: add reproducible monitoring triage demo`).
- The report is committed separately after that engineering commit. No remote updates were made.

The working tree contained the README, demo, investigator typing, and drift typing changes before
work began. Those changes were inspected, retained, verified, and committed; none were discarded.

## Delivered

- Added `scripts/run_demo.py`, a seeded synthetic walkthrough composing schema validation, KS
  drift detection, incident creation, and evidence-grounded investigation. It prints Markdown,
  optionally saves JSON, and exits with an error if injected findings are missing.
- Added integration tests in `tests/test_demo.py` and a trade-off note at
  `docs/design-notes/evidence-boundary.md`.
- Updated README quickstart, security boundaries, demo usage, and synthetic baseline results.
- Tightened drift result status typing and removed a now-unneeded investigator evaluation type
  ignore.

## Commands and results

All commands used the repository environment at `.venv\Scripts\python.exe`; the global
`python` launcher was inaccessible in this shell.

- `python -m pytest -q`: **162 passed**. Pytest emitted a cache-permission warning while trying
  to write `.pytest_cache`; tests themselves passed.
- `python -m ruff check .`: passed.
- `python -m ruff format --check .`: passed (31 files).
- `python -m mypy src scripts`: passed (15 source files).
- `python -m pip check`: passed.
- `python -m compileall -q src scripts tests`: passed.
- `git diff --check`: passed.
- Application import smoke check: passed.
- CI drift scenario and detector-method smoke commands: passed.
- `python scripts/run_demo.py`: passed; reported the injected 10-point account-age location
  shift, 12 null balance values, an unexpected `campaign_source` column, one drift incident, and
  the three expected investigation categories.
- `python scripts/evaluate_investigator.py --seed 42`: passed. Top-1 accuracy `3/3`, multi-cause
  hit-at-3 `2/2`, MRR `1.00` over five rankable cases, abstention precision/recall `5/5`, normal
  false-positive diagnoses `0/1`, evidence-reference violations `0/10`, unsupported diagnoses
  `0/2`. The category-name baseline matched these ranking metrics.
- `python scripts/evaluate_drift_scenarios.py`: passed. Seeds `42`, `123`, `2026`; 100 trials
  per seed; 200 samples; five features; alpha `0.05`. Target detections: location `300/300`,
  variance increase `56/300`, heavy tails `69/300`, skew `234/300`. No-drift false positives:
  `8/1,500` feature predictions, across `8/300` trials.
- `python -m pip_audit --local --skip-editable --progress-spinner off --timeout 60
  --cache-dir .pip-audit-cache`: no known issues in scanned installed dependencies. The scan
  skipped editable distributions, including this project, so this is not a source-code audit.

The demo and synthetic evaluations use generated data only. The evaluation measurements are
scenario-specific and do not establish production accuracy, model impact, or causation. No
benchmark runtime or memory measurements were collected.

## Remaining limitations and next steps

1. Evaluate detector behavior across correlated features, additional sample sizes, and multiple
   simultaneous shifts; current published scenarios mostly use independent features.
2. Measure runtime and memory against larger bounded input sizes before production deployment.
3. Validate alert usefulness and investigator behavior on appropriately governed,
   representative data; caller-supplied evidence is not authenticated.

The API has bounded body and dataset dimensions but does not impose an application-level request
deadline or concurrency limit. Dependency auditing skipped the editable project. Nothing was
pushed; publication remains for explicit approval.
