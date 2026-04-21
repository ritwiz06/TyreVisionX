# AGENTS.md

## Project
TyreVisionX is an early-stage academic research prototype for tyre defect inspection.
It is not yet a production system. Always distinguish:
- implemented now
- experimental / partial
- future planned work

## Main goals in this repo
1. Keep the repository clean and reproducible.
2. Support research discussions with clear docs, reports, and notebooks.
3. Avoid overclaiming results.
4. Prefer maintainable structure over rapid but messy expansion.

## Coding rules
- Keep imports consistent after refactors.
- Do not fabricate metrics, artifacts, or reports.
- If a result file is missing, leave a clear placeholder instead of inventing data.
- If moving files, update references in README, scripts, configs, and tests.
- Preserve useful legacy code, but mark it clearly as legacy.
- Add docstrings for nontrivial modules and functions.

## Repo priorities
1. Reproducibility
2. Clear structure
3. Honest status reporting
4. Professor-readiness
5. Research extensibility

## What to verify after changes
- Main training entry point imports correctly.
- Main evaluation entry point imports correctly.
- README commands are valid.
- Config paths are valid.
- Notebooks reference real files.
- Tests, lint, and formatting are run when possible.

## Current architectural direction
Current validated focus:
- dataset ingestion and manifests
- supervised binary classification
- evaluation and error analysis
- research reporting

Near-term research direction:
- supervised baseline stabilization
- anomaly detection trained on good-only tyre images
- web-data curation with human review
- cross-dataset robustness
- localization / segmentation

Long-term roadmap:
- multi-view 3D reconstruction
- defect projection to mesh
- knowledge graph reasoning

## Documentation expectations
Any major refactor should update:
- README.md
- docs/project/PROJECT_STATUS.md
- docs/architecture/REPO_ARCHITECTURE.md
- docs/project/ROADMAP.md
- reports/current_status/repo_audit.md
- reports/historical/REPORTED_RESULTS.md

## Prompt logging expectations
For substantial Codex work:
- read docs/codex/BASE_CONTEXT.md first
- read logs/work_logs/LATEST.md and logs/process_logs/LATEST.md when present
- create a timestamped work log and process log
- update latest log pointers
- never fabricate experiment metrics

## Review expectations
Before considering work done:
- summarize changes
- list remaining risks
- identify any assumptions
- suggest next steps
