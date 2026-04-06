# Changelog

## 2026-04-06
### Repository cleanup
- added canonical config layout under `configs/data/`, `configs/train/`, and `configs/aug/`
- kept root-level config files as compatibility copies and marked them as deprecated paths
- unified train and eval around the same runtime dataset-loading contract
- fixed missing `register_model` import in `src/train.py`
- normalized manifest preparation and fold-preparation defaults toward the canonical config path
- added `src/legacy/` namespace wrappers for archived baseline components
- updated tests to cover runtime dataset config loading
- added local pytest path bootstrapping and multipart-aware service-test skipping

### Documentation and reporting
- rewrote `README.md` to match actual repository maturity
- updated `docs/PROJECT_STATUS.md`
- added `docs/ARCHITECTURE.md`
- added `reports/project_status/current_status.md`
- updated `reports/project_status/repo_audit.md` with cleanup results
- added `reports/research_notes/concepts_used.md`
- added project logs for changes, decisions, experiments, and next steps
