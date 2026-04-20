# TyreVisionX Cleanup Blueprint

## Main cleanup targets
- unify canonical training/evaluation path
- resolve manifest path inconsistencies
- clean package structure
- separate legacy from active code
- add professor-ready docs
- add reproducible notebooks
- add centralized logs and project status reports

## Recommended canonical structure
- `src/tyrevisionx/` for active code
- `src/legacy/` for older baseline paths if retained
- `configs/data/`, `configs/train/`, `configs/aug/`
- `docs/` for narrative and conceptual docs
- `logs/` for chronological and tabular tracking
- `reports/` for status and experiment summaries
- `notebooks/` for clear presentation and analysis
- `artifacts/` for generated outputs only

## Recommended docs to create
- `docs/PROJECT_STATUS.md`
- `docs/ARCHITECTURE.md`
- `docs/RESEARCH_ROADMAP.md`
- `docs/ML_CONCEPTS_USED.md`
- `docs/PROFESSOR_DISCUSSION_NOTES.md`
- `docs/REPO_STRUCTURE.md`

## Recommended logs to create
- `logs/PROJECT_LOG.md`
- `logs/EXPERIMENT_LOG.csv`
- `logs/CHANGELOG_RESEARCH.md`
- `logs/DECISIONS.md`

## Recommended notebooks to create
- `00_repo_overview.ipynb`
- `01_dataset_audit.ipynb`
- `02_baseline_results.ipynb`
- `03_error_analysis.ipynb`
- `04_model_comparison.ipynb`
- `05_research_roadmap.ipynb`

## Recommended interactive report
Use Streamlit + Plotly for a lightweight dashboard that reads:
- `artifacts/reports/**/*.json`
- `logs/EXPERIMENT_LOG.csv`
- `reports/project_status/*.md`

## Acceptance criteria
- README matches actual repo structure
- config paths are consistent
- one canonical pipeline is obvious
- legacy code is clearly labeled
- notebooks exist and are honest about missing artifacts
- logs and status reports are centralized
- no documented files are missing
