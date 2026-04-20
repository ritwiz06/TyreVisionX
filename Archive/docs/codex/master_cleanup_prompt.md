# TyreVisionX Codex Master Prompt

You are working inside my TyreVisionX repository. This project is at an **early research stage**. Do **not** pretend that results are production-grade or that future modules are already complete. Your goal is to clean, standardize, document, and stabilize the repo so it becomes a strong foundation for research with my professor.

## Mission
Refactor the repository into a clean, reproducible, research-friendly structure. Fix inconsistencies across code, folders, configs, docs, reports, and notebooks. Preserve useful existing work, but reduce confusion between legacy and current pipelines. Treat this as a **repo cleanup + research infrastructure setup** task, not a model-improvement task.

## High-level goals
1. Audit the repository structure and code paths.
2. Fix inconsistencies and obvious code irregularities.
3. Standardize folder structure and update README accordingly.
4. Separate **current implemented work** from **future planned work**.
5. Generate clear notebooks, reports, and logs that explain the project state.
6. Add durable guidance files so future Codex sessions stay consistent.
7. Do not invent results. Only summarize results already present in logs/reports/artifacts.

## Current repo irregularities to fix
Please explicitly inspect and fix these:

1. **Dual pipeline confusion**
   - There is an older baseline path and a newer config-driven path.
   - Keep both only if clearly labeled.
   - If the older path is retained, move it under a `legacy/` or clearly documented baseline section.
   - Make it obvious which training/evaluation path is the canonical one for current work.

2. **Manifest/config inconsistency**
   - `configs/train_resnet18.yaml` currently points to `data/processed/D1_manifest.csv`.
   - `configs/data.yaml` points to `data/manifests/D1_tyrenet_manifest.csv` and similar manifest files.
   - Unify this. Pick one consistent manifest convention and use it everywhere.

3. **Missing or inconsistent registry usage**
   - In `src/train.py`, `register_model(...)` is called but appears not to be imported.
   - Fix the import or the registry flow.
   - If registry is optional, handle it cleanly and document the fallback behavior.

4. **Dataset class duplication / naming confusion**
   - There is `src/dataset.py` and also `src/data/datasets.py` with overlapping responsibilities.
   - Consolidate or clearly separate them.
   - Prefer one canonical dataset-loading module and update imports everywhere.

5. **README vs actual repo state mismatch**
   - The README describes a polished structure and outputs, but the repo still feels partially evolving.
   - Rewrite README to accurately reflect the current status:
     - what is already implemented
     - what is experimental
     - what is planned

6. **Notebook/report mismatch**
   - `PROGRESS_LOG.md` references notebooks/reports that are not all clearly present.
   - Make notebook/report structure consistent.
   - If a referenced notebook is missing, either generate it or update documentation honestly.

7. **Logs and status tracking are not centralized enough**
   - Create a clean project-level research log system.
   - Preserve historical observations from existing logs/reports.

8. **Future work is mixed into implementation docs**
   - Separate future plans from current validated code.
   - Create roadmap docs rather than leaving future work scattered.

## Required final folder structure
Refactor the repo to follow this structure as closely as possible without breaking imports:

TyreVisionX/
├── .codex/
│   └── config.toml
├── .github/
│   └── workflows/
├── configs/
│   ├── data/
│   │   └── datasets.yaml
│   ├── train/
│   │   ├── train_resnet18.yaml
│   │   ├── train_resnet34.yaml
│   │   └── train_simplecnn.yaml
│   ├── aug/
│   │   ├── light.yaml
│   │   └── strong.yaml
│   └── README.md
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── manifests/
├── docs/
│   ├── PROJECT_STATUS.md
│   ├── ARCHITECTURE.md
│   ├── RESEARCH_ROADMAP.md
│   ├── ML_CONCEPTS_USED.md
│   ├── PROFESSOR_DISCUSSION_NOTES.md
│   └── REPO_STRUCTURE.md
├── logs/
│   ├── PROJECT_LOG.md
│   ├── EXPERIMENT_LOG.csv
│   ├── CHANGELOG_RESEARCH.md
│   └── DECISIONS.md
├── notebooks/
│   ├── 00_repo_overview.ipynb
│   ├── 01_dataset_audit.ipynb
│   ├── 02_baseline_results.ipynb
│   ├── 03_error_analysis.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_research_roadmap.ipynb
├── reports/
│   ├── project_status/
│   │   ├── current_status.md
│   │   ├── implemented_vs_planned.md
│   │   └── repo_audit.md
│   ├── experiments/
│   ├── figures/
│   └── dashboard/
├── scripts/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── reporting/
│   └── utilities/
├── src/
│   ├── tyrevisionx/
│   │   ├── data/
│   │   ├── models/
│   │   ├── training/
│   │   ├── evaluation/
│   │   ├── serving/
│   │   ├── reporting/
│   │   └── utils/
│   └── legacy/
├── tests/
├── artifacts/
│   ├── experiments/
│   ├── reports/
│   ├── registry/
│   └── exports/
├── AGENTS.md
├── README.md
├── PROGRESS_LOG.md
├── pyproject.toml
└── requirements.txt

## Refactor strategy
Use an incremental, low-risk approach:

### Phase 1 — Repo audit and safe cleanup
- Inventory current files and modules.
- Map legacy vs current pipeline.
- Identify broken imports, duplicated logic, dead references, and stale docs.
- Rename/move files only when imports and scripts are updated accordingly.
- Add compatibility shims if needed to avoid breaking everything at once.

### Phase 2 — Standardize code layout
- Move source code under `src/tyrevisionx/`.
- Keep temporary compatibility wrappers if old import paths exist.
- Group by responsibility: data, models, training, evaluation, serving, reporting, utils.
- Move older experimental scripts into `src/legacy/` or `scripts/legacy/` and label them clearly.

### Phase 3 — Config cleanup
- Split config files into `configs/data/`, `configs/train/`, and `configs/aug/`.
- Use one naming convention and one manifest convention.
- Update all paths in code, README, and scripts.

### Phase 4 — Documentation cleanup
Update `README.md` so it contains:
1. Project vision
2. Current stage
3. What is implemented now
4. What is experimental
5. What is planned for future phases
6. Clean repository structure tree
7. How to run data prep, training, evaluation, serving, notebooks, and reports
8. Known limitations
9. Research roadmap

### Phase 5 — Reporting and logs
Create or update the following:

1. `docs/PROJECT_STATUS.md`
   - current state of the repo
   - completed work
   - partially completed work
   - broken / inconsistent parts fixed in this refactor
   - next steps

2. `reports/project_status/repo_audit.md`
   - list of irregularities found
   - exact fixes applied

3. `reports/project_status/implemented_vs_planned.md`
   - two-column separation:
     - implemented now
     - future planned modules

4. `logs/PROJECT_LOG.md`
   - chronological log of what has been done so far
   - import content from `PROGRESS_LOG.md` if useful
   - make it readable and ongoing

5. `logs/EXPERIMENT_LOG.csv`
   - columns:
     `date, experiment_id, dataset, model, augmentation, loss, optimizer, epochs, split_strategy, primary_metric, result_summary, artifact_path, notes, status`
   - prefill from any committed reports or progress notes where possible

6. `logs/DECISIONS.md`
   - major architecture and research decisions
   - why they were made
   - what remains undecided

7. `docs/ML_CONCEPTS_USED.md`
   - plain-English explanations first
   - then project-specific usage
   - cover topics already used and topics planned for future use

## ML concepts file requirements
`docs/ML_CONCEPTS_USED.md` must explain, in easy language first, then technical detail:
- CNN
- convolution, filters, pooling
- transfer learning
- ResNet
- overfitting
- generalization
- augmentation
- class imbalance
- BCE / CrossEntropy / BCEWithLogitsLoss
- confusion matrix
- precision / recall / F1 / AUROC / AUPRC
- threshold tuning
- misclassification analysis
- anomaly detection
- defect localization
- segmentation
- multi-view learning
- 3D reconstruction
- GNN / CNN→GNN idea
- knowledge graph / root-cause reasoning
For each topic, include:
1. simple explanation
2. why it matters in TyreVisionX
3. whether it is already used now or planned for future work

## Notebook requirements
Generate notebooks that are readable, research-friendly, and presentation-friendly.
They must run without hidden state and should clearly annotate what they are showing.

### 1. `00_repo_overview.ipynb`
- explain repository layout
- explain pipeline stages
- explain which code is current vs legacy

### 2. `01_dataset_audit.ipynb`
- load manifests
- show class counts
- split counts
- sample images if available
- basic imbalance analysis
- missing file checks if possible

### 3. `02_baseline_results.ipynb`
- summarize current available baseline results from committed artifacts/logs/reports
- clearly state if data is loaded from saved JSON/CSV or manually summarized from existing files
- no invented numbers

### 4. `03_error_analysis.ipynb`
- show confusion matrix
- false negatives / false positives if artifacts exist
- explain why this matters for industrial inspection

### 5. `04_model_comparison.ipynb`
- compare SimpleCNN vs ResNet variants using whatever committed results exist
- if exact result files are missing, clearly mark empty sections as placeholders
- no fabricated metrics

### 6. `05_research_roadmap.ipynb`
- current stage
- next experiments
- future modules
- open questions for professor discussion

## Interactive reporting
Create one small interactive reporting tool, preferably lightweight and maintainable.
Use **Streamlit + Plotly** if dependencies are already reasonable.

Suggested file:
- `src/tyrevisionx/reporting/dashboard.py`

It should:
- read experiment logs / report JSONs if present
- show project status cards
- show implemented vs planned modules
- show available experiment results
- show links/paths to artifacts
- still run gracefully if some artifacts are missing

Also add a script entry or command in README for launching it.

## README structure to produce
Rewrite README with this structure:
1. Project title and one-paragraph overview
2. Current maturity level (early-stage academic research prototype)
3. Research motivation
4. Current implemented components
5. Current limitations
6. Repository structure tree
7. Quickstart
8. Data preparation workflow
9. Training workflow
10. Evaluation workflow
11. Reports and notebooks
12. Interactive dashboard
13. Current results summary (only committed or reproducible results)
14. Planned future work
15. Research discussion topics for professor meetings
16. Contribution / maintenance notes

## Code quality requirements
- Keep code readable and modular.
- Use consistent naming.
- Update imports after refactors.
- Add docstrings where useful.
- Add or update tests for critical paths.
- Do not remove working code unless it is clearly obsolete and replaced.
- Prefer backward-compatible wrappers when renaming modules.

## Validation requirements
Before finishing:
1. Run formatting/lint/test commands if configured.
2. Verify imports for main entry points.
3. Verify README commands match actual file paths.
4. Verify notebook paths are correct.
5. Verify docs do not claim nonexistent files.
6. Print a concise summary of changed files and why.

## Output expectations
At the end, provide:
1. A repo audit summary
2. A list of key fixes made
3. The final directory structure
4. Any remaining known issues
5. Suggested next technical steps
6. Suggested next research steps

Important:
- Be honest.
- Do not fabricate experimental results.
- Do not overstate maturity.
- Optimize for clarity, reproducibility, and professor-readiness.
