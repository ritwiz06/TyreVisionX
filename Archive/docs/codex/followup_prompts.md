# TyreVisionX Codex Follow-up Prompts

## Prompt 1 — Audit first, do not refactor yet
Audit this repository and produce a written repo audit before changing code.

Tasks:
1. Identify the canonical training/evaluation path.
2. List duplicated modules and conflicting config paths.
3. Find broken imports, dead references, and mismatched docs.
4. Compare README claims against actual files present.
5. Propose a low-risk refactor plan.
6. Create `reports/project_status/repo_audit.md` and `docs/PROJECT_STATUS.md`.

Rules:
- Do not invent results.
- Do not move files yet unless necessary for the audit output.
- Be explicit about what is implemented, partial, missing, and future work.

## Prompt 2 — Refactor structure safely
Use the repo audit to refactor the repository safely.

Tasks:
1. Standardize source code under `src/tyrevisionx/`.
2. Move older or parallel code into `src/legacy/` if needed.
3. Unify dataset loading and manifest conventions.
4. Fix imports and config paths.
5. Add compatibility wrappers where useful.
6. Update README structure and commands.

Rules:
- Preserve working behavior.
- Prefer low-risk moves.
- Keep documentation honest.

## Prompt 3 — Add reports, logs, and concept docs
Create professor-ready documentation and tracking files.

Generate:
- `docs/ARCHITECTURE.md`
- `docs/RESEARCH_ROADMAP.md`
- `docs/ML_CONCEPTS_USED.md`
- `logs/PROJECT_LOG.md`
- `logs/EXPERIMENT_LOG.csv`
- `logs/DECISIONS.md`
- `reports/project_status/implemented_vs_planned.md`

Rules:
- Use simple explanations first.
- Separate current implementation from future plans.
- Do not fabricate experiment values.

## Prompt 4 — Generate notebooks and dashboard
Create clean notebooks and one interactive dashboard for the current repo.

Generate:
- `notebooks/00_repo_overview.ipynb`
- `notebooks/01_dataset_audit.ipynb`
- `notebooks/02_baseline_results.ipynb`
- `notebooks/03_error_analysis.ipynb`
- `notebooks/04_model_comparison.ipynb`
- `notebooks/05_research_roadmap.ipynb`
- `src/tyrevisionx/reporting/dashboard.py`

Rules:
- Notebooks must run without hidden state.
- If artifacts are missing, show placeholders and explanations instead of invented plots.
- The dashboard must degrade gracefully when files are missing.

## Prompt 5 — Verify and review
Before finishing, run the appropriate checks and review the refactor.

Tasks:
1. Run lint, tests, and import checks.
2. Verify README commands.
3. Verify notebooks reference correct paths.
4. Review for stale references.
5. Summarize remaining known issues.
