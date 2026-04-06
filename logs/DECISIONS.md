# Decisions Log

## 2026-04-06: Establish honest repo status documents before refactoring
Decision:
- Create status and audit documents before changing code.

Reason:
- The repository has multiple parallel paths and documentation drift.
- A written baseline is needed before structural refactor work.

Implication:
- Refactor work should use these documents as the source of truth for what is active, legacy, partial, and planned.

## 2026-04-06: Treat the config-driven ResNet path as the active intended pipeline
Decision:
- Treat `src/train.py` and `src/evaluate.py` as the active intended pipeline.

Reason:
- This path is the closest match to the project’s stated architectural direction.
- It supports YAML configuration, combined dataset loading, reporting, and future model extensibility.

Implication:
- Future cleanup should converge other active docs and commands onto this path.

## 2026-04-06: Treat the baseline CLI path as legacy / parallel research code
Decision:
- Treat `src/train_baseline.py`, `src/eval_baseline.py`, `src/dataset.py`, `src/transforms.py`, and `scripts/day5_seed_sweep.py` as legacy or parallel baseline code.

Reason:
- These files are still useful for historical results and baseline comparisons.
- They do not match the desired consolidated architecture.

Implication:
- Preserve them for reproducibility, but label them clearly and keep them separate from the canonical path.

## 2026-04-06: Standardize on one manifest convention in the next refactor
Decision:
- The next safe refactor should choose a single manifest convention and make train/eval share it.

Observed conflict:
- `data/processed/D1_manifest.csv` uses repo-relative image paths under `data/raw/...`
- `data/manifests/D1_tyrenet_manifest.csv` uses dataset-root-relative image paths such as `defect/...`

Implication:
- Dataset loaders and docs should converge on one convention before larger package moves.

## 2026-04-06: Do not describe D2 and D3 support as currently validated
Decision:
- Describe D2 and D3 support as partial or scaffolded, not currently validated.

Reason:
- Their manifest files are present but empty except for headers in this checkout.
- No current experiment outputs were identified demonstrating cross-dataset reporting from the active path.

Implication:
- README and status docs should avoid claiming completed cross-dataset robustness results.

## 2026-04-06: Keep future model families labeled as planned work
Decision:
- Keep CNN→GNN, detection, and segmentation work labeled as experimental or planned unless backed by verified artifacts and updated docs.

Reason:
- The repo contains code hooks and markdown stubs, but not a stable validated end-to-end pipeline for those tracks.

Implication:
- Research roadmap language is appropriate.
- Production-style maturity language is not appropriate.

## 2026-04-06: Refactor in low-risk order
Decision:
- Use the following refactor order when code changes begin:
1. unify manifest convention
2. unify train/eval data source selection
3. consolidate duplicated dataset and transform modules
4. fix runtime/documentation mismatches
5. then move or relabel legacy code

Reason:
- This order reduces the chance of breaking historical experiment reproducibility while making the current path coherent first.

Implication:
- Large package moves should wait until the active data contract is stable.

## 2026-04-06: Use config subdirectories as the canonical config layout
Decision:
- Canonical config paths now live under `configs/data/`, `configs/train/`, and `configs/aug/`.

Reason:
- This makes the active path easier to explain and reduces ambiguity between dataset, augmentation, and train configs.

Implication:
- Root-level config files remain only as compatibility copies.

## 2026-04-06: Keep compatibility paths instead of deleting them in this cleanup
Decision:
- Do not delete root-level configs or root-level baseline modules in this refactor.

Reason:
- The safest cleanup path is to preserve reproducibility while clearly labeling active versus compatibility paths.

Implication:
- Another follow-up pass can remove compatibility paths after verifying scripts, notebooks, and historical workflows.

## 2026-04-06: Treat `src/legacy/` as the archived baseline namespace
Decision:
- Expose archived baseline components through `src/legacy/`.

Reason:
- This makes the current pipeline easier to identify without breaking older imports immediately.

Implication:
- New legacy-facing references should prefer `src.legacy.*` over the old root-level baseline paths.
