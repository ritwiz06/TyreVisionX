# Current Status

Date: 2026-04-06

## Summary
TyreVisionX now has one clear canonical current pipeline for supervised binary classification and one clearly archived legacy baseline path. The repository is cleaner and more honest than before this cleanup, but it is still an early-stage research prototype.

## Implemented Now
- canonical config-driven training and evaluation pipeline
- D1 manifest-driven binary classification workflow
- evaluation reports with confusion matrix and ROC/PR plots
- FastAPI and Streamlit utilities
- historical baseline reports and experiment notes
- centralized project status, changelog, decisions, and experiment log files

## Partial / Experimental
- D2 and D3 support in config but not populated in tracked manifests
- CNN→GNN support behind an optional dependency
- registry-backed inference only when local experiment outputs exist
- compatibility copies for old config paths and root-level baseline imports

## Future Planned Work
- anomaly detection
- defect localization
- segmentation
- cross-dataset robustness studies
- multi-view 3D reconstruction
- knowledge-graph-style reasoning

## What Was Fixed In This Cleanup
- active and legacy pipelines are now documented separately
- train and eval now share the same dataset-loading contract
- default config paths were normalized under `configs/data/`, `configs/train/`, and `configs/aug/`
- baseline code now has a clear `src/legacy/` namespace
- root-level compatibility paths remain, but are explicitly marked as compatibility

## Current Risks
- the repository still mixes active paths and compatibility paths
- the strongest documented results are historical baseline results, not fresh canonical-pipeline outputs
- missing D2/D3 manifests limit any claim of current cross-dataset validation
- notebooks still reflect the older research workflow more than the cleaned status-reporting workflow
