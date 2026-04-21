# TyreVisionX Roadmap

Updated: 2026-04-19

This roadmap is intentionally conservative. It separates what should be stabilized now from research directions that should be explored later.

## Phase 1: Supervised Baseline Stabilization

Goals:
- Confirm one canonical supervised pipeline.
- Standardize manifest paths and dataset-root behavior.
- Recompute supervised baseline results from the cleaned repo.
- Keep `good = 0` and `defect = 1` consistent everywhere.
- Report recall, precision, F1, AUROC, AUPRC, and false negatives.

Deliverables:
- Recomputed ResNet-18 or ResNet-34 baseline.
- Clean evaluation report under `artifacts/reports/`.
- Updated `reports/historical/REPORTED_RESULTS.md` with recomputed status.

## Phase 2: Good-Only Anomaly Baseline

Goals:
- Train an anomaly baseline using only good tyre images.
- Validate on good and defect images.
- Compare anomaly scores against supervised classifier predictions.
- Decide whether anomaly detection helps identify subtle or unexpected defects.

Candidate methods:
- frozen feature extractor plus nearest-neighbor distance
- embedding centroid distance
- patch-level embeddings
- lightweight autoencoder only after feature-baseline review

Deliverables:
- `configs/anomaly/anomaly_baseline.yaml` promoted from placeholder to runnable config. Done.
- anomaly train/eval script. Done for the first baseline.
- anomaly report with threshold policy. Done.
- first D1 pretrained ResNet18 Mahalanobis run. Done.
- false-negative review. Pending.
- false-negative analysis script/report/contact sheet. Done.
- controlled benchmark across ResNet18/ResNet50, Mahalanobis/kNN, threshold sweep, and a first patch-grid variant. Done.
- current best executed variant: ResNet50 + kNN. Done.
- corruption robustness benchmark and mild noise-robust ResNet50+kNN variant. Done.
- local-feature ResNet50 benchmark with threshold sweep, multicrop, and fine patch-grid variants. Done.
- patch-aware ResNet50 feature-map memory benchmark. Done; first layer4 patch-memory variants did not beat the high-recall reference.
- lower/mid-level patch-aware follow-up using layer3 and layer2+layer3 descriptors with robust score normalization. Done; these variants also did not beat the high-recall reference.

## Phase 3: Web-Data Collection and Curation

Goals:
- Define approved image providers or data sources.
- Collect metadata and raw image files without hard-coded credentials.
- Deduplicate images.
- Filter low-quality images.
- Use anomaly scoring to flag likely-good candidates for human review.

Important constraint:
- Anomaly scores must not be treated as ground-truth labels. They are triage signals for curation.

Current implementation status:
- web collection policy: done
- query catalog generator: done
- manual CSV/JSON provider: done
- provider API stubs: done
- source acquisition guide and provider source config: done
- manual Google discovery import: done
- metadata schema and filters: done
- review queue scaffolding: done
- manual pilot orchestrator and promotion path: done
- actual approved candidate ingestion: pending user-provided CSV

## Phase 4: Localization / Segmentation

Goals:
- Move beyond image-level labels when bounding boxes or masks are available.
- Add defect localization once classification and anomaly baselines are understood.

Candidate methods:
- YOLO-style detector
- U-Net
- Mask R-CNN
- Grad-CAM as a weak localization analysis tool

## Phase 5: Multi-View, 3D, and Knowledge Reasoning

Goals:
- Explore multi-view tyre inspection after single-image baselines are stable.
- Map defect evidence across views or onto a simple tyre geometry representation.
- Explore knowledge reasoning for defect type, risk, and likely inspection causes.

This is future research, not current implementation.

## Immediate Next Steps

1. Run import and test checks after cleanup.
2. Recompute the supervised baseline from the canonical config path.
3. Review anomaly false negatives from `artifacts/anomaly/local_features/resnet50_knn_threshold_sweep/false_negatives_test.csv`.
4. Inspect higher false-positive load from threshold refinement.
5. If continuing patch-aware work, diagnose patch-memory scoring before adding more variants; layer4, layer3, and layer2+layer3 attempts have not reduced false negatives.
6. Audit the Roboflow shortlist before any external import. Do not merge external data into D1.
7. Create `data/external/manual_candidate_urls/approved_tyres_pilot_urls_001.csv`.
8. Run `scripts/web_collection/run_manual_pilot.py`.
9. Export and review the pilot review pack.
10. Promote only human-approved likely-normal candidates.
