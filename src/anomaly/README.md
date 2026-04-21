# Anomaly Module

This package contains the first TyreVisionX good-only anomaly baseline.

Current status:
- manifest-driven anomaly datasets are implemented
- frozen ResNet embedding extraction is implemented
- Mahalanobis distance scoring is implemented
- optional kNN distance scoring is implemented
- validation-only threshold selection is implemented
- test evaluation/export is implemented

The default baseline is intentionally simple:

1. Train only on normal/good tyre images.
2. Extract frozen CNN embeddings.
3. Fit the normal embedding distribution.
4. Score validation/test images by distance from normal.
5. Select a threshold on validation only.
6. Evaluate once on test.

The implementation is meant to be reusable for other manufacturing products through manifests. Tyre-specific assumptions should stay in manifest generation and reports, not in the core scorer.
