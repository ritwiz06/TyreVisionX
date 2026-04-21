# Patch-Aware Best Variant Recommendation

## Recommendation
Keep `resnet50_knn_threshold_sweep` as the current best recall-oriented TyreVisionX anomaly candidate.

## Evidence
The patch-aware variants were executed, but none beat the high-recall reference:

- Reference: recall `0.9462`, FN `7`, FP `36`
- Feature-map patch kNN: recall `0.1000`, FN `117`, FP `6`
- Feature-map patch kNN threshold sweep: recall `0.4615`, FN `70`, FP `30`
- PatchCore-lite: recall `0.2385`, FN `99`, FP `11`

## What This Means
The current PatchCore-style local memory implementation is not yet useful for TyreVisionX. It reduced false positives in some settings but missed far too many defects. In a recall-critical tyre-inspection workflow, that tradeoff is not acceptable.

## Is PatchCore-Style Local Memory Still Promising?
Possibly, but not in the current form. The concept is still relevant because tyre defects can be local. The implementation likely needs better feature layers and scoring:

- layer3 or layer2+layer3 descriptors instead of only layer4
- patch score normalization
- memory coreset selection based on feature diversity rather than random sampling
- spatial smoothing or top-k scoring tuned on validation

## Next Experiment
Before more benchmark runs, visually inspect the 7 remaining high-recall false negatives and the 36 false positives. Then run a smaller layer3 patch-memory experiment if the missed defects are visibly local.
