# Web Candidate Review Guidelines

Updated: 2026-04-19

## Goal

Review web-image candidates for possible inclusion in future good-only anomaly training data. Do not treat candidates as automatically good.

## Review Statuses

| Status | Use when |
|---|---|
| `pending_review` | Not reviewed yet. |
| `approved_likely_normal` | Tyre is relevant, visible, high enough quality, and no obvious defect is visible. |
| `rejected_irrelevant` | Image is not a tyre or not useful for tyre inspection. |
| `rejected_low_quality` | Too blurry, too small, obstructed, watermarked over the tyre surface, or otherwise unusable. |
| `rejected_likely_defect` | Visible crack, puncture, burst, severe wear, damage, or deformation. |
| `uncertain_holdout` | Ambiguous case that should not enter training yet. |

## Tyre-Specific Guidance

Prefer:
- clean sidewall close-ups
- clean tread close-ups
- product photos with visible tyre surface
- industrial tyres with undamaged visible surface
- inspection-like views with stable lighting

Reject or hold:
- cracked or punctured tyres
- heavily worn or dirty tyres if normality is unclear
- images where the tyre is tiny in the frame
- diagrams, drawings, logos, or synthetic images unless explicitly needed
- images with uncertain rights for any future non-research use

## Generalization To Other Products

For another product, replace the tyre-specific visual rules with product-specific normality and defect criteria while keeping the same review statuses and metadata workflow.
