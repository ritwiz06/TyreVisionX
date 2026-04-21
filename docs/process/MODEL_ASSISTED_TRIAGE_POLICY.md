# Model-Assisted Triage Policy

Updated: 2026-04-19

Model-assisted triage means using model outputs to prioritize human review. It does not mean automatic labeling.

## Allowed Uses

Allowed:
- rank candidates for review
- flag likely anomalous candidates for closer inspection
- identify likely normal candidates for efficient review
- combine quality status, duplicate status, query priority, and anomaly buckets into a review-priority bucket

Not allowed:
- auto-promote candidates into training
- auto-reject candidates as defective ground truth
- retrain supervised classifiers with model-generated web labels

## Review-Priority Buckets

- `highest_priority_review`: likely important or suspicious candidates that need human attention.
- `normal_candidate_review`: candidates that may be likely-normal but still require review.
- `uncertain_review`: ambiguous candidates.
- `reject_before_review`: failed quality/provenance checks, exact duplicates, or unusable files.

## TyreVisionX Rule

Human review remains the decision gate. Model scores are advisory signals only.
