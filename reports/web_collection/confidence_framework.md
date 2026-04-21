# Confidence Framework

Updated: 2026-04-19

## Purpose

The confidence framework assigns review-priority buckets for web candidates. It does not assign labels.

## Inputs

- `quality_status`
- duplicate/review notes
- `anomaly_triage_bucket`
- query priority

## Buckets

| Bucket | Meaning |
|---|---|
| `highest_priority_review` | Suspicious or likely anomalous candidate; inspect carefully. |
| `normal_candidate_review` | Candidate may be likely-normal, but still needs human review. |
| `uncertain_review` | Not enough evidence for efficient prioritization. |
| `reject_before_review` | Failed hard checks or exact duplicate; do not promote. |

## Safety Rule

These buckets are for queue ordering and review planning only. They must not be converted into ground-truth labels.
