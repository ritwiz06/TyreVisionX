# Web Collection Policy

Updated: 2026-04-19

This policy defines how TyreVisionX may collect and use web-sourced image candidates for research.

## Approved Collection Modes

Approved now:
- manual CSV/JSON import of URLs selected by the researcher
- local-file import for testing the metadata, filtering, and review workflow
- future approved provider APIs that return structured metadata

Not approved:
- scraping Google HTML
- bypassing provider terms
- collecting images without source URL metadata
- treating web images as automatically labeled training data

## Research-Candidate Status

Every web image is a research candidate until it passes review. A candidate can be useful for exploration, deduplication tests, anomaly triage, or review queues, but it is not a confirmed good tyre example by default.

## No Automatic Labeling

The pipeline must not auto-label images as `good` or `defect`. It may assign:
- quality status
- duplicate status
- anomaly triage bucket
- human review status

Only human review can move a candidate toward `approved_likely_normal`.

## Human Review Requirement

Human review is required before any candidate is used for anomaly training. Reviewers should reject images that are irrelevant, visibly defective, too low quality, heavily edited, watermarked in a way that hides the tyre surface, or ambiguous.

## Metadata Retention

The pipeline must retain:
- original source URL
- page URL when available
- provider name
- query ID and query text
- retrieval timestamp
- local path
- hash/dedupe fields
- quality and review fields

This supports traceability, reproducibility, and later rights review.

## Rights Caution

Public web images may have copyright, license, or usage restrictions. For now, web candidates are for research workflow development only. Future productization must use licensed, owned, or otherwise approved data.

## TyreVisionX Position

This policy supports a disciplined data-curation engine for tyre anomaly research. It does not claim the project has a production web-scraping or data-rights solution.
