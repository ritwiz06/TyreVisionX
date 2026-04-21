# Platform Strategy

Updated: 2026-04-19

TyreVisionX is a tyre anomaly prototype first. It is not yet a universal manufacturing inspection product.

## Near-Term Strategy: Tyres First

The immediate research goal is to build a credible tyre anomaly baseline:
- good-only training images
- mixed validation/test evaluation
- calibrated thresholds
- false-negative and false-positive review
- disciplined expansion of likely-normal candidates

## Reusable Core

The reusable part should be the architecture, not a claim that one model solves every product:
- manifest-based data contracts
- frozen feature extraction and scoring modules
- threshold calibration policies
- metadata and review queues
- report outputs that separate validation from test behavior

## Future Product Adapters

A future product adapter should define:
- product-specific query catalog
- product-specific review guidelines
- product-specific normal/anomaly label mapping
- product-specific minimum resolution and view assumptions
- product-specific thresholds after validation

## Technical Moat

The defensible part of the project is not one backbone. Pretrained CNNs can be replaced.

The stronger moat is:
- a disciplined normal-data curation workflow
- retained metadata and provenance
- deduplication and quality control
- human-in-the-loop review
- calibrated anomaly thresholds
- consistent evaluation and deployment discipline

## Honest Current Maturity

Current status:
- supervised tyre baselines exist historically
- a first anomaly baseline code path exists
- web curation now has manual/provider-stub infrastructure
- actual large-scale candidate ingestion and review are pending

The project is research-ready for the next data-curation experiment, not product-ready.
