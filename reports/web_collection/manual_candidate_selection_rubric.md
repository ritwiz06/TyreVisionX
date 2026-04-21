# Manual Candidate Selection Rubric

Updated: 2026-04-20

## Purpose

This rubric helps choose likely-normal tyre web candidates during manual browsing. It does not create labels. Every selected image remains a candidate until it passes filtering and human review.

## Strong Candidate Signals

Approve a candidate row for the pilot CSV when most of these are true:

- The tyre appears undamaged and likely normal.
- The tyre is the main subject of the image.
- The image has enough resolution and sharpness to inspect tread or sidewall detail.
- The view matches one of the pilot needs: tread, sidewall, mounted tyre, inspection-like, or industrial/off-highway.
- Lighting is clear enough to avoid confusing shadows with cracks.
- The source page URL can be recorded.
- License, attribution, or source ownership notes can be recorded or marked as unknown.

## Avoid These Images

Reject the result while browsing if any of these are visible:

- cracks, punctures, burst sidewalls, cuts, exposed cords, deformation, or severe wear
- scrap-yard, repair-shop, accident, failure, or defect examples
- tiny tyres in the background
- diagrams, icons, renderings, illustrations, memes, or ad collages
- heavy watermark across the tyre surface
- low-resolution, blurry, overexposed, or heavily occluded images
- source pages that cannot be traced or look unreliable

## How To Record Source and License Notes

In `approved_tyres_pilot_urls_001_prefilled.csv`, fill:

- `source_url`: direct image URL, if available
- `page_url`: webpage where the image was found
- `license_name`: license shown by the page, or `unknown`
- `license_url`: license link, if shown
- `attribution_text`: photographer/source text, if shown
- `notes`: short reason for approval or uncertainty

If a license is unclear, write `unknown` rather than guessing.

## Why Candidates Are Not Labels

A search result may look normal but still be used, edited, mislabeled, defective, or legally unsuitable. TyreVisionX treats manual web rows as research candidates only. A row can enter a curated likely-normal manifest only after filtering and human review marks it `approved_likely_normal`.

## Recommended Browsing Order

1. Tread priority-1 queries.
2. Sidewall priority-1 queries.
3. Inspection-like priority-1 queries.
4. Mounted tyre queries.
5. Industrial/off-highway queries.

This order prioritizes images most useful for learning normal tyre texture before broader context images.
