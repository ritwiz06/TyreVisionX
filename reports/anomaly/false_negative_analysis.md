# False Negative Analysis

Status: completed from the first D1 anomaly run.

## Summary

- false negatives: `93`
- threshold: `12.147390651173238`
- score range: `9.01895191441506` to `12.137679785506592`
- mean score: `10.885576388758198`
- near threshold count (<= 1.0 below threshold): `43`
- far below threshold count (> 3.0 below threshold): `2`

## Outputs

- review table: `reports/anomaly/false_negative_review_table.csv`
- contact sheet: `reports/anomaly/false_negative_contact_sheet.png`

## Interpretation

The CSV metadata confirms these are defect-labeled tyres that scored below the anomaly threshold. It does not reveal the visual cause by itself. Visual review is required to determine whether missed defects are small cracks, low-contrast defects, edge-of-image defects, or possible label noise.

Global pooled ResNet features can miss local tyre defects because a small crack may occupy only a small part of the image. Pooling compresses the whole image into one vector, so local evidence can be diluted by normal tread, sidewall, and lighting patterns.
