# Roboflow Universe Landscape

## Scope
This is a limited registry/audit layer for external tyre datasets. It does not download data and does not merge any external dataset into D1.

## Shortlisted Sources
| Dataset | Task | Images | Visible License | Current Role |
|---|---|---:|---|---|
| Good Tire Bad Tire | Classification | ~1.8k | CC BY 4.0 | Near-term anomaly support candidate |
| Tires Defects | Classification | ~3.3k | CC BY 4.0 | Near-term supervised benchmark candidate |
| Tire Tread | Object detection | ~6.3k | Public Domain | Later localization/tread benchmark |
| defect (Hemant) | Object detection | 880 | CC BY 4.0 | Later tyre damage localization |
| tire (College segmentation) | Segmentation | Unknown | Pending | Hold |
| Tire Quality | Classification | 1,839 | Pending explicit verification | Hold |

## Policy
- External data remains separate from D1 until license, duplicate, label, and split leakage audits are complete.
- Classification datasets are the only near-term candidates for anomaly/support work.
- Detection and segmentation datasets are later-phase resources unless a specific localization experiment is being run.
- Roboflow exports must be imported through a review manifest, not directly merged into canonical manifests.

## Sources Consulted
- Roboflow Universe pages and search results for the shortlisted datasets.
- Repository-local TyreVisionX anomaly status and manifest conventions.
