# External Label Mapping Plan

## Canonical TyreVisionX Labels
For image-level anomaly work:

| Canonical Meaning | Target |
|---|---:|
| normal / good | 0 |
| anomaly / defect | 1 |

## Classification Mappings
| Dataset | Source Label | Canonical Label | Target |
|---|---|---|---:|
| Good Tire Bad Tire | `good` | normal | 0 |
| Good Tire Bad Tire | `defective` | anomaly | 1 |
| Tires Defects | `Good` | normal | 0 |
| Tires Defects | `Defected` | anomaly | 1 |
| Tire Quality | `good` | normal | 0 |
| Tire Quality | `defective` | anomaly | 1 |

## Detection/Segmentation Mappings
Detection and segmentation labels should not be collapsed blindly into image-level anomaly manifests. They need task-specific import logic.

Examples:
- `NORMAL_Tyres` can map to normal for tread-specific analysis.
- `BAD_Tyres`, `BALD_Tyres`, `bead_damage`, `cut`, and similar labels indicate defect/anomaly regions.
- Region labels must preserve boxes/masks for future localization.

## Required Review Before Use
- Check whether class folders contain augmented duplicates.
- Check whether split names are train/valid/test or custom.
- Check whether labels are mutually exclusive.
- Check for D1 leakage or duplicate images.
