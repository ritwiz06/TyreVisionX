# External Import Readiness Status

## Status
Import-preparation scaffolding exists. No external dataset was downloaded, imported, or merged.

## Ready for Review-Manifest Preparation
- `roboflow_good_tire_bad_tire`
- `roboflow_tires_defects_omar`

These are classification datasets with visible CC BY 4.0 licenses.

## Later Phase
- `roboflow_tire_tread_mark`
- `roboflow_defect_hemant`

These are detection datasets and should be reserved for localization/detection experiments.

## Blocked
- `roboflow_tire_college_segmentation`: exact source/license not verified.
- `roboflow_tire_quality_tirescanner`: Roboflow page references a Kaggle source and needs explicit license compatibility review.

## Import Command Template
```bash
python scripts/data/import_roboflow_export.py \
  --export_dir data/external/roboflow/<local_export_folder> \
  --dataset_id roboflow_good_tire_bad_tire \
  --out_csv data/interim/roboflow_good_tire_bad_tire_review_manifest.csv
```

The output is a review manifest, not a canonical TyreVisionX manifest.
