# Roboflow Universe Import Guide

## Purpose
Roboflow Universe can help TyreVisionX find external tyre datasets, but external data must be handled carefully. This guide explains the safe process.

## Step 1: Registry First
Every candidate dataset must be listed in:

```text
configs/data/external_dataset_registry.yaml
```

The registry records source URL, task type, license, classes, image count, label mapping, and recommended use.

## Step 2: License Audit
Do not use a dataset until the visible license and any upstream source license are understood. For CC BY 4.0 datasets, keep attribution. For pending licenses, hold the dataset.

## Step 3: Manual Export Only
This repo does not download Roboflow datasets automatically. The researcher may manually export a dataset after checking terms and license.

## Step 4: Review Manifest
After a local export exists, run:

```bash
python scripts/data/import_roboflow_export.py \
  --export_dir data/external/roboflow/<export_folder> \
  --dataset_id <registry_id> \
  --out_csv data/interim/<registry_id>_review_manifest.csv
```

This creates a review manifest. It does not merge into D1.

## Step 5: Audit Before Use
Before any training:

- verify label mapping
- inspect example images
- check split leakage
- check duplicates against D1
- preserve source and license metadata

## Current Rule
Only classification datasets are near-term import candidates. Detection and segmentation datasets are later-phase localization resources.
