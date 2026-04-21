# Roboflow Integration Recommendation

## Recommendation
Use Roboflow datasets only as separately tracked external datasets, not as automatic additions to D1.

## Near-Term
- Prepare classification exports only after manual download by the researcher.
- Use `scripts/data/import_roboflow_export.py` to create review manifests.
- Keep external manifests separate under `data/interim/` until reviewed.

## Hold
- Hold `Tire Quality` until license compatibility is confirmed against both Roboflow and the referenced Kaggle source.
- Hold `tire (College segmentation)` until exact source URL and license are known.

## Later
Detection/segmentation datasets can support localization after the anomaly classifier direction is clearer. They should not be mixed into current image-level D1 anomaly metrics.
