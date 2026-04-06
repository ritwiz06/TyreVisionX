# Next Steps

## Highest Priority
1. Run one fresh canonical-pipeline experiment with `configs/train/train_resnet18.yaml` and save the resulting report under `artifacts/experiments/` plus `artifacts/reports/`.
2. Populate or regenerate D2 and D3 manifests if those datasets are available locally.
3. Replace README historical-result emphasis with canonical-pipeline results once they exist.

## Medium Priority
1. Decide whether root-level compatibility config files should remain for one more cycle or be removed after verification.
2. Decide whether root-level baseline modules should be converted into deprecation wrappers rather than full legacy compatibility paths.
3. Generate the professor-ready notebook set described in the cleanup blueprint.

## Lower Priority
1. Introduce a proper `src/tyrevisionx/` package when the active path is stable enough to move without churn.
2. Add a lightweight reporting dashboard once canonical experiment outputs are available.
3. Expand docs with a formal roadmap and professor discussion notes.
