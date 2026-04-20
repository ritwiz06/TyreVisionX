# Professor Meeting Summary

Date: 2026-04-06

## Project Overview
TyreVisionX is an early-stage academic research prototype for tyre defect inspection. The current validated focus is supervised binary classification of tyre images, supported by dataset manifests, evaluation artifacts, reports, logs, and research-facing notebooks.

## Current Stage
- canonical config-driven classification pipeline is now clear
- legacy baseline path is preserved but separated conceptually
- D1 is the only populated tracked manifest
- D2 and D3 remain partial / scaffolded
- anomaly detection, localization, segmentation, multi-view 3D, and knowledge reasoning remain future work

## Results So Far
Only real tracked repository evidence is included:
- historical Day 3 transfer-learning baseline reached strong D1 performance
  - accuracy `0.9804`
  - defect recall `0.9692`
  - defect F1 `0.9805`
  - AUROC `0.9985`
- historical Day 5 multi-seed study suggests:
  - baseline had the highest mean recall
  - augmentation had the strongest stability and AUPRC trade-off

Important caution:
- these are historical baseline results
- fresh canonical-pipeline outputs are still pending

## Key Bottlenecks
- limited tracked dataset breadth
- D2 / D3 manifests not yet populated
- strongest evidence still comes from historical rather than freshly rerun canonical experiments
- compatibility layers remain in the repo
- future directions are still roadmap-level rather than validated

## Future Work
### High priority
- fresh canonical-pipeline experiment run
- stronger experiment discipline and reporting
- limited-data generalization study
- cross-dataset evaluation once D2 / D3 are available

### Medium priority
- anomaly detection
- defect localization / detection
- segmentation

### Later priority
- multi-view 3D reconstruction
- defect projection onto 3D mesh
- knowledge graph / reasoning support

## How Professor Can Help
- refine the most meaningful problem formulation
- define a publishable next milestone
- suggest benchmark and evaluation priorities
- guide experimental design and scope control
- point to the most relevant literature direction
- help prioritize robustness vs anomaly detection vs localization vs 3D
- connect the prototype to industry-relevant deliverables

## Talking Points For The Meeting
1. The repository is now structurally cleaner and has one canonical classification pipeline.
2. The strongest tracked evidence is still historical baseline evidence on D1.
3. The project should still be described as an early-stage research prototype.
4. The next milestone should likely be one strong fresh canonical experiment, not more uncontrolled branching.
5. D2 / D3 support exists in config but is not yet experimentally validated.
6. The main scientific challenge is limited-data generalization, not just code implementation.
7. Binary classification is a solid starting point, but may need a stronger research angle soon.
8. The most plausible next branches are robustness, anomaly detection, or early localization.
9. I need guidance on what constitutes the most valuable publishable scope.
10. I want to align the next step with both academic relevance and practical inspection value.
