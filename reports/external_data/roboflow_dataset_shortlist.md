# Roboflow Dataset Shortlist

| ID | Dataset | Task | Classes | Immediate Use Phase | Recommended Action |
|---|---|---|---|---|---|
| `roboflow_good_tire_bad_tire` | Good Tire Bad Tire | Classification | `good`, `defective` | `anomaly_support` | Candidate after license citation, duplicate audit, and label review. |
| `roboflow_tires_defects_omar` | Tires Defects | Classification | `Good`, `Defected` | `supervised_benchmark` | Candidate external benchmark; keep separate from D1. |
| `roboflow_tire_tread_mark` | Tire Tread | Object detection | `BAD_Tyres`, `BALD_Tyres`, `NORMAL_Tyres` | `localization_future` | Later-phase tread/localization benchmark. |
| `roboflow_defect_hemant` | defect (Hemant) | Object detection | `bead_damage`, `CBU`, `cut`, `tr` | `localization_future` | Later-phase damage localization dataset. |
| `roboflow_tire_college_segmentation` | tire (College segmentation) | Segmentation | `tire` | `hold` | Hold until exact project URL and license are verified. |
| `roboflow_tire_quality_tirescanner` | Tire Quality | Classification | `good`, `defective` | `hold` | Hold until explicit Roboflow and original Kaggle license compatibility are verified. |

## Near-Term Candidates
The two near-term candidates are classification datasets:

1. Good Tire Bad Tire
2. Tires Defects

They are still not automatically trusted. They must go through import review, duplicate checks, label checks, and separate evaluation.

## Later-Phase Candidates
Tire Tread and defect (Hemant) are useful for localization/detection research, not for immediate good-only anomaly training.
