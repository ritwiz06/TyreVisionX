# Roboflow License Audit

## Current License Status
| Dataset | Visible License | Status |
|---|---|---|
| Good Tire Bad Tire | CC BY 4.0 | Recorded; attribution required. |
| Tires Defects | CC BY 4.0 | Recorded; attribution required. |
| Tire Tread | Public Domain | Recorded; still audit version provenance. |
| defect (Hemant) | CC BY 4.0 | Recorded; attribution required. |
| tire (College segmentation) | Pending | Blocked until exact source is verified. |
| Tire Quality | Pending explicit verification | Blocked because Roboflow page references a Kaggle source and visible license was not explicit in the current page text. |

## Audit Rules
- Do not use a dataset if the license is unknown.
- Preserve source URL, author/workspace, Roboflow dataset ID, export version, and citation text.
- For CC BY 4.0, retain attribution in reports and any derived dataset cards.
- If a dataset references an upstream source, audit the upstream source as well.

## Decision
Only datasets with visible license and classification labels can proceed to import-preparation. No external dataset is approved for automatic merge.
