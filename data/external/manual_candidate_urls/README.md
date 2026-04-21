# Manual Candidate URLs

This folder is for researcher-approved TyreVisionX pilot input files.

Use `approved_tyres_pilot_urls_template.csv` as the schema. For the first pilot, place one reviewed CSV here with 20-50 rows when ready. The pipeline will ignore files with `template` in the filename when auto-detecting a real pilot input.

Supported input modes:
- URL rows using `source_url`
- local controlled rows using `local_source_path`
- `file://` rows using `source_url`

Do not add unapproved scraped URLs. Web candidates are not labels and still require filtering plus human review.
