# Pilot Input Status

Status: `blocked_missing_approved_input`

Preflight found no real approved pilot CSV under:

```text
data/external/manual_candidate_urls/
```

Created input contract files:

- `data/external/manual_candidate_urls/README.md`
- `data/external/manual_candidate_urls/approved_tyres_pilot_urls_template.csv`
- `docs/process/MANUAL_PILOT_INPUT_GUIDE.md`

Next required user action: create a real 20-50 row CSV such as:

```text
data/external/manual_candidate_urls/approved_tyres_pilot_urls_001.csv
```

Do not use the template row as a real candidate.
