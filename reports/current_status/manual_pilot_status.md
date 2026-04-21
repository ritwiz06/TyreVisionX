# Manual Pilot Status

Updated: 2026-04-19

## Current Status

Status: `blocked_missing_approved_input`

Prompt 4 was not previously executed. The latest logs before this run pointed to the Prompt 3 web-foundation stage.

No real approved pilot input CSV was found under:

```text
data/external/manual_candidate_urls/
```

## Implemented In This Prompt

- Manual pilot input contract and template.
- End-to-end manual pilot orchestrator.
- Local-file and `file://` ingestion support.
- Pilot promotion script for reviewed candidates.
- Review pack export script.
- Review decision schema and promotion policy docs.
- Blocked-status reports for the missing input case.

## Pending

1. User provides a real approved 20-50 row CSV.
2. Run the pilot.
3. Generate a visual review pack from actual local candidate images.
4. Human review fills final decisions.
5. Promote `approved_likely_normal` rows into the curated manifest.
