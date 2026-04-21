# Prompt Contract for Future Codex Work

Updated: 2026-04-19

Every future Codex prompt in this repository should follow this contract unless the user explicitly overrides it.

## Required Reading Before Work

1. Read `docs/codex/BASE_CONTEXT.md`.
2. Read `logs/work_logs/LATEST.md` if it exists.
3. Read `logs/process_logs/LATEST.md` if it exists.
4. Inspect only the files needed for the specific task before editing.

## Required Logs For Every Substantial Prompt

Create a new timestamped work log:

```text
logs/work_logs/WORK_LOG_<YYYYMMDD_HHMMSS>.md
```

Create a new timestamped process log:

```text
logs/process_logs/PROCESS_LOG_<YYYYMMDD_HHMMSS>.md
```

Update:

```text
logs/work_logs/LATEST.md
logs/process_logs/LATEST.md
```

## Required Work Log Contents

- user prompt text
- timestamp
- files inspected
- files changed/created
- issues found
- implementation summary
- why each change was made
- what was intentionally not changed
- outputs produced
- recommended next steps

## Required Process Log Contents

Explain beginner concepts related to the task:

- what each concept is
- why it is used here
- why a more advanced approach is not used yet
- how it relates to TyreVisionX

## Result Integrity Rules

- Never fabricate experiment metrics.
- Never claim a model was trained unless the prompt actually ran it.
- Historical results must be labeled as historical.
- Missing artifacts must be labeled as missing or pending recomputation.
- Web collection and anomaly detection must not be described as implemented until code and verification exist.

## Engineering Rules

- Prefer low-risk changes.
- Preserve useful prior work.
- Archive or mark legacy code instead of deleting it.
- Keep import breakage to a minimum.
- Document risky refactors instead of forcing them.

