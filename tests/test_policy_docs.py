from __future__ import annotations

from pathlib import Path


def test_policy_docs_exist_and_forbid_auto_labeling() -> None:
    paths = [
        Path("docs/process/SAFE_WEB_NORMAL_EXPANSION_POLICY.md"),
        Path("docs/process/MODEL_ASSISTED_TRIAGE_POLICY.md"),
        Path("reports/web_collection/confidence_framework.md"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert "label" in text
        assert "review" in text
    assert "must not automatically label" in paths[0].read_text(encoding="utf-8").lower()


def test_latest_logs_point_to_existing_files() -> None:
    for latest_path in [Path("logs/work_logs/LATEST.md"), Path("logs/process_logs/LATEST.md")]:
        text = latest_path.read_text(encoding="utf-8")
        refs = [part.strip("`") for part in text.split() if part.startswith("`logs/")]
        assert refs, f"No log reference found in {latest_path}"
        for ref in refs:
            assert Path(ref).exists(), ref
