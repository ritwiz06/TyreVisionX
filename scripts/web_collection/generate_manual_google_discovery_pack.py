"""Generate a browser-ready manual Google discovery pack.

This script does not scrape Google. It creates query links and a prefilled CSV
template so a researcher can manually browse, approve, and record candidate
tyre image sources.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


NEGATIVE_KEYWORDS = [
    "damage",
    "damaged",
    "cracked",
    "crack",
    "puncture",
    "burst",
    "worn out",
    "scrap",
    "used",
    "failure",
    "defect",
    "flat tire",
    "accident",
]


@dataclass(frozen=True)
class ManualGoogleQuery:
    query_id: str
    group: str
    query_text: str
    view_type: str
    priority: int
    notes: str

    @property
    def browser_query(self) -> str:
        negatives = " ".join(f'-"{keyword}"' for keyword in NEGATIVE_KEYWORDS)
        return f"{self.query_text} {negatives}"

    @property
    def google_url(self) -> str:
        return f"https://www.google.com/search?tbm=isch&q={quote_plus(self.browser_query)}"


def build_manual_google_queries() -> list[ManualGoogleQuery]:
    """Return 40 tyre-focused manual browser queries grouped by review intent."""

    grouped: list[tuple[str, str, list[str], str]] = [
        (
            "tread",
            "tread",
            [
                "new tyre tread close up high resolution",
                "clean tire tread macro product photo",
                "undamaged tyre tread pattern close up",
                "new car tyre tread inspection lighting",
                "premium tyre tread close up studio",
                "normal tire tread grooves close up",
                "unused tyre tread surface macro",
                "clean truck tyre tread close up",
            ],
            "Prefer clear groove geometry and normal tread repetition.",
        ),
        (
            "sidewall",
            "sidewall",
            [
                "clean tyre sidewall close up studio",
                "new tire sidewall product photo",
                "undamaged tyre sidewall lettering close up",
                "clean car tyre sidewall inspection view",
                "black tyre sidewall macro plain background",
                "normal tire sidewall texture close up",
                "new radial tyre sidewall close up",
                "clean truck tyre sidewall product photo",
            ],
            "Prefer intact sidewalls, readable lettering, and no visible cracks.",
        ),
        (
            "mounted_tyre",
            "mounted_wheel",
            [
                "new tyre mounted on wheel product photo",
                "clean tire on rim studio photo",
                "undamaged mounted tyre close up",
                "new car wheel tyre plain background",
                "clean mounted truck tire product photo",
                "new alloy wheel tyre side view",
                "normal vehicle tyre mounted close up",
                "clean wheel and tyre inspection photo",
            ],
            "Useful for whole-tyre context; avoid vehicles with accidents or heavy wear.",
        ),
        (
            "inspection_like",
            "inspection",
            [
                "tyre inspection close up normal surface",
                "clean tyre inspection lighting tread",
                "factory tyre inspection normal close up",
                "undamaged tire inspection view",
                "new tyre quality inspection photo",
                "normal tyre surface inspection close up",
                "clean tire close up inspection plain background",
                "tyre manufacturing inspection normal tread",
            ],
            "Prefer neutral lighting and inspection-like framing.",
        ),
        (
            "industrial_off_highway",
            "exterior",
            [
                "new off highway tyre exterior close up",
                "clean industrial tyre product photo",
                "undamaged heavy equipment tire exterior",
                "new agriculture tyre tread close up",
                "clean loader tire sidewall product photo",
                "new mining tyre exterior studio",
                "normal industrial tire tread close up",
                "clean off road tyre inspection view",
            ],
            "Broader tyre variety; reject dirt, cuts, weathering, or visible wear.",
        ),
    ]

    queries: list[ManualGoogleQuery] = []
    for group, view_type, texts, notes in grouped:
        for index, text in enumerate(texts, start=1):
            queries.append(
                ManualGoogleQuery(
                    query_id=f"manual_google_{group}_{index:02d}",
                    group=group,
                    query_text=text,
                    view_type=view_type,
                    priority=1 if index <= 4 else 2,
                    notes=notes,
                )
            )
    return queries


def write_query_batches(path: Path, queries: list[ManualGoogleQuery]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = _group_queries(queries)
    lines = [
        "# Manual Google Query Batches",
        "",
        "Status: browser-ready manual discovery pack. No scraping is used or allowed.",
        "",
        f"Total queries: `{len(queries)}`",
        "",
        "Use these links in a normal browser, inspect results manually, and copy only approved image/page URLs into the CSV.",
        "",
        "Negative keywords included in every browser query:",
        "",
        ", ".join(f"`{keyword}`" for keyword in NEGATIVE_KEYWORDS),
        "",
    ]
    for group, items in groups.items():
        lines.extend([f"## {group.replace('_', ' ').title()}", ""])
        for query in items:
            lines.append(
                f"- `{query.query_id}`: [{query.query_text}]({query.google_url})  "
                f"View: `{query.view_type}`. Notes: {query.notes}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_checklist(path: Path, queries: list[ManualGoogleQuery]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual Google Query Checklist",
        "",
        "Use this checklist while browsing. A candidate is not a label; it is only a review candidate.",
        "",
        "## Approve If",
        "- The tyre appears likely normal and undamaged.",
        "- The image is clear enough for texture/tread/sidewall review.",
        "- The tyre is the main subject, not a tiny background object.",
        "- The page URL and source URL can be recorded.",
        "- License or attribution notes are visible or can be marked as unknown.",
        "",
        "## Reject If",
        "- The tyre has visible cracks, punctures, burst damage, severe wear, or repairs.",
        "- The result is a diagram, icon, render, meme, ad collage, or irrelevant object.",
        "- The image is too small, blurred, watermarked over the tyre, or heavily occluded.",
        "- The source page appears unreliable or the image cannot be traced to a page URL.",
        "",
        "## Browsing Progress",
        "",
        "| Done | Query ID | Group | Priority | Query Text | Notes |",
        "|---|---|---|---:|---|---|",
    ]
    for query in queries:
        lines.append(
            f"| [ ] | `{query.query_id}` | {query.group} | {query.priority} | {query.query_text} | {query.notes} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_prefilled_csv(path: Path, queries: list[ManualGoogleQuery]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "provider",
        "query_id",
        "query_text",
        "source_url",
        "page_url",
        "local_source_path",
        "license_name",
        "license_url",
        "attribution_text",
        "retrieval_timestamp",
        "product_type",
        "view_type",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, query in enumerate(queries, start=1):
            writer.writerow(
                {
                    "candidate_id": f"pilot001_manual_google_{index:03d}",
                    "provider": "manual_google_discovery",
                    "query_id": query.query_id,
                    "query_text": query.query_text,
                    "source_url": "",
                    "page_url": "",
                    "local_source_path": "",
                    "license_name": "",
                    "license_url": "",
                    "attribution_text": "",
                    "retrieval_timestamp": "",
                    "product_type": "tyre",
                    "view_type": query.view_type,
                    "notes": "",
                }
            )


def write_notebook(path: Path, queries: list[ManualGoogleQuery]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "query_id": query.query_id,
            "group": query.group,
            "query_text": query.query_text,
            "view_type": query.view_type,
            "priority": query.priority,
            "google_url": query.google_url,
            "notes": query.notes,
        }
        for query in queries
    ]
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Manual Google Discovery Pack\n",
                    "\n",
                    "Purpose: help the researcher browse 40 tyre-focused Google image queries manually, record approved candidate URLs, and keep every image as a review candidate rather than a label.\n",
                    "\n",
                    "Inputs: generated query batches and the prefilled candidate CSV.\n",
                    "\n",
                    "Outputs: a manually completed approved candidate CSV for the first tyre web-candidate pilot.\n",
                    "\n",
                    "Status: browser-ready pack only. No scraping and no automatic labeling.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "\n",
                    "queries = pd.DataFrame(",
                    repr(rows),
                    ")\n",
                    "queries.head()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "queries.groupby('group').size().rename('query_count')\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## How To Use\n",
                    "\n",
                    "1. Open `reports/web_collection/manual_google_query_batches.md`.\n",
                    "2. Start with priority-1 tread and sidewall queries.\n",
                    "3. For each approved candidate, fill `source_url`, `page_url`, license fields if known, and `notes` in `data/external/manual_candidate_urls/approved_tyres_pilot_urls_001_prefilled.csv`.\n",
                    "4. Do not mark any row as good. Human review happens after filtering and review-pack generation.\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def _group_queries(queries: list[ManualGoogleQuery]) -> dict[str, list[ManualGoogleQuery]]:
    groups: dict[str, list[ManualGoogleQuery]] = {}
    for query in queries:
        groups.setdefault(query.group, []).append(query)
    return groups


def generate_pack(root: Path = ROOT) -> dict[str, Path | int]:
    queries = build_manual_google_queries()
    outputs = {
        "query_count": len(queries),
        "csv_rows": len(queries),
        "query_batches": root / "reports/web_collection/manual_google_query_batches.md",
        "checklist": root / "reports/web_collection/manual_google_query_checklist.md",
        "prefilled_csv": root / "data/external/manual_candidate_urls/approved_tyres_pilot_urls_001_prefilled.csv",
        "notebook": root / "notebooks/04_web_data_curation/manual_google_discovery_pack.ipynb",
    }
    write_query_batches(outputs["query_batches"], queries)  # type: ignore[arg-type]
    write_checklist(outputs["checklist"], queries)  # type: ignore[arg-type]
    write_prefilled_csv(outputs["prefilled_csv"], queries)  # type: ignore[arg-type]
    write_notebook(outputs["notebook"], queries)  # type: ignore[arg-type]
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the manual Google discovery pack.")
    parser.add_argument("--root", default=str(ROOT), help="Repository root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_pack(Path(args.root))
    print(f"Generated {outputs['query_count']} manual Google queries.")
    print(f"Generated {outputs['csv_rows']} prefilled candidate rows.")
    for key in ["query_batches", "checklist", "prefilled_csv", "notebook"]:
        print(f"{key}: {outputs[key]}")


if __name__ == "__main__":
    main()
