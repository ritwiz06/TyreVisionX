"""Generate an editable query catalog for likely-normal tyre image collection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.web_collection.io import write_yaml
from src.web_collection.schemas import QuerySpec


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


QUERY_FAMILIES = [
    {
        "family": "clean_sidewall",
        "base": "clean tyre sidewall close up",
        "view": "sidewall",
        "contexts": ["studio product photo", "inspection lighting", "factory inspection", "plain background"],
        "notes": "Targets undamaged sidewall texture and lettering.",
    },
    {
        "family": "new_tread_closeup",
        "base": "new tyre tread close up",
        "view": "tread",
        "contexts": ["macro photo", "product catalog", "inspection view", "high resolution"],
        "notes": "Targets clean tread grooves and normal pattern repetition.",
    },
    {
        "family": "off_highway_exterior",
        "base": "off highway tyre exterior undamaged",
        "view": "exterior",
        "contexts": ["industrial vehicle", "warehouse", "product photo", "inspection"],
        "notes": "Adds larger industrial/off-highway tyre appearances.",
    },
    {
        "family": "mounted_product",
        "base": "mounted tyre product photo new",
        "view": "mounted_wheel",
        "contexts": ["vehicle wheel", "studio", "catalog", "side view"],
        "notes": "Targets mounted but visibly undamaged tyres.",
    },
    {
        "family": "industrial_surface",
        "base": "industrial tyre surface close up clean",
        "view": "surface_closeup",
        "contexts": ["manufacturing", "warehouse", "inspection light", "macro"],
        "notes": "Targets industrial surface texture with fewer consumer-photo distractions.",
    },
    {
        "family": "undamaged_inspection",
        "base": "undamaged tyre inspection view",
        "view": "inspection",
        "contexts": ["sidewall", "tread", "factory", "quality control"],
        "notes": "Targets images that resemble inspection views.",
    },
]


def build_default_queries() -> list[QuerySpec]:
    queries: list[QuerySpec] = []
    for family_index, family in enumerate(QUERY_FAMILIES, start=1):
        for variant_index, context in enumerate(family["contexts"], start=1):
            priority = 1 if variant_index <= 2 else 2
            query_text = f"{family['base']} {context}"
            queries.append(
                QuerySpec(
                    query_id=f"tyre_{family['family']}_{variant_index:02d}",
                    query_text=query_text,
                    product_family="tyre",
                    domain="manufacturing_inspection",
                    view_type=family["view"],
                    condition_adjective="clean undamaged new",
                    environment_context=context,
                    positive_keywords=[family["base"], context, "tyre"],
                    negative_keywords=NEGATIVE_KEYWORDS,
                    language="en",
                    priority=priority if family_index <= 4 else 3,
                    intended_use="candidate_likely_normal_review",
                    notes=family["notes"],
                )
            )
    return queries


def write_report(path: str | Path, queries: list[QuerySpec]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_priority: dict[int, int] = {}
    by_view: dict[str, int] = {}
    for query in queries:
        by_priority[query.priority] = by_priority.get(query.priority, 0) + 1
        by_view[query.view_type] = by_view.get(query.view_type, 0) + 1

    lines = [
        "# Query Catalog Report",
        "",
        "Status: generated scaffold for likely-normal tyre web-candidate discovery.",
        "",
        f"Total queries: `{len(queries)}`",
        "",
        "## Priority Strategy",
        "- Priority 1: high-yield likely-normal product/inspection images.",
        "- Priority 2: useful variants with more context noise.",
        "- Priority 3: broader coverage for later expansion.",
        "",
        "## Counts By Priority",
        "",
        "| Priority | Count |",
        "|---:|---:|",
    ]
    lines.extend(f"| {priority} | {count} |" for priority, count in sorted(by_priority.items()))
    lines.extend(["", "## Counts By View Type", "", "| View type | Count |", "|---|---:|"])
    lines.extend(f"| {view} | {count} |" for view, count in sorted(by_view.items()))
    lines.extend(
        [
            "",
            "## Use Notes",
            "- These queries are candidates for provider APIs or manual search workflows.",
            "- Results are research candidates, not labels.",
            "- Negative keywords should be reviewed before each provider run because syntax differs by provider.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TyreVisionX web-query catalog.")
    parser.add_argument("--out", default="configs/web_collection/query_catalog.yaml")
    parser.add_argument("--report", default="reports/web_collection/query_catalog_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = build_default_queries()
    payload = {
        "catalog_version": 1,
        "status": "editable_seed_catalog",
        "policy": "Use only with approved providers or manual workflows; do not scrape Google HTML.",
        "queries": [query.to_dict() for query in queries],
    }
    write_yaml(args.out, payload)
    write_report(args.report, queries)
    print(f"Wrote {len(queries)} queries to {args.out}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
