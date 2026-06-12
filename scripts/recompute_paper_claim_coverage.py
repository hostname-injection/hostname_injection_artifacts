#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


EXPECTED_IDS = (
    "contribution_deferred_reuse",
    "contribution_hib",
    "contribution_ccd",
    "contribution_operational_eval",
    "contribution_scope_repro",
    "eq1",
    "proposition5_1",
    "lemma_c1",
    "figure1",
    "figure2",
    "figure3",
    "figure4",
    "figure5",
    "figure6",
    "figure7",
    "table1",
    "table2",
    "table3",
    "table4",
    "table5",
    "table6",
    "table7",
    "table8",
    "table9",
    "table10",
    "table11",
    "table12",
    "appendix_a",
    "appendix_b",
    "appendix_c",
    "appendix_d",
    "appendix_e",
    "appendix_f",
)
VALID_STATUSES = {
    "documentation",
    "executable_public",
    "external_full_replay",
    "release_safe_aggregate",
    "sample_functional",
}
EXPECTED_EXTERNAL_REQUIRED_IDS = {
    "appendix_a",
    "appendix_b",
    "appendix_c",
    "appendix_d",
    "appendix_e",
    "appendix_f",
    "contribution_deferred_reuse",
    "contribution_hib",
    "contribution_operational_eval",
    "contribution_scope_repro",
    "figure1",
    "figure5",
    "figure6",
    "figure7",
    "table3",
    "table4",
    "table5",
    "table6",
    "table7",
    "table8",
    "table9",
    "table10",
    "table11",
    "table12",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def require_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def require_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be true or false")
    return value


def require_string_list(value: object, *, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    return [require_string(item, path=f"{path}[{index}]") for index, item in enumerate(value)]


def manifest_text(manifest: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for claim in manifest.get("claims", []):
        if isinstance(claim, dict):
            parts.extend(str(claim.get(key, "")) for key in ("claim", "script", "expected"))
    parts.extend(str(item) for item in manifest.get("external_completion_items", []))
    return "\n".join(parts).lower()


def validate_item(
    raw_item: object,
    *,
    index: int,
    root: Path,
    manifest: Mapping[str, Any],
    manifest_blob: str,
    external_blob: str,
) -> dict[str, Any]:
    item = require_mapping(raw_item, path=f"coverage_items[{index}]")
    item_id = require_string(item.get("item_id"), path=f"coverage_items[{index}].item_id")
    status = require_string(item.get("coverage_status"), path=f"{item_id}.coverage_status")
    if status not in VALID_STATUSES:
        raise ValueError(f"{item_id}.coverage_status must be one of {sorted(VALID_STATUSES)}")
    paths = require_string_list(item.get("artifact_paths"), path=f"{item_id}.artifact_paths")
    missing_paths = [rel for rel in paths if not (root / rel).exists()]
    if missing_paths:
        raise ValueError(f"{item_id} references missing artifact paths: {missing_paths}")
    commands = require_string_list(item.get("verification_commands"), path=f"{item_id}.verification_commands")
    keywords = require_string_list(item.get("manifest_claim_keywords"), path=f"{item_id}.manifest_claim_keywords")
    matched_keywords = [keyword for keyword in keywords if keyword.lower() in manifest_blob]
    if not matched_keywords:
        raise ValueError(f"{item_id} does not match any manifest claim keyword: {keywords}")
    external_required = require_bool(item.get("external_required"), path=f"{item_id}.external_required")
    external_keyword = str(item.get("external_completion_keyword", "")).strip()
    if external_required:
        if not external_keyword:
            raise ValueError(f"{item_id} requires external completion but has no keyword")
        if external_keyword.lower() not in external_blob:
            raise ValueError(f"{item_id} external keyword not found in manifest completion items: {external_keyword!r}")
    elif external_keyword:
        raise ValueError(f"{item_id} has an external keyword but external_required is false")
    return {
        "item_id": item_id,
        "paper_ref": require_string(item.get("paper_ref"), path=f"{item_id}.paper_ref"),
        "claim_summary": require_string(item.get("claim_summary"), path=f"{item_id}.claim_summary"),
        "coverage_status": status,
        "artifact_paths": paths,
        "verification_commands": commands,
        "matched_manifest_keywords": matched_keywords,
        "external_required": external_required,
        "external_completion_keyword": external_keyword,
        "privacy_boundary": require_string(item.get("privacy_boundary"), path=f"{item_id}.privacy_boundary"),
    }


def validate_coverage(data: Mapping[str, Any], *, root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    blob = manifest_text(manifest)
    external_blob = "\n".join(str(item) for item in manifest.get("external_completion_items", [])).lower()
    raw_items = data.get("coverage_items")
    if not isinstance(raw_items, list):
        raise ValueError("coverage_items must be a list")
    items = [
        validate_item(
            raw_item,
            index=index,
            root=root,
            manifest=manifest,
            manifest_blob=blob,
            external_blob=external_blob,
        )
        for index, raw_item in enumerate(raw_items)
    ]
    ids = [item["item_id"] for item in items]
    duplicates = sorted(item_id for item_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate coverage item ids: {duplicates}")
    if tuple(ids) != EXPECTED_IDS:
        missing = [item_id for item_id in EXPECTED_IDS if item_id not in ids]
        extra = [item_id for item_id in ids if item_id not in EXPECTED_IDS]
        raise ValueError(f"coverage item order/content mismatch; missing={missing}, extra={extra}")
    observed_external = {item["item_id"] for item in items if item["external_required"]}
    if observed_external != EXPECTED_EXTERNAL_REQUIRED_IDS:
        missing = sorted(EXPECTED_EXTERNAL_REQUIRED_IDS - observed_external)
        extra = sorted(observed_external - EXPECTED_EXTERNAL_REQUIRED_IDS)
        raise ValueError(f"external_required item mismatch; missing={missing}, extra={extra}")
    status_counts = Counter(item["coverage_status"] for item in items)
    external_required = [item["item_id"] for item in items if item["external_required"]]
    return {
        "items": items,
        "derived": {
            "n_coverage_items": len(items),
            "n_contribution_items": sum(1 for item in ids if item.startswith("contribution_")),
            "n_figures": sum(1 for item in ids if item.startswith("figure")),
            "n_tables": sum(1 for item in ids if item.startswith("table")),
            "n_appendices": sum(1 for item in ids if item.startswith("appendix_")),
            "n_equation_or_formal_items": sum(1 for item in ids if item in {"eq1", "proposition5_1", "lemma_c1"}),
            "coverage_status_counts": dict(sorted(status_counts.items())),
            "external_required_count": len(external_required),
            "external_required_items": external_required,
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    counts_path = root / args.counts
    data = load_json(counts_path)
    coverage = validate_coverage(data, root=root, manifest_path=root / args.manifest)
    return {
        "status": "pass",
        "counts_file": str(counts_path),
        "source": data.get("source", "paper_claim_coverage_inventory"),
        "paper": data.get("paper", "IEEE_S_P_Hostnames.pdf"),
        "scope": data.get("scope", ""),
        "manifest": str(root / args.manifest),
        "derived": coverage["derived"],
        "coverage_items": coverage["items"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute the release-safe paper-claim coverage matrix.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--counts",
        default="paper_claim_coverage/paper_claim_coverage.json",
        help="Paper claim coverage matrix.",
    )
    parser.add_argument("--manifest", default="ARTIFACT_MANIFEST.json", help="Artifact manifest.")
    parser.add_argument("--out", default=None, help="Optional path for the JSON report.")
    args = parser.parse_args()

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
