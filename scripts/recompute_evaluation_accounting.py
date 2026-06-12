#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_UNITS = (
    "replay_entry",
    "verified_positive",
    "live_comparison_item",
    "composite_alert",
    "tenant_visible_alert",
    "unresolved",
)
EXPECTED_BOUNDARY_CLAIMS = ("replay_accuracy", "baselines", "live_overlap", "sink_evidence")
WITHHELD_TERMS = (
    "tenant identities",
    "reversible mappings",
    "raw operational logs",
    "raw tenant logs",
    "exact sensitive strings",
    "response-owner records",
    "exploitable sink details",
    "callback domains",
    "credentials",
    "tenant names",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be true or false")
    return value


def require_int(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def require_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def require_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def require_string_list(value: object, *, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    return [require_string(item, path=f"{path}[{index}]") for index, item in enumerate(value)]


def validate_units(units_value: object) -> list[dict[str, Any]]:
    if not isinstance(units_value, list):
        raise ValueError("table2_evidence_units must be a list")
    if len(units_value) != len(EXPECTED_UNITS):
        raise ValueError(f"table2_evidence_units must contain exactly {len(EXPECTED_UNITS)} units")
    validated: list[dict[str, Any]] = []
    for index, raw_unit in enumerate(units_value):
        unit = require_mapping(raw_unit, path=f"table2_evidence_units[{index}]")
        unit_id = require_string(unit.get("unit_id"), path=f"table2_evidence_units[{index}].unit_id")
        if unit_id != EXPECTED_UNITS[index]:
            raise ValueError(f"table2_evidence_units[{index}].unit_id must be {EXPECTED_UNITS[index]}")
        denominator_for = require_string_list(
            unit.get("denominator_for"), path=f"table2_evidence_units[{index}].denominator_for"
        )
        clean = {
            "unit_id": unit_id,
            "paper_unit": require_string(unit.get("paper_unit"), path=f"table2_evidence_units[{index}].paper_unit"),
            "meaning": require_string(unit.get("meaning"), path=f"table2_evidence_units[{index}].meaning"),
            "denominator_for": denominator_for,
            "resolved_replay_denominator": require_bool(
                unit.get("resolved_replay_denominator"),
                path=f"table2_evidence_units[{index}].resolved_replay_denominator",
            ),
            "live_stream_unit": require_bool(unit.get("live_stream_unit"), path=f"table2_evidence_units[{index}].live_stream_unit"),
            "requires_downstream_or_harness_support": require_bool(
                unit.get("requires_downstream_or_harness_support"),
                path=f"table2_evidence_units[{index}].requires_downstream_or_harness_support",
            ),
        }
        if unit_id == "verified_positive" and require_int(unit.get("reported_count"), path="verified_positive.reported_count") != 363401:
            raise ValueError("verified_positive.reported_count must be 363401")
        if unit_id == "tenant_visible_alert" and require_int(unit.get("tenant_slice_size"), path="tenant_visible_alert.tenant_slice_size") != 50:
            raise ValueError("tenant_visible_alert.tenant_slice_size must be 50")
        if unit_id == "unresolved":
            if require_bool(unit.get("excluded_from_resolved_replay_denominators"), path="unresolved.excluded") is not True:
                raise ValueError("unresolved entries must be excluded from resolved replay denominators")
            if require_bool(unit.get("retained_in_uncertainty_views"), path="unresolved.retained") is not True:
                raise ValueError("unresolved entries must be retained in uncertainty views")
        validated.append(clean)
    resolved = [unit["unit_id"] for unit in validated if unit["resolved_replay_denominator"]]
    if resolved != ["replay_entry", "verified_positive"]:
        raise ValueError("only replay_entry and verified_positive may be resolved replay denominators")
    return validated


def validate_boundary(boundary_value: object) -> list[dict[str, Any]]:
    if not isinstance(boundary_value, list):
        raise ValueError("table11_reproducibility_boundary must be a list")
    if len(boundary_value) != len(EXPECTED_BOUNDARY_CLAIMS):
        raise ValueError(
            f"table11_reproducibility_boundary must contain exactly {len(EXPECTED_BOUNDARY_CLAIMS)} rows"
        )
    validated: list[dict[str, Any]] = []
    for index, raw_row in enumerate(boundary_value):
        row = require_mapping(raw_row, path=f"table11_reproducibility_boundary[{index}]")
        claim_id = require_string(row.get("claim_id"), path=f"table11_reproducibility_boundary[{index}].claim_id")
        if claim_id != EXPECTED_BOUNDARY_CLAIMS[index]:
            raise ValueError(f"table11_reproducibility_boundary[{index}].claim_id must be {EXPECTED_BOUNDARY_CLAIMS[index]}")
        released = require_string_list(
            row.get("released_or_recomputable"),
            path=f"table11_reproducibility_boundary[{index}].released_or_recomputable",
        )
        withheld = require_string_list(row.get("withheld"), path=f"table11_reproducibility_boundary[{index}].withheld")
        released_text = " ".join(released).lower()
        for term in WITHHELD_TERMS:
            if term in released_text:
                raise ValueError(f"{claim_id} releases withheld privacy term: {term}")
        keyword = require_string(
            row.get("external_completion_keyword"),
            path=f"table11_reproducibility_boundary[{index}].external_completion_keyword",
        )
        if require_bool(row.get("external_completion_required"), path=f"{claim_id}.external_completion_required") is not True:
            raise ValueError(f"{claim_id}.external_completion_required must be true")
        validated.append(
            {
                "claim_id": claim_id,
                "paper_claim": require_string(row.get("paper_claim"), path=f"{claim_id}.paper_claim"),
                "released_or_recomputable": released,
                "withheld": withheld,
                "source_artifact_status": require_string(row.get("source_artifact_status"), path=f"{claim_id}.source_artifact_status"),
                "external_completion_required": True,
                "external_completion_keyword": keyword,
            }
        )
    return validated


def validate_manifest_alignment(boundary: list[dict[str, Any]], manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None:
        return {"checked": False, "matched_external_boundaries": []}
    manifest = load_json(manifest_path)
    external_items = [str(item) for item in manifest.get("external_completion_items", [])]
    matched: list[str] = []
    missing: list[str] = []
    external_text = "\n".join(external_items).lower()
    for row in boundary:
        keyword = row["external_completion_keyword"].lower()
        if keyword in external_text:
            matched.append(row["claim_id"])
        else:
            missing.append(f"{row['claim_id']} missing external completion item containing {row['external_completion_keyword']!r}")
    if missing:
        raise ValueError("; ".join(missing))
    return {
        "checked": True,
        "manifest": str(manifest_path),
        "matched_external_boundaries": matched,
        "external_completion_item_count": len(external_items),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    counts_path = Path(args.counts)
    data = load_json(counts_path)
    units = validate_units(data.get("table2_evidence_units"))
    boundary = validate_boundary(data.get("table11_reproducibility_boundary"))
    privacy_invariants = require_string_list(data.get("privacy_invariants"), path="privacy_invariants")
    manifest_alignment = validate_manifest_alignment(boundary, None if args.no_manifest else Path(args.manifest))
    derived = {
        "n_evidence_units": len(units),
        "resolved_replay_denominator_units": [unit["unit_id"] for unit in units if unit["resolved_replay_denominator"]],
        "live_stream_units": [unit["unit_id"] for unit in units if unit["live_stream_unit"]],
        "n_reproducibility_boundary_rows": len(boundary),
        "external_completion_required_for": [row["claim_id"] for row in boundary if row["external_completion_required"]],
        "withheld_category_count": sum(len(row["withheld"]) for row in boundary),
        "privacy_invariant_count": len(privacy_invariants),
    }
    return {
        "status": "pass",
        "counts_file": str(counts_path),
        "source": data.get("source", "published_evaluation_accounting_tables"),
        "paper_sections": data.get("paper_sections", []),
        "table2_evidence_units": units,
        "table11_reproducibility_boundary": boundary,
        "manifest_alignment": manifest_alignment,
        "derived": derived,
        "privacy_invariants": privacy_invariants,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute release-safe evaluation-accounting and reproducibility-boundary checks.")
    parser.add_argument(
        "--counts",
        default="evaluation_accounting/paper_evaluation_accounting.json",
        help="Release-safe Table 2/Table 11 accounting summary.",
    )
    parser.add_argument("--manifest", default="ARTIFACT_MANIFEST.json", help="Artifact manifest to align with Table 11.")
    parser.add_argument("--no-manifest", action="store_true", help="Skip manifest external-boundary alignment.")
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
