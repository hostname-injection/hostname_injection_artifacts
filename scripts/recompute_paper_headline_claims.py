#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


EXPECTED_IDS = (
    "abstract_replay_scale",
    "abstract_fixed_fpr_recall_latency",
    "abstract_live_added_value",
    "hib_label_profile",
    "decision_stability_k6",
    "synthetic_training_gap",
    "scope_public_and_static",
    "hostile_mimicry_range",
    "production_throughput_tail",
)


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


def require_number(value: object, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def find_row(rows: object, key: str, value: str, *, path: str) -> Mapping[str, Any]:
    if not isinstance(rows, list):
        raise ValueError(f"{path} must be a list")
    for row in rows:
        if isinstance(row, dict) and row.get(key) == value:
            return row
    raise ValueError(f"{path} does not contain row with {key}={value!r}")


def percent(part: int, total: int) -> float:
    if total == 0:
        raise ValueError("cannot compute percentage with denominator 0")
    return 100.0 * part / total


def as_pp(value: float) -> float:
    return round(100.0 * value, 1)


def build_observed(root: Path) -> dict[str, dict[str, Any]]:
    hib = load_json(root / "hib_profile" / "paper_hib_profile_counts.json")
    metrics = load_json(root / "paper_metric_tables" / "paper_metric_tables.json")
    latency = load_json(root / "production_latency" / "paper_production_latency_counts.json")
    live = load_json(root / "live_overlap" / "paper_live_overlap_counts.json")
    stability = load_json(root / "stability_scope" / "paper_stability_scope_counts.json")
    source = load_json(root / "source_reachability" / "paper_source_reachability_counts.json")
    public_scope = load_json(root / "public_scope" / "paper_public_scope_counts.json")

    replay = require_mapping(hib.get("replay"), path="hib.replay")
    labels = require_mapping(replay.get("label_counts"), path="hib.replay.label_counts")
    ccd_row = find_row(metrics.get("table5_baseline_audit"), "method", "CCD", path="metrics.table5_baseline_audit")
    generator_llm = find_row(metrics.get("generator_comparison"), "training_source", "LLM priors", path="metrics.generator_comparison")
    generator_real = find_row(
        metrics.get("generator_comparison"),
        "training_source",
        "Real-payload training",
        path="metrics.generator_comparison",
    )
    appendix_f = require_mapping(metrics.get("appendix_f_synthetic_validity"), path="metrics.appendix_f_synthetic_validity")
    table12 = metrics.get("table12_hostile_mimicry")
    if not isinstance(table12, list) or not table12:
        raise ValueError("metrics.table12_hostile_mimicry must be a non-empty list")

    full_path_latency = require_mapping(latency.get("full_path_latency"), path="latency.full_path_latency")
    scoring_kernel_latency = require_mapping(latency.get("scoring_kernel_latency"), path="latency.scoring_kernel_latency")
    live_table = require_mapping(live.get("table7"), path="live.table7")
    live_overlap = require_mapping(live_table.get("overlap"), path="live.table7.overlap")
    ccd_only = require_mapping(live_table.get("ccd_only"), path="live.table7.ccd_only")
    regex_waf_only = require_mapping(live_table.get("regex_waf_only"), path="live.table7.regex_waf_only")
    stability_block = require_mapping(stability.get("decision_stability"), path="stability.decision_stability")
    source_corpus = require_mapping(source.get("corpus"), path="source.corpus")
    source_tools = require_mapping(source.get("tools"), path="source.tools")
    codeql = require_mapping(source_tools.get("codeql"), path="source.tools.codeql")
    semgrep = require_mapping(source_tools.get("semgrep"), path="source.tools.semgrep")
    public_reports = require_mapping(public_scope.get("public_reports"), path="public_scope.public_reports")

    benign = int(labels["resolved_benign"])
    positive = int(labels["verified_executable_semantics"])
    unresolved = int(labels["unresolved"])
    llm_recall = require_number(generator_llm.get("recall_at_1e_4"), path="generator.LLM priors.recall_at_1e_4")
    real_recall = require_number(generator_real.get("recall_at_1e_4"), path="generator.Real-payload training.recall_at_1e_4")
    ccd_recalls = [require_number(row.get("ccd_recall_1e_4"), path=f"table12[{idx}].ccd_recall_1e_4") for idx, row in enumerate(table12)]
    baseline_recalls = [
        require_number(row.get("best_baseline_recall_1e_4"), path=f"table12[{idx}].best_baseline_recall_1e_4")
        for idx, row in enumerate(table12)
    ]

    return {
        "abstract_replay_scale": {
            "replay_rows": replay.get("n_rows"),
            "replay_rows_millions_rounded": round(require_number(replay.get("n_rows"), path="hib.replay.n_rows") / 1_000_000, 1),
            "tenant_count": replay.get("n_tenants"),
        },
        "abstract_fixed_fpr_recall_latency": {
            "ccd_tpr_at_1e_4": ccd_row.get("tpr_at_1e_4"),
            "ccd_observed_fpr": ccd_row.get("fpr"),
            "fpr_budget": 1e-4,
            "fpr_within_budget": require_number(ccd_row.get("fpr"), path="table5.CCD.fpr") <= 1.05e-4,
            "full_path_median_ms": full_path_latency.get("median_ms"),
        },
        "abstract_live_added_value": {
            "overlap_verified_positive": live_overlap.get("verified_positive"),
            "ccd_only_verified_positive": ccd_only.get("verified_positive"),
            "regex_waf_only_verified_positive": regex_waf_only.get("verified_positive"),
            "ccd_total_verified_live_positives": int(live_overlap["verified_positive"]) + int(ccd_only["verified_positive"]),
        },
        "hib_label_profile": {
            "resolved_benign": benign,
            "verified_executable_semantics": positive,
            "unresolved": unresolved,
            "resolved_replay_denominator": benign + positive,
            "positive_prevalence_percent_rounded": round(percent(positive, benign + positive), 2),
        },
        "decision_stability_k6": {
            "alpha": stability_block.get("alpha"),
            "edit_budget_k": stability_block.get("edit_budget_k"),
            "combined_stable_detected_positive_coverage": stability_block.get("combined_stable_detected_positive_coverage"),
            "sec_alone_stable_detected_positive_coverage": stability_block.get("sec_alone_stable_detected_positive_coverage"),
            "native_feature_baseline_min_coverage": stability_block.get("native_feature_baseline_min_coverage"),
            "native_feature_baseline_max_coverage": stability_block.get("native_feature_baseline_max_coverage"),
        },
        "synthetic_training_gap": {
            "llm_prior_recall_at_1e_4": llm_recall,
            "real_payload_training_recall_at_1e_4": real_recall,
            "real_training_gain_over_llm_pp": as_pp(real_recall - llm_recall),
            "tuned_generator_recall_at_1e_4": appendix_f.get("tuned_generator_recall_at_1e_4"),
            "real_training_gap_tuned_pp": appendix_f.get("real_training_gap_tuned_pp"),
        },
        "scope_public_and_static": {
            "source_repositories": source_corpus.get("n_repositories"),
            "codeql_true_positive": codeql.get("true_positive"),
            "codeql_false_positive": codeql.get("false_positive"),
            "codeql_missed_delayed_path": codeql.get("missed_delayed_path"),
            "semgrep_true_positive": semgrep.get("true_positive"),
            "semgrep_false_positive": semgrep.get("false_positive"),
            "semgrep_missed_delayed_path": semgrep.get("missed_delayed_path"),
            "public_reports_total": public_reports.get("total"),
            "public_reports_mapped": public_reports.get("mapped_to_target_categories"),
            "public_reports_excluded": public_reports.get("excluded"),
        },
        "hostile_mimicry_range": {
            "min_ccd_recall_1e_4": min(ccd_recalls),
            "max_ccd_recall_1e_4": max(ccd_recalls),
            "min_best_baseline_recall_1e_4": min(baseline_recalls),
            "max_best_baseline_recall_1e_4": max(baseline_recalls),
            "n_hostile_mimicry_suites": len(table12),
        },
        "production_throughput_tail": {
            "full_path_p95_ms": full_path_latency.get("p95_ms"),
            "full_path_p99_ms": full_path_latency.get("p99_ms"),
            "full_path_p999_ms": full_path_latency.get("p999_ms"),
            "single_host_throughput_per_s": full_path_latency.get("single_host_throughput_per_s"),
            "scoring_kernel_p50_ms": scoring_kernel_latency.get("p50_ms"),
            "scoring_kernel_p99_ms": scoring_kernel_latency.get("p99_ms"),
        },
    }


def compare_value(expected: object, observed: object, *, path: str) -> None:
    if isinstance(expected, bool):
        if observed is not expected:
            raise ValueError(f"{path} expected {expected}, observed {observed}")
        return
    if isinstance(expected, int) and not isinstance(expected, bool):
        if observed != expected:
            raise ValueError(f"{path} expected {expected}, observed {observed}")
        return
    if isinstance(expected, float):
        observed_number = require_number(observed, path=path)
        if not math.isclose(observed_number, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{path} expected {expected}, observed {observed}")
        return
    if observed != expected:
        raise ValueError(f"{path} expected {expected!r}, observed {observed!r}")


def validate_claim(
    raw_claim: object,
    *,
    index: int,
    root: Path,
    observed_by_id: Mapping[str, Mapping[str, Any]],
    external_blob: str,
) -> dict[str, Any]:
    claim = require_mapping(raw_claim, path=f"headline_claims[{index}]")
    item_id = require_string(claim.get("item_id"), path=f"headline_claims[{index}].item_id")
    source_paths = require_string_list(claim.get("source_paths"), path=f"{item_id}.source_paths")
    missing_paths = [rel for rel in source_paths if not (root / rel).exists()]
    if missing_paths:
        raise ValueError(f"{item_id} references missing source paths: {missing_paths}")
    expected = require_mapping(claim.get("expected"), path=f"{item_id}.expected")
    observed = observed_by_id.get(item_id)
    if observed is None:
        raise ValueError(f"{item_id} has no recomputed observed values")
    for key, expected_value in expected.items():
        if key not in observed:
            raise ValueError(f"{item_id}.observed missing expected key: {key}")
        compare_value(expected_value, observed[key], path=f"{item_id}.{key}")
    external_required = require_bool(claim.get("external_required"), path=f"{item_id}.external_required")
    external_keyword = str(claim.get("external_completion_keyword", "")).strip()
    if external_required:
        if not external_keyword:
            raise ValueError(f"{item_id} requires external completion but has no keyword")
        if external_keyword.lower() not in external_blob:
            raise ValueError(f"{item_id} external keyword not found in manifest completion items: {external_keyword!r}")
    elif external_keyword:
        raise ValueError(f"{item_id} has an external keyword but external_required is false")
    return {
        "item_id": item_id,
        "paper_refs": require_string_list(claim.get("paper_refs"), path=f"{item_id}.paper_refs"),
        "claim_summary": require_string(claim.get("claim_summary"), path=f"{item_id}.claim_summary"),
        "source_paths": source_paths,
        "expected": dict(expected),
        "observed": dict(observed),
        "external_required": external_required,
        "external_completion_keyword": external_keyword,
        "privacy_boundary": require_string(claim.get("privacy_boundary"), path=f"{item_id}.privacy_boundary"),
    }


def validate_headlines(data: Mapping[str, Any], *, root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    external_blob = "\n".join(str(item) for item in manifest.get("external_completion_items", [])).lower()
    observed = build_observed(root)
    raw_claims = data.get("headline_claims")
    if not isinstance(raw_claims, list):
        raise ValueError("headline_claims must be a list")
    claims = [
        validate_claim(
            raw_claim,
            index=index,
            root=root,
            observed_by_id=observed,
            external_blob=external_blob,
        )
        for index, raw_claim in enumerate(raw_claims)
    ]
    ids = [claim["item_id"] for claim in claims]
    duplicates = sorted(item_id for item_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate headline claim ids: {duplicates}")
    if tuple(ids) != EXPECTED_IDS:
        missing = [item_id for item_id in EXPECTED_IDS if item_id not in ids]
        extra = [item_id for item_id in ids if item_id not in EXPECTED_IDS]
        raise ValueError(f"headline claim order/content mismatch; missing={missing}, extra={extra}")
    return {
        "headline_claims": claims,
        "derived": {
            "n_headline_claims": len(claims),
            "external_required_count": sum(1 for claim in claims if claim["external_required"]),
            "source_paths": sorted({path for claim in claims for path in claim["source_paths"]}),
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    counts_path = root / args.counts
    data = load_json(counts_path)
    validated = validate_headlines(data, root=root, manifest_path=root / args.manifest)
    return {
        "status": "pass",
        "counts_file": str(counts_path),
        "source": data.get("source", "paper_headline_claim_inventory"),
        "paper": data.get("paper", "IEEE_S_P_Hostnames.pdf"),
        "scope": data.get("scope", ""),
        "manifest": str(root / args.manifest),
        "derived": validated["derived"],
        "headline_claims": validated["headline_claims"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute release-safe paper headline claim anchors.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--counts",
        default="paper_headline_claims/paper_headline_claims.json",
        help="Paper headline claim inventory.",
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
