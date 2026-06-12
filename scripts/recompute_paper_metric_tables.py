#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_number(value: object, *, path: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def validate_table5(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("table5_baseline_audit must contain at least one row")
    validated: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        method = str(row.get("method", "")).strip()
        if not method:
            raise ValueError(f"table5 row {idx} missing method")
        tpr = require_number(row.get("tpr_at_1e_4"), path=f"table5.{method}.tpr_at_1e_4")
        ci = row.get("ci95")
        if not isinstance(ci, list) or len(ci) != 2:
            raise ValueError(f"table5.{method}.ci95 must be [low, high]")
        ci_low = require_number(ci[0], path=f"table5.{method}.ci95[0]")
        ci_high = require_number(ci[1], path=f"table5.{method}.ci95[1]")
        fpr = require_number(row.get("fpr"), path=f"table5.{method}.fpr")
        p99 = require_number(row.get("p99_ms"), path=f"table5.{method}.p99_ms")
        p999 = require_number(row.get("p999_ms"), path=f"table5.{method}.p999_ms")
        if not (0.0 <= ci_low <= tpr <= ci_high <= 1.0):
            raise ValueError(f"table5.{method}: CI must contain TPR and be within [0,1]")
        if not (0.0 <= fpr <= 1.0):
            raise ValueError(f"table5.{method}.fpr must be in [0,1]")
        if p999 < p99:
            raise ValueError(f"table5.{method}.p999_ms must be >= p99_ms")
        validated.append(
            {
                **row,
                "method": method,
                "tpr_at_1e_4": tpr,
                "ci95": [ci_low, ci_high],
                "fpr": fpr,
                "p99_ms": p99,
                "p999_ms": p999,
            }
        )
    return validated


def validate_table6(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = ("hib_1e_4_drop", "hib_1e_5_drop", "holdout_drop", "whitebox_drop")
    validated: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        variant = str(row.get("variant_removed", "")).strip()
        if not variant:
            raise ValueError(f"table6 row {idx} missing variant_removed")
        clean = {"variant_removed": variant}
        for key in required:
            clean[key] = require_number(row.get(key), path=f"table6.{variant}.{key}")
        validated.append({**row, **clean})
    if not any(row["variant_removed"].lower().startswith("none") for row in validated):
        raise ValueError("table6 must include full CCD reference row")
    return validated


def validate_table12(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = ("ccd_recall_1e_4", "ccd_recall_1e_5", "best_baseline_recall_1e_4")
    validated: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        suite = str(row.get("suite", "")).strip()
        if not suite:
            raise ValueError(f"table12 row {idx} missing suite")
        clean = {"suite": suite}
        for key in required:
            value = require_number(row.get(key), path=f"table12.{suite}.{key}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"table12.{suite}.{key} must be in [0,1]")
            clean[key] = value
        validated.append({**row, **clean})
    return validated


def validate_generator(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        source = str(row.get("training_source", "")).strip()
        if not source:
            raise ValueError(f"generator row {idx} missing training_source")
        recall = require_number(row.get("recall_at_1e_4"), path=f"generator.{source}.recall_at_1e_4")
        if not 0.0 <= recall <= 1.0:
            raise ValueError(f"generator.{source}.recall_at_1e_4 must be in [0,1]")
        validated.append({**row, "training_source": source, "recall_at_1e_4": recall})
    return validated


def validate_table10(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    expected = [
        ("70B instruct", "meta-llama/Llama-3.3-70B-Instruct"),
        ("32B instruct", "Qwen/Qwen2.5-32B-Instruct"),
        ("14B instruct", "Qwen/Qwen2.5-14B-Instruct"),
        ("8B instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    ]
    if len(rows) != len(expected):
        raise ValueError("table10_llm_baselines must contain exactly four checkpoint identifiers")
    validated: list[dict[str, str]] = []
    for idx, (row, (expected_family, expected_checkpoint)) in enumerate(zip(rows, expected, strict=True), start=1):
        family = str(row.get("family", "")).strip()
        checkpoint = str(row.get("checkpoint_identifier", "")).strip()
        if (family, checkpoint) != (expected_family, expected_checkpoint):
            raise ValueError(
                f"table10 row {idx} must be {(expected_family, expected_checkpoint)}, observed {(family, checkpoint)}"
            )
        validated.append({"family": family, "checkpoint_identifier": checkpoint})
    return validated


def summarize_table5(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(rows, key=lambda row: row["tpr_at_1e_4"])
    non_ccd = [row for row in rows if row["method"].lower() != "ccd"]
    best_non_ccd = max(non_ccd, key=lambda row: row["tpr_at_1e_4"]) if non_ccd else None
    return {
        "n_methods": len(rows),
        "best_tpr_method": best["method"],
        "best_tpr": best["tpr_at_1e_4"],
        "best_non_ccd_method": None if best_non_ccd is None else best_non_ccd["method"],
        "ccd_tpr_lead_over_best_non_ccd": None
        if best_non_ccd is None
        else next(row["tpr_at_1e_4"] for row in rows if row["method"].lower() == "ccd")
        - best_non_ccd["tpr_at_1e_4"],
        "methods_at_or_below_1e_4_fpr_with_tolerance": [
            row["method"] for row in rows if row["fpr"] <= 1.05e-4
        ],
        "fastest_p99_method": min(rows, key=lambda row: row["p99_ms"])["method"],
    }


def summarize_table6(rows: list[dict[str, Any]]) -> dict[str, Any]:
    non_reference = [row for row in rows if not row["variant_removed"].lower().startswith("none")]
    largest_whitebox = min(non_reference, key=lambda row: row["whitebox_drop"])
    largest_hib_1e5 = min(non_reference, key=lambda row: row["hib_1e_5_drop"])
    return {
        "n_variants": len(rows),
        "largest_whitebox_drop_variant": largest_whitebox["variant_removed"],
        "largest_whitebox_drop_points": abs(largest_whitebox["whitebox_drop"]),
        "largest_hib_1e_5_drop_variant": largest_hib_1e5["variant_removed"],
        "largest_hib_1e_5_drop_points": abs(largest_hib_1e5["hib_1e_5_drop"]),
    }


def summarize_table12(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weakest_1e5 = min(rows, key=lambda row: row["ccd_recall_1e_5"])
    margins = [
        {
            "suite": row["suite"],
            "ccd_minus_best_baseline_at_1e_4": row["ccd_recall_1e_4"] - row["best_baseline_recall_1e_4"],
        }
        for row in rows
    ]
    min_margin = min(margins, key=lambda row: row["ccd_minus_best_baseline_at_1e_4"])
    return {
        "n_suites": len(rows),
        "weakest_ccd_1e_5_suite": weakest_1e5["suite"],
        "weakest_ccd_1e_5_recall": weakest_1e5["ccd_recall_1e_5"],
        "minimum_ccd_margin_over_best_baseline_1e_4_suite": min_margin["suite"],
        "minimum_ccd_margin_over_best_baseline_1e_4": min_margin["ccd_minus_best_baseline_at_1e_4"],
    }


def summarize_generator(rows: list[dict[str, Any]], appendix_f: dict[str, Any]) -> dict[str, Any]:
    by_source = {row["training_source"].lower(): row["recall_at_1e_4"] for row in rows}
    llm = by_source.get("llm priors")
    real = by_source.get("real-payload training")
    pcfg = by_source.get("pcfg-only priors")
    return {
        "n_sources": len(rows),
        "pcfg_to_llm_gain": None if pcfg is None or llm is None else llm - pcfg,
        "real_training_gain_over_llm": None if real is None or llm is None else real - llm,
        "fid": appendix_f.get("fid"),
        "mmd": appendix_f.get("mmd"),
        "public_text_overlap_rate": appendix_f.get("public_text_overlap_rate"),
    }


def summarize_table10(rows: list[dict[str, str]]) -> dict[str, Any]:
    providers = sorted({row["checkpoint_identifier"].split("/", 1)[0] for row in rows})
    return {
        "n_llm_checkpoints": len(rows),
        "families": [row["family"] for row in rows],
        "checkpoint_providers": providers,
        "largest_family": rows[0]["family"] if rows else None,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    counts_path = Path(args.counts)
    data = load_json(counts_path)
    table5 = validate_table5(data.get("table5_baseline_audit", []))
    table6 = validate_table6(data.get("table6_ablation_drops", []))
    table12 = validate_table12(data.get("table12_hostile_mimicry", []))
    generator = validate_generator(data.get("generator_comparison", []))
    table10 = validate_table10(data.get("table10_llm_baselines", []))
    appendix_f = data.get("appendix_f_synthetic_validity", {})
    if not isinstance(appendix_f, dict):
        raise ValueError("appendix_f_synthetic_validity must be an object")

    report = {
        "status": "pass",
        "counts_file": str(counts_path),
        "paper_sections": data.get("paper_sections", []),
        "source": "published_aggregate_metric_tables",
        "table5_baseline_audit": table5,
        "table6_ablation_drops": table6,
        "table10_llm_baselines": table10,
        "table12_hostile_mimicry": table12,
        "generator_comparison": generator,
        "appendix_f_synthetic_validity": appendix_f,
        "derived": {
            "table5": summarize_table5(table5),
            "table6": summarize_table6(table6),
            "table10": summarize_table10(table10),
            "table12": summarize_table12(table12),
            "generator_comparison": summarize_generator(generator, appendix_f),
        },
        "private_by_design": data.get("private_by_design", []),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute release-safe aggregate accounting for paper metric tables."
    )
    parser.add_argument(
        "--counts",
        default="paper_metric_tables/paper_metric_tables.json",
        help="Release-safe aggregate metric-table summary.",
    )
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
