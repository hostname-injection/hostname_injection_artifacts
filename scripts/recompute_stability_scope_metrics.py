#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_number(value: object, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def require_int(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    if value < 0:
        raise ValueError(f"{path} must be non-negative")
    return value


def require_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be true or false")
    return value


def require_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def require_probability(value: object, *, path: str) -> float:
    observed = require_number(value, path=path)
    if not 0.0 <= observed <= 1.0:
        raise ValueError(f"{path} must be in [0,1]")
    return observed


def validate_certificate_scope(data: Mapping[str, Any]) -> dict[str, Any]:
    alpha = require_probability(data.get("alpha"), path="decision_stability.alpha")
    edit_budget = require_int(data.get("edit_budget_k"), path="decision_stability.edit_budget_k")
    combined = require_probability(data.get("combined_stable_detected_positive_coverage"), path="decision_stability.combined")
    sec_alone = require_probability(data.get("sec_alone_stable_detected_positive_coverage"), path="decision_stability.sec_alone")
    native_min = require_probability(data.get("native_feature_baseline_min_coverage"), path="decision_stability.native_min")
    native_max = require_probability(data.get("native_feature_baseline_max_coverage"), path="decision_stability.native_max")
    if native_min > native_max:
        raise ValueError("native-feature baseline minimum exceeds maximum")
    if combined <= sec_alone:
        raise ValueError("combined certificate coverage must exceed SEC-alone coverage")
    if combined <= native_max:
        raise ValueError("combined certificate coverage must exceed native-feature baselines")

    scope = data.get("scope_bindings")
    if not isinstance(scope, list) or not scope:
        raise ValueError("decision_stability.scope_bindings must be a non-empty list")
    required_scope = {"normalizer", "cone_sketch", "score_path", "threshold", "edit_manifest_version"}
    if not required_scope.issubset(set(str(item) for item in scope)):
        raise ValueError(f"decision_stability.scope_bindings missing {sorted(required_scope - set(scope))}")
    for key in ("deterministic_edit_ball_closure", "randomized_sampling_only_orders_traversal"):
        if require_bool(data.get(key), path=f"decision_stability.{key}") is not True:
            raise ValueError(f"decision_stability.{key} must be true")
    if require_bool(data.get("downstream_sink_safety_claim"), path="decision_stability.downstream_sink_safety_claim") is not False:
        raise ValueError("decision_stability.downstream_sink_safety_claim must be false")

    return {
        "alpha": alpha,
        "edit_budget_k": edit_budget,
        "combined_minus_sec_alone_points": 100.0 * (combined - sec_alone),
        "combined_minus_native_max_points": 100.0 * (combined - native_max),
        "combined_stable_detected_positive_coverage_percent": 100.0 * combined,
        "sec_alone_stable_detected_positive_coverage_percent": 100.0 * sec_alone,
        "native_feature_baseline_coverage_range_percent": [100.0 * native_min, 100.0 * native_max],
    }


def validate_holdout_depth(data: Mapping[str, Any]) -> dict[str, Any]:
    fpr = require_probability(data.get("fpr"), path="holdout_depth.fpr")
    gap = require_mapping(data.get("ccd_gain_over_strongest_non_llm_baseline"), path="holdout_depth.gap")
    min_gap = require_number(gap.get("min_points"), path="holdout_depth.gap.min_points")
    max_gap = require_number(gap.get("max_points"), path="holdout_depth.gap.max_points")
    if min_gap <= 0.0 or max_gap < min_gap:
        raise ValueError("holdout-depth CCD gain range is invalid")
    zero_shot = require_mapping(data.get("zero_shot_composition_recall"), path="holdout_depth.zero_shot")
    d3 = require_mapping(zero_shot.get("depth3"), path="holdout_depth.zero_shot.depth3")
    d5 = require_mapping(zero_shot.get("depth5"), path="holdout_depth.zero_shot.depth5")
    d3_min = require_probability(d3.get("min"), path="holdout_depth.zero_shot.depth3.min")
    d3_max = require_probability(d3.get("max"), path="holdout_depth.zero_shot.depth3.max")
    d5_min = require_probability(d5.get("min"), path="holdout_depth.zero_shot.depth5.min")
    d5_max = require_probability(d5.get("max"), path="holdout_depth.zero_shot.depth5.max")
    if d3_min > d3_max or d5_min > d5_max:
        raise ValueError("zero-shot recall range is invalid")
    seen = require_mapping(data.get("seen_deeper_composition_recall"), path="holdout_depth.seen_deeper")
    macro_depth2 = require_probability(seen.get("macro_depth_le_2"), path="holdout_depth.seen_deeper.macro_depth_le_2")
    macro_depth5 = require_probability(seen.get("macro_depth5"), path="holdout_depth.seen_deeper.macro_depth5")
    min_depth5 = require_probability(seen.get("min_mechanism_depth5"), path="holdout_depth.seen_deeper.min_mechanism_depth5")
    if macro_depth5 > macro_depth2:
        raise ValueError("depth-5 macro recall should not exceed depth<=2 macro recall in this stress test")
    if min_depth5 > macro_depth5:
        raise ValueError("minimum depth-5 mechanism recall should not exceed macro depth-5 recall")

    return {
        "fpr": fpr,
        "ccd_gain_over_strongest_non_llm_baseline_points": [min_gap, max_gap],
        "zero_shot_depth3_recall_range": [d3_min, d3_max],
        "zero_shot_depth5_recall_range": [d5_min, d5_max],
        "zero_shot_depth5_range_drop_from_depth3": [d3_min - d5_min, d3_max - d5_max],
        "seen_macro_recall_drop_depth2_to_depth5_points": 100.0 * (macro_depth2 - macro_depth5),
        "seen_min_mechanism_depth5_recall": min_depth5,
    }


def validate_drift_refresh(data: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("refreshes_benign_reference_only", "positive_reference_fixed", "encoder_parameters_fixed"):
        if require_bool(data.get(key), path=f"drift_refresh.{key}") is not True:
            raise ValueError(f"drift_refresh.{key} must be true")
    window_days = require_int(data.get("refresh_window_days"), path="drift_refresh.refresh_window_days")
    runtime_seconds = require_number(data.get("refresh_runtime_seconds"), path="drift_refresh.refresh_runtime_seconds")
    replay_delta = require_probability(data.get("independent_replay_max_count_metric_delta_rate"), path="drift_refresh.replay_delta")
    holdouts = require_int(data.get("blind_tenant_holdouts"), path="drift_refresh.blind_tenant_holdouts")
    fpr_target = require_probability(data.get("fpr_target"), path="drift_refresh.fpr_target")
    tolerance = require_probability(data.get("max_fpr_deviation"), path="drift_refresh.max_fpr_deviation")
    if runtime_seconds <= 0.0:
        raise ValueError("drift refresh runtime must be positive")
    return {
        "refresh_window_days": window_days,
        "refresh_runtime_seconds": runtime_seconds,
        "independent_replay_max_count_metric_delta_percent": 100.0 * replay_delta,
        "blind_tenant_holdouts": holdouts,
        "fpr_target": fpr_target,
        "max_fpr_deviation": tolerance,
        "max_fpr_deviation_as_target_fraction": tolerance / fpr_target if fpr_target else None,
    }


def validate_public_real(data: Mapping[str, Any]) -> dict[str, Any]:
    fpr = require_probability(data.get("fpr"), path="public_real.fpr")
    synthetic = require_probability(data.get("synthetic_only_recall"), path="public_real.synthetic_only_recall")
    public_real = require_probability(data.get("public_real_training_recall"), path="public_real.public_real_training_recall")
    mixed = require_probability(data.get("synthetic_plus_5_percent_public_real_recall"), path="public_real.mixed_recall")
    if not synthetic < public_real < mixed:
        raise ValueError("public-real recall ordering should be synthetic < public-real < mixed")
    return {
        "fpr": fpr,
        "synthetic_only_recall": synthetic,
        "public_real_training_recall": public_real,
        "synthetic_plus_5_percent_public_real_recall": mixed,
        "public_real_gain_over_synthetic_points": 100.0 * (public_real - synthetic),
        "mixed_gain_over_public_real_points": 100.0 * (mixed - public_real),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    counts_path = Path(args.counts)
    data = load_json(counts_path)
    decision_stability = require_mapping(data.get("decision_stability"), path="decision_stability")
    holdout_depth = require_mapping(data.get("holdout_depth"), path="holdout_depth")
    drift_refresh = require_mapping(data.get("drift_refresh"), path="drift_refresh")
    public_real = require_mapping(data.get("public_real_scope"), path="public_real_scope")
    return {
        "status": "pass",
        "counts_file": str(counts_path),
        "source": data.get("source", "published_aggregate_stability_scope"),
        "paper_sections": data.get("paper_sections", []),
        "decision_stability": decision_stability,
        "holdout_depth": holdout_depth,
        "drift_refresh": drift_refresh,
        "public_real_scope": public_real,
        "derived": {
            "decision_stability": validate_certificate_scope(decision_stability),
            "holdout_depth": validate_holdout_depth(holdout_depth),
            "drift_refresh": validate_drift_refresh(drift_refresh),
            "public_real_scope": validate_public_real(public_real),
        },
        "private_by_design": data.get("private_by_design", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute release-safe aggregate stability and scope accounting.")
    parser.add_argument(
        "--counts",
        default="stability_scope/paper_stability_scope_counts.json",
        help="Release-safe aggregate stability/scope counts.",
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
