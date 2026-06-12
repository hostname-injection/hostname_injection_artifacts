#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


POSITIVE_LABEL = "verified_executable_semantics"
NEGATIVE_LABEL = "resolved_benign"
UNRESOLVED_LABEL = "unresolved"
DEFAULT_SCORE_KEYS = (
    "ccd_score",
    "ccd_score_public",
    "ccd_public_score",
    "score",
)
DEFAULT_CALIBRATION_GROUP_KEYS = (
    "public_calibration_group",
    "calibration_group",
    "public_calibration_bucket",
    "calibration_bucket",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def compute_metrics(
    path: Path,
    *,
    alpha: float = 1e-4,
    score_keys: Iterable[str] | None = None,
    calibration_group_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    score_key_list = tuple(score_keys or DEFAULT_SCORE_KEYS)
    group_key_list = tuple(calibration_group_keys or DEFAULT_CALIBRATION_GROUP_KEYS)
    calibration_scores, score_key_hits, grouped_calibration_scores = collect_calibration_scores(path, score_key_list, group_key_list)
    threshold, rank = calibrate_threshold(calibration_scores, alpha)
    grouped_thresholds: dict[str, dict[str, Any]] = {}
    for group, scores in sorted(grouped_calibration_scores.items()):
        group_threshold, group_rank = calibrate_threshold(scores, alpha)
        if group_threshold is not None:
            grouped_thresholds[group] = {
                "threshold": group_threshold,
                "order_statistic_rank": group_rank,
                "order_statistic_sample_count": len(scores),
            }
    use_grouped_thresholds = bool(grouped_thresholds)

    by_split: Counter[Any] = Counter()
    by_source_family: Counter[Any] = Counter()
    by_label: Counter[Any] = Counter()
    by_evidence_tier: Counter[Any] = Counter()
    by_sink_family: Counter[Any] = Counter()
    ccd_flags: Counter[str] = Counter()
    regex_flags: Counter[str] = Counter()
    ccd_score_bins: Counter[str] = Counter()
    calibration_groups: Counter[str] = Counter()

    n_rows = 0
    resolved_rows = 0
    unresolved_rows = 0
    positive_rows = 0
    negative_rows = 0
    other_resolved_rows = 0
    metric_resolved_rows = 0
    metric_positive_rows = 0
    metric_negative_rows = 0
    benign_calibration_rows = 0

    ccd_flag_metrics = new_confusion()
    ccd_score_metrics = new_confusion()
    regex_metrics = new_confusion()
    overlap = {
        "overall": new_overlap(),
        "resolved": new_overlap(),
        "positives": new_overlap(),
        "negatives": new_overlap(),
        "by_label": {},
    }
    effective_ccd_sources: Counter[str] = Counter()

    for row in iter_jsonl(path):
        n_rows += 1
        split = row.get("split")
        label = str(row.get("label"))
        outputs = detector_outputs(row)
        calibration_group = extract_calibration_group(row, outputs, group_key_list)

        by_split[split] += 1
        by_source_family[row.get("source_family")] += 1
        by_label[label] += 1
        by_evidence_tier[row.get("evidence_tier")] += 1
        by_sink_family[row.get("sink_family")] += 1

        if split == "calibration" and label == NEGATIVE_LABEL:
            benign_calibration_rows += 1

        is_unresolved = label == UNRESOLVED_LABEL
        is_positive = label == POSITIVE_LABEL
        is_negative = label == NEGATIVE_LABEL
        if is_unresolved:
            unresolved_rows += 1
        else:
            resolved_rows += 1
            if is_positive:
                positive_rows += 1
            elif is_negative:
                negative_rows += 1
            else:
                other_resolved_rows += 1
            if split != "calibration" and (is_positive or is_negative):
                metric_resolved_rows += 1
                if is_positive:
                    metric_positive_rows += 1
                if is_negative:
                    metric_negative_rows += 1

        ccd_flag = as_bool(outputs.get("ccd_flag"))
        regex_flag = as_bool(outputs.get("regex_waf_flag"))
        ccd_flags[str(ccd_flag)] += 1
        regex_flags[str(regex_flag)] += 1
        ccd_score_bins[str(outputs.get("ccd_score_bin"))] += 1
        calibration_groups[str(calibration_group)] += 1

        update_confusion(
            ccd_flag_metrics,
            ccd_flag,
            is_positive=is_positive,
            is_negative=is_negative,
            is_unresolved=is_unresolved,
            is_calibration=split == "calibration",
        )
        update_confusion(
            regex_metrics,
            regex_flag,
            is_positive=is_positive,
            is_negative=is_negative,
            is_unresolved=is_unresolved,
            is_calibration=split == "calibration",
        )

        score, _score_key = extract_score(row, outputs, score_key_list)
        row_threshold = threshold_for_group(calibration_group, threshold, grouped_thresholds)
        score_flag = score > row_threshold if score is not None and row_threshold is not None else None
        update_confusion(
            ccd_score_metrics,
            score_flag,
            is_positive=is_positive,
            is_negative=is_negative,
            is_unresolved=is_unresolved,
            is_calibration=split == "calibration",
        )

        effective_ccd_flag, source = effective_detector_flag(ccd_flag, score_flag)
        effective_ccd_sources[source] += 1
        update_overlap(overlap["overall"], effective_ccd_flag, regex_flag)
        if not is_unresolved:
            update_overlap(overlap["resolved"], effective_ccd_flag, regex_flag)
        if is_positive:
            update_overlap(overlap["positives"], effective_ccd_flag, regex_flag)
        if is_negative:
            update_overlap(overlap["negatives"], effective_ccd_flag, regex_flag)
        label_overlap = overlap["by_label"].setdefault(label, new_overlap())
        update_overlap(label_overlap, effective_ccd_flag, regex_flag)

    finalize_confusion(ccd_flag_metrics)
    finalize_confusion(ccd_score_metrics)
    finalize_confusion(regex_metrics)

    if use_grouped_thresholds:
        threshold_source = "recomputed_from_public_grouped_calibration_scores"
    else:
        threshold_source = "recomputed_from_public_calibration_scores" if threshold is not None else "not_recomputed_no_public_scores"
    fixed_fpr_status = "available" if threshold is not None or use_grouped_thresholds else "not_available"
    fixed_fpr_reason = None if fixed_fpr_status == "available" else "No numeric public CCD score field was present on benign calibration rows."

    return {
        "n_rows": n_rows,
        "by_split": dict(by_split),
        "by_source_family": dict(by_source_family),
        "by_label": dict(by_label),
        "by_evidence_tier": dict(by_evidence_tier),
        "by_sink_family": dict(by_sink_family),
        "label_accounting": {
            "positive_label": POSITIVE_LABEL,
            "negative_label": NEGATIVE_LABEL,
            "unresolved_label": UNRESOLVED_LABEL,
            "resolved_rows": resolved_rows,
            "unresolved_rows": unresolved_rows,
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "other_resolved_rows": other_resolved_rows,
            "metric_protocol": "exclude unresolved rows and calibration split rows from TPR/FPR denominators",
            "metric_resolved_rows": metric_resolved_rows,
            "metric_positive_rows": metric_positive_rows,
            "metric_negative_rows": metric_negative_rows,
            "unresolved_excluded_from_tpr_fpr": True,
        },
        "resolved_rows": resolved_rows,
        "unresolved_rows": unresolved_rows,
        "calibration": {
            "alpha": alpha,
            "benign_calibration_rows": benign_calibration_rows,
            "scored_benign_calibration_rows": len(calibration_scores),
            "score_fields_considered": list(score_key_list),
            "calibration_group_fields_considered": list(group_key_list),
            "score_field_hits_on_calibration_rows": dict(score_key_hits),
            "threshold_source": threshold_source,
            "threshold": threshold,
            "order_statistic_rank": rank,
            "order_statistic_sample_count": len(calibration_scores),
            "grouped_thresholds": grouped_thresholds,
            "n_calibration_groups": len(grouped_thresholds),
            "calibration_group_counts": dict(calibration_groups),
            "split_conformal_rule": "ceil((1-alpha)*(n+1))-th sorted benign calibration score, clamped to [1,n]",
        },
        "fixed_fpr_replay": {
            "status": fixed_fpr_status,
            "reason": fixed_fpr_reason,
            "requested_fpr": alpha,
            "threshold": None if use_grouped_thresholds else threshold,
            "grouped_thresholds": grouped_thresholds if use_grouped_thresholds else {},
            "threshold_source": threshold_source,
            **ccd_score_metrics,
        },
        "resolved_row_replay": {
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "metric_positive_rows": metric_positive_rows,
            "metric_negative_rows": metric_negative_rows,
            "ccd_scored_rows": ccd_score_metrics["rows_with_predictions"],
            "ccd_tpr": ccd_flag_metrics["tpr"],
            "ccd_fpr": ccd_flag_metrics["fpr"],
            "ccd_flag_metrics": ccd_flag_metrics,
            "regex_waf_tpr": regex_metrics["tpr"],
            "regex_waf_fpr": regex_metrics["fpr"],
            "regex_waf_metrics": regex_metrics,
        },
        "detector_overlap": {
            "ccd_flag_counts": dict(ccd_flags),
            "ccd_score_bin_counts": dict(ccd_score_bins),
            "regex_waf_flag_counts": dict(regex_flags),
            "effective_ccd_flag_sources": dict(effective_ccd_sources),
            **overlap,
        },
        "public_anchor_replay_checks": {
            "included": False,
            "reason": "No public anchor replay subset is included in this release artifact.",
        },
    }


def collect_calibration_scores(
    path: Path,
    score_keys: Iterable[str],
    calibration_group_keys: Iterable[str],
) -> tuple[list[float], Counter[str], dict[str, list[float]]]:
    scores: list[float] = []
    hits: Counter[str] = Counter()
    grouped_scores: dict[str, list[float]] = {}
    score_key_list = tuple(score_keys)
    group_key_list = tuple(calibration_group_keys)
    for row in iter_jsonl(path):
        if row.get("split") != "calibration" or row.get("label") != NEGATIVE_LABEL:
            continue
        outputs = detector_outputs(row)
        score, key = extract_score(row, outputs, score_key_list)
        if score is not None:
            scores.append(score)
            hits[str(key)] += 1
            group = extract_calibration_group(row, outputs, group_key_list)
            if group is not None:
                grouped_scores.setdefault(group, []).append(score)
    return scores, hits, grouped_scores


def detector_outputs(row: Mapping[str, Any]) -> Mapping[str, Any]:
    outputs = row.get("detector_outputs", {})
    return outputs if isinstance(outputs, Mapping) else {}


def extract_score(row: Mapping[str, Any], outputs: Mapping[str, Any], score_keys: Iterable[str]) -> tuple[float | None, str | None]:
    for key in score_keys:
        if key in outputs:
            score = as_float(outputs.get(key))
            if score is not None:
                return score, f"detector_outputs.{key}"
        if key in row:
            score = as_float(row.get(key))
            if score is not None:
                return score, key
    return None, None


def extract_calibration_group(row: Mapping[str, Any], outputs: Mapping[str, Any], group_keys: Iterable[str]) -> str | None:
    for key in group_keys:
        value = outputs.get(key)
        if value is None:
            value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def threshold_for_group(group: str | None, global_threshold: float | None, grouped_thresholds: Mapping[str, Mapping[str, Any]]) -> float | None:
    if group is not None and group in grouped_thresholds:
        value = grouped_thresholds[group].get("threshold")
        return as_float(value)
    return global_threshold


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def calibrate_threshold(benign_scores: list[float], alpha: float) -> tuple[float | None, int | None]:
    if not benign_scores:
        return None, None
    scores = sorted(benign_scores)
    n = len(scores)
    rank = math.ceil((1.0 - alpha) * (n + 1))
    rank = max(1, min(rank, n))
    return float(scores[rank - 1]), rank


def new_confusion() -> dict[str, Any]:
    return {
        "rows_with_predictions": 0,
        "metric_rows_with_predictions": 0,
        "metric_positive_rows_with_predictions": 0,
        "metric_negative_rows_with_predictions": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "tpr": None,
        "fpr": None,
    }


def update_confusion(
    metrics: dict[str, Any],
    flag: bool | None,
    *,
    is_positive: bool,
    is_negative: bool,
    is_unresolved: bool,
    is_calibration: bool,
) -> None:
    if flag is None:
        return
    metrics["rows_with_predictions"] += 1
    if is_unresolved or is_calibration:
        return
    if not is_positive and not is_negative:
        return
    metrics["metric_rows_with_predictions"] += 1
    if is_positive:
        metrics["metric_positive_rows_with_predictions"] += 1
        metrics["tp" if flag else "fn"] += 1
    elif is_negative:
        metrics["metric_negative_rows_with_predictions"] += 1
        metrics["fp" if flag else "tn"] += 1


def finalize_confusion(metrics: dict[str, Any]) -> None:
    metrics["tpr"] = rate(metrics["tp"], metrics["tp"] + metrics["fn"])
    metrics["fpr"] = rate(metrics["fp"], metrics["fp"] + metrics["tn"])


def new_overlap() -> dict[str, int]:
    return {
        "rows_with_both_predictions": 0,
        "ccd_true_regex_true": 0,
        "ccd_true_regex_false": 0,
        "ccd_false_regex_true": 0,
        "ccd_false_regex_false": 0,
    }


def update_overlap(metrics: dict[str, int], ccd_flag: bool | None, regex_flag: bool | None) -> None:
    if ccd_flag is None or regex_flag is None:
        return
    metrics["rows_with_both_predictions"] += 1
    if ccd_flag and regex_flag:
        metrics["ccd_true_regex_true"] += 1
    elif ccd_flag and not regex_flag:
        metrics["ccd_true_regex_false"] += 1
    elif not ccd_flag and regex_flag:
        metrics["ccd_false_regex_true"] += 1
    else:
        metrics["ccd_false_regex_false"] += 1


def effective_detector_flag(released_flag: bool | None, score_flag: bool | None) -> tuple[bool | None, str]:
    if released_flag is not None:
        return released_flag, "released_ccd_flag"
    if score_flag is not None:
        return score_flag, "public_score_threshold"
    return None, "not_available"


def rate(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return num / den


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute public HIB release row counts, fixed-FPR metrics, and detector overlap.")
    parser.add_argument("--public-release", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=1e-4, help="Fixed-FPR target used for split-conformal threshold replay.")
    parser.add_argument(
        "--score-key",
        action="append",
        default=None,
        help="Candidate public CCD score key to read from detector_outputs or the row. May be repeated.",
    )
    parser.add_argument(
        "--calibration-group-key",
        action="append",
        default=None,
        help="Candidate public calibration group key to read from detector_outputs or the row. May be repeated.",
    )
    args = parser.parse_args()

    metrics = compute_metrics(
        args.public_release,
        alpha=args.alpha,
        score_keys=args.score_key,
        calibration_group_keys=args.calibration_group_key,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
