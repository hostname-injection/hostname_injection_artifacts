#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


SEGMENTS = ("overlap", "ccd_only", "regex_waf_only")
COUNT_KEYS = ("verified_positive", "verified_or_uncertain", "all_alerts")
LABEL_ALIASES = {
    "verified": "verified_positive",
    "verified-positive": "verified_positive",
    "verified_positive": "verified_positive",
    "positive": "verified_positive",
    "uncertain": "uncertain",
    "unresolved": "uncertain",
    "verified-benign": "verified_benign",
    "verified_benign": "verified_benign",
    "benign": "verified_benign",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_token(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def normalize_label(value: object) -> str:
    label = normalize_token(value).replace("_", "-")
    normalized = LABEL_ALIASES.get(label)
    if normalized is None:
        normalized = LABEL_ALIASES.get(normalize_token(value))
    if normalized is None:
        raise ValueError(f"unsupported live-overlap label: {value!r}")
    return normalized


def empty_segment_counts() -> dict[str, int]:
    return {key: 0 for key in COUNT_KEYS}


def validate_table(table: dict[str, Any], *, source: str) -> dict[str, dict[str, int]]:
    validated: dict[str, dict[str, int]] = {}
    for segment in SEGMENTS:
        raw_counts = table.get(segment)
        if not isinstance(raw_counts, dict):
            raise ValueError(f"{source}: missing table7.{segment}")
        counts = empty_segment_counts()
        for key in COUNT_KEYS:
            value = raw_counts.get(key)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{source}: table7.{segment}.{key} must be a non-negative integer")
            counts[key] = value
        if counts["verified_positive"] > counts["verified_or_uncertain"]:
            raise ValueError(f"{source}: table7.{segment}.verified_positive exceeds verified_or_uncertain")
        if counts["verified_or_uncertain"] > counts["all_alerts"]:
            raise ValueError(f"{source}: table7.{segment}.verified_or_uncertain exceeds all_alerts")
        validated[segment] = counts
    return validated


def load_count_summary(path: Path) -> dict[str, Any]:
    data = load_json(path)
    table = data.get("table7")
    if not isinstance(table, dict):
        raise ValueError(f"{path} must contain a table7 object")
    data["table7"] = validate_table(table, source=str(path))
    window = data.get("window", {})
    days = window.get("days")
    if days is not None and (not isinstance(days, int) or days <= 0):
        raise ValueError(f"{path}: window.days must be a positive integer when present")
    return data


def segment_for_row(row: dict[str, Any], *, path: Path, line_no: int) -> str:
    ccd = row.get("ccd_flag")
    regex_waf = row.get("regex_waf_flag")
    if not isinstance(ccd, bool) or not isinstance(regex_waf, bool):
        raise ValueError(f"{path}:{line_no}: ccd_flag and regex_waf_flag must be booleans")
    if ccd and regex_waf:
        return "overlap"
    if ccd:
        return "ccd_only"
    if regex_waf:
        return "regex_waf_only"
    raise ValueError(f"{path}:{line_no}: live-overlap rows must have at least one detector flag")


def aggregate_labeled_items(path: Path) -> dict[str, dict[str, int]]:
    table = {segment: empty_segment_counts() for segment in SEGMENTS}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL record: {exc}") from exc
        segment = segment_for_row(row, path=path, line_no=line_no)
        label = normalize_label(row.get("label"))
        table[segment]["all_alerts"] += 1
        if label == "verified_positive":
            table[segment]["verified_positive"] += 1
            table[segment]["verified_or_uncertain"] += 1
        elif label == "uncertain":
            table[segment]["verified_or_uncertain"] += 1
    if sum(counts["all_alerts"] for counts in table.values()) == 0:
        raise ValueError(f"{path} did not contain any labeled live-overlap rows")
    return table


def compare_table(actual: dict[str, dict[str, int]], expected: dict[str, dict[str, int]]) -> list[str]:
    failures: list[str] = []
    for segment in SEGMENTS:
        for key in COUNT_KEYS:
            observed = actual[segment][key]
            wanted = expected[segment][key]
            if observed != wanted:
                failures.append(f"table7.{segment}.{key}: observed {observed}, expected {wanted}")
    return failures


def enrich_segment_counts(table: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    enriched: dict[str, dict[str, int]] = {}
    for segment, counts in table.items():
        uncertain = counts["verified_or_uncertain"] - counts["verified_positive"]
        verified_benign = counts["all_alerts"] - counts["verified_or_uncertain"]
        enriched[segment] = {
            **counts,
            "uncertain": uncertain,
            "verified_benign": verified_benign,
        }
    return enriched


def combine_segments(table: dict[str, dict[str, int]], segments: tuple[str, ...]) -> dict[str, int]:
    out = {key: 0 for key in (*COUNT_KEYS, "uncertain", "verified_benign")}
    enriched = enrich_segment_counts(table)
    for segment in segments:
        for key in out:
            out[key] += enriched[segment][key]
    return out


def summarize_detector_bearing(table: dict[str, dict[str, int]], days: int | None) -> dict[str, dict[str, Any]]:
    detectors = {
        "ccd_bearing": combine_segments(table, ("overlap", "ccd_only")),
        "regex_waf_bearing": combine_segments(table, ("overlap", "regex_waf_only")),
    }
    summary: dict[str, dict[str, Any]] = {}
    for name, counts in detectors.items():
        all_alerts = counts["all_alerts"]
        nonverified = counts["uncertain"] + counts["verified_benign"]
        row: dict[str, Any] = {
            **counts,
            "nonverified": nonverified,
            "lower_bound_reviewed_item_ppv": None if all_alerts == 0 else counts["verified_positive"] / all_alerts,
        }
        if days:
            row["verified_benign_per_day"] = counts["verified_benign"] / days
            row["nonverified_per_day"] = nonverified / days
        summary[name] = row
    return summary


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    counts_path = Path(args.counts)
    counts_data = load_count_summary(counts_path)
    expected_table = counts_data["table7"]
    if args.labels:
        source = "labeled_live_items"
        table = aggregate_labeled_items(Path(args.labels))
    else:
        source = "published_aggregate_counts"
        table = deepcopy(expected_table)

    failures: list[str] = []
    if args.expect_counts:
        failures.extend(compare_table(table, expected_table))

    window = counts_data.get("window", {})
    days = window.get("days") if isinstance(window, dict) else None
    derived = {
        "ccd_verified_live_positives": table["overlap"]["verified_positive"] + table["ccd_only"]["verified_positive"],
        "ccd_only_verified_live_positives": table["ccd_only"]["verified_positive"],
        "regex_waf_only_verified_live_positives": table["regex_waf_only"]["verified_positive"],
        "baseline_only_all_alerts": table["regex_waf_only"]["all_alerts"],
        "baseline_only_uncertain": table["regex_waf_only"]["verified_or_uncertain"]
        - table["regex_waf_only"]["verified_positive"],
        "baseline_only_verified_benign": table["regex_waf_only"]["all_alerts"]
        - table["regex_waf_only"]["verified_or_uncertain"],
    }
    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "source": source,
        "counts_file": str(counts_path),
        "paper_section": counts_data.get("paper_section"),
        "window": window,
        "table7": enrich_segment_counts(table),
        "detector_bearing": summarize_detector_bearing(table, days),
        "derived": {
            **derived,
            **counts_data.get("reported_derived", {}),
        },
        "case_study_boundary": counts_data.get("case_study_boundary", {}),
        "private_by_design": counts_data.get("private_by_design", []),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute release-safe live-overlap accounting for the paper.")
    parser.add_argument(
        "--counts",
        default="live_overlap/paper_live_overlap_counts.json",
        help="Release-safe paper count summary or expected-count file.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional JSONL file with ccd_flag, regex_waf_flag, and label fields.",
    )
    parser.add_argument(
        "--expect-counts",
        action="store_true",
        help="Fail unless --labels aggregates exactly match --counts.",
    )
    parser.add_argument("--out", default=None, help="Optional path for the JSON report.")
    args = parser.parse_args()

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
