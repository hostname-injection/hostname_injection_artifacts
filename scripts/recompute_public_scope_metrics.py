#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


TARGET_STATUSES = {"target", "mapped", "in_scope", "covered"}
EXCLUDED_STATUSES = {"excluded", "out_of_scope", "boundary"}
ALLOWED_EXCLUSIONS = {"rebinding", "certificate-validation", "quic"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_token(value: object) -> str:
    return str(value).strip().lower().replace("_", "-")


def load_count_summary(path: Path) -> dict[str, Any]:
    data = load_json(path)
    summary = data.get("public_reports")
    if not isinstance(summary, dict):
        raise ValueError(f"{path} must contain a public_reports object")
    for key in ("total", "mapped_to_target_categories", "excluded"):
        value = summary.get(key)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{path}: public_reports.{key} must be a non-negative integer")
    if summary["mapped_to_target_categories"] + summary["excluded"] != summary["total"]:
        raise ValueError(f"{path}: mapped + excluded must equal total")

    exclusions = summary.get("exclusion_counts", {})
    if not isinstance(exclusions, dict):
        raise ValueError(f"{path}: public_reports.exclusion_counts must be an object")
    for reason, value in exclusions.items():
        normalized_reason = normalize_token(reason)
        if normalized_reason not in ALLOWED_EXCLUSIONS:
            raise ValueError(f"{path}: unsupported exclusion reason {reason!r}")
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{path}: invalid exclusion count for {reason!r}: {value!r}")
    if sum(int(value) for value in exclusions.values()) != summary["excluded"]:
        raise ValueError(f"{path}: exclusion_counts must sum to public_reports.excluded")
    return data


def summarize_public_reports_from_jsonl(path: Path) -> dict[str, Any]:
    total = 0
    mapped = 0
    exclusions: Counter[str] = Counter()
    target_categories: Counter[str] = Counter()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL record: {exc}") from exc
        total += 1
        status = normalize_token(row.get("status"))
        if status in TARGET_STATUSES:
            mapped += 1
            category = normalize_token(row.get("target_category", "unspecified"))
            target_categories[category or "unspecified"] += 1
        elif status in EXCLUDED_STATUSES:
            reason = normalize_token(row.get("exclusion_reason"))
            if reason not in ALLOWED_EXCLUSIONS:
                raise ValueError(f"{path}:{line_no}: unsupported exclusion reason {reason!r}")
            exclusions[reason] += 1
        else:
            raise ValueError(f"{path}:{line_no}: unsupported status {row.get('status')!r}")

    if total == 0:
        raise ValueError(f"{path} did not contain any public report rows")
    return {
        "total": total,
        "mapped_to_target_categories": mapped,
        "excluded": total - mapped,
        "exclusion_counts": dict(sorted(exclusions.items())),
        "target_category_counts": dict(sorted(target_categories.items())),
    }


def summarize_anchors(data: dict[str, Any]) -> dict[str, Any]:
    anchors = data.get("public_anchors", [])
    if not isinstance(anchors, list):
        raise ValueError("public_anchors must be a list")
    counts = Counter()
    failures: list[str] = []
    for idx, anchor in enumerate(anchors, start=1):
        if not isinstance(anchor, dict):
            failures.append(f"anchor {idx} is not an object")
            continue
        if not str(anchor.get("name", "")).strip():
            failures.append(f"anchor {idx} missing name")
        category = normalize_token(anchor.get("target_category", "unspecified")) or "unspecified"
        counts[category] += 1
        if anchor.get("counted_in_hib_training_or_positives") is not False:
            failures.append(f"anchor {idx} must set counted_in_hib_training_or_positives=false")
    return {
        "n_public_anchors": len(anchors),
        "target_category_counts": dict(sorted(counts.items())),
        "failures": failures,
    }


def compare_report_summary(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in ("total", "mapped_to_target_categories", "excluded"):
        if actual.get(key) != expected.get(key):
            failures.append(f"public_reports.{key}: observed {actual.get(key)}, expected {expected.get(key)}")
    actual_exclusions = {normalize_token(k): int(v) for k, v in actual.get("exclusion_counts", {}).items()}
    expected_exclusions = {normalize_token(k): int(v) for k, v in expected.get("exclusion_counts", {}).items()}
    if actual_exclusions != expected_exclusions:
        failures.append(f"public_reports.exclusion_counts: observed {actual_exclusions}, expected {expected_exclusions}")
    return failures


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    counts_path = Path(args.counts)
    counts_data = load_count_summary(counts_path)
    expected_summary = counts_data["public_reports"]
    if args.reports:
        source = "labeled_public_reports"
        public_reports = summarize_public_reports_from_jsonl(Path(args.reports))
    else:
        source = "published_aggregate_counts"
        public_reports = expected_summary

    failures: list[str] = []
    if args.expect_counts:
        failures.extend(compare_report_summary(public_reports, expected_summary))

    anchors = summarize_anchors(counts_data)
    failures.extend(anchors["failures"])

    mapped = public_reports["mapped_to_target_categories"]
    total = public_reports["total"]
    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "source": source,
        "counts_file": str(counts_path),
        "paper_section": counts_data.get("paper_section"),
        "public_reports": {
            **public_reports,
            "mapping_rate": None if total == 0 else mapped / total,
            "allowed_exclusion_reasons": sorted(ALLOWED_EXCLUSIONS),
        },
        "public_anchors": {
            "n_public_anchors": anchors["n_public_anchors"],
            "target_category_counts": anchors["target_category_counts"],
            "counting_rule": "Public anchors test taxonomy scope and are not counted as HIB training or production positives.",
        },
        "private_by_design": counts_data.get("private_by_design", []),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute public-real taxonomy scope accounting for the hostname-injection paper."
    )
    parser.add_argument(
        "--counts",
        default="public_scope/paper_public_scope_counts.json",
        help="Release-safe public-scope count summary or expected-count file.",
    )
    parser.add_argument(
        "--reports",
        default=None,
        help="Optional JSONL public-report labels with status and category/exclusion fields.",
    )
    parser.add_argument(
        "--expect-counts",
        action="store_true",
        help="Fail unless --reports aggregates exactly match --counts.",
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
