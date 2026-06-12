#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VERDICT_MAP = {
    "tp": "true_positive",
    "true_positive": "true_positive",
    "true-positive": "true_positive",
    "fp": "false_positive",
    "false_positive": "false_positive",
    "false-positive": "false_positive",
    "fn": "missed_delayed_path",
    "false_negative": "missed_delayed_path",
    "false-negative": "missed_delayed_path",
    "missed": "missed_delayed_path",
    "missed_delayed_path": "missed_delayed_path",
    "missed-delayed-path": "missed_delayed_path",
}
COUNT_KEYS = ("true_positive", "false_positive", "missed_delayed_path")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_verdict(value: object) -> str:
    verdict = VERDICT_MAP.get(str(value).strip().lower())
    if verdict is None:
        raise ValueError(f"unsupported source-reachability verdict: {value!r}")
    return verdict


def empty_tool_counts() -> dict[str, int]:
    return {key: 0 for key in COUNT_KEYS}


def load_count_summary(path: Path) -> dict[str, dict[str, int]]:
    data = load_json(path)
    tools = data.get("tools")
    if not isinstance(tools, dict) or not tools:
        raise ValueError(f"{path} must contain a non-empty 'tools' object")
    counts: dict[str, dict[str, int]] = {}
    for tool, info in tools.items():
        if not isinstance(info, dict):
            raise ValueError(f"{path}: tool {tool!r} must be an object")
        tool_counts = empty_tool_counts()
        for key in COUNT_KEYS:
            value = info.get(key)
            if value is None:
                raise ValueError(f"{path}: tool {tool!r} missing {key}")
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{path}: tool {tool!r} has invalid {key}: {value!r}")
            tool_counts[key] = value
        counts[str(tool)] = tool_counts
    return counts


def aggregate_labeled_findings(path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL record: {exc}") from exc
        tool = str(row.get("tool", "")).strip().lower()
        if not tool:
            raise ValueError(f"{path}:{line_no}: missing tool")
        verdict = normalize_verdict(row.get("verdict"))
        counts.setdefault(tool, empty_tool_counts())[verdict] += 1
    if not counts:
        raise ValueError(f"{path} did not contain any labeled findings")
    return counts


def summarize_counts(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for tool, tool_counts in sorted(counts.items()):
        tp = tool_counts["true_positive"]
        fp = tool_counts["false_positive"]
        missed = tool_counts["missed_delayed_path"]
        reported = tp + fp
        relevant = tp + missed
        summary[tool] = {
            **tool_counts,
            "reported_findings": reported,
            "relevant_delayed_flows": relevant,
            "precision": None if reported == 0 else tp / reported,
            "recall": None if relevant == 0 else tp / relevant,
            "missed_delayed_path_rate": None if relevant == 0 else missed / relevant,
        }
    return summary


def compare_counts(
    actual: dict[str, dict[str, int]],
    expected: dict[str, dict[str, int]],
) -> list[str]:
    failures: list[str] = []
    for tool, expected_counts in sorted(expected.items()):
        if tool not in actual:
            failures.append(f"missing tool in labeled findings: {tool}")
            continue
        for key in COUNT_KEYS:
            observed = actual[tool].get(key)
            wanted = expected_counts[key]
            if observed != wanted:
                failures.append(f"{tool}.{key}: observed {observed}, expected {wanted}")
    for tool in sorted(set(actual) - set(expected)):
        failures.append(f"unexpected tool in labeled findings: {tool}")
    return failures


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    counts_path = Path(args.counts)
    expected_counts = load_count_summary(counts_path)
    if args.labels:
        source = "labeled_findings"
        counts = aggregate_labeled_findings(Path(args.labels))
    else:
        source = "published_aggregate_counts"
        counts = expected_counts

    failures: list[str] = []
    if args.expect_counts:
        failures.extend(compare_counts(counts, expected_counts))

    paper_counts = load_json(counts_path)
    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "source": source,
        "counts_file": str(counts_path),
        "paper_section": paper_counts.get("paper_section"),
        "corpus": paper_counts.get("corpus", {}),
        "private_by_design": paper_counts.get("private_by_design", []),
        "tools": summarize_counts(counts),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute source-code reachability accounting for the paper's static-analysis scope check."
    )
    parser.add_argument(
        "--counts",
        default="source_reachability/paper_source_reachability_counts.json",
        help="Release-safe paper count summary or expected-count file.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional JSONL file with labeled findings. Each row needs tool and verdict.",
    )
    parser.add_argument(
        "--expect-counts",
        action="store_true",
        help="Fail unless labeled finding counts exactly match --counts.",
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
