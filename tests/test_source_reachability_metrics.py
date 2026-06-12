from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_source_reachability(*args: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "scripts/recompute_source_reachability_metrics.py", *args],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_paper_source_reachability_counts_are_recomputable() -> None:
    report = _run_source_reachability()

    assert report["status"] == "pass"
    assert report["source"] == "published_aggregate_counts"
    assert report["corpus"]["n_repositories"] == 50
    assert report["tools"]["codeql"]["true_positive"] == 10
    assert report["tools"]["codeql"]["false_positive"] == 2
    assert report["tools"]["codeql"]["missed_delayed_path"] == 38
    assert report["tools"]["codeql"]["reported_findings"] == 12
    assert report["tools"]["codeql"]["relevant_delayed_flows"] == 48
    assert report["tools"]["semgrep"]["true_positive"] == 6
    assert report["tools"]["semgrep"]["false_positive"] == 2
    assert report["tools"]["semgrep"]["missed_delayed_path"] == 42
    assert report["tools"]["semgrep"]["reported_findings"] == 8
    assert report["tools"]["semgrep"]["relevant_delayed_flows"] == 48


def test_labeled_source_reachability_findings_can_be_checked() -> None:
    report = _run_source_reachability(
        "--labels",
        "source_reachability/fixtures/labeled_findings.jsonl",
        "--counts",
        "source_reachability/fixtures/expected_fixture_counts.json",
        "--expect-counts",
    )

    assert report["status"] == "pass"
    assert report["source"] == "labeled_findings"
    assert report["tools"]["codeql"]["precision"] == 0.5
    assert report["tools"]["codeql"]["recall"] == 0.5
    assert report["tools"]["semgrep"]["precision"] == 1.0
    assert report["tools"]["semgrep"]["recall"] == 1 / 3
