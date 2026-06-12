from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_public_scope(*args: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "scripts/recompute_public_scope_metrics.py", *args],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_paper_public_scope_counts_are_recomputable() -> None:
    report = _run_public_scope()

    assert report["status"] == "pass"
    assert report["source"] == "published_aggregate_counts"
    assert report["public_reports"]["total"] == 127
    assert report["public_reports"]["mapped_to_target_categories"] == 118
    assert report["public_reports"]["excluded"] == 9
    assert report["public_reports"]["mapping_rate"] == 118 / 127
    assert set(report["public_reports"]["allowed_exclusion_reasons"]) == {
        "certificate-validation",
        "quic",
        "rebinding",
    }
    assert report["public_anchors"]["n_public_anchors"] == 4


def test_labeled_public_scope_reports_can_be_checked() -> None:
    report = _run_public_scope(
        "--reports",
        "public_scope/fixtures/public_report_labels.jsonl",
        "--counts",
        "public_scope/fixtures/expected_fixture_counts.json",
        "--expect-counts",
    )

    assert report["status"] == "pass"
    assert report["source"] == "labeled_public_reports"
    assert report["public_reports"]["mapping_rate"] == 0.6
    assert report["public_reports"]["exclusion_counts"] == {"quic": 1, "rebinding": 1}
    assert report["public_anchors"]["n_public_anchors"] == 1
