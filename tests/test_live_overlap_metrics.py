from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run_live_overlap(*args: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "scripts/recompute_live_overlap_metrics.py", *args],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_paper_live_overlap_counts_are_recomputable() -> None:
    report = _run_live_overlap()

    assert report["status"] == "pass"
    assert report["source"] == "published_aggregate_counts"
    assert report["table7"]["overlap"]["verified_positive"] == 2300
    assert report["table7"]["ccd_only"]["verified_positive"] == 850
    assert report["table7"]["regex_waf_only"]["verified_positive"] == 0
    assert report["derived"]["ccd_verified_live_positives"] == 3150
    assert report["derived"]["ccd_only_verified_live_positives"] == 850
    assert report["derived"]["baseline_only_all_alerts"] == 200
    assert report["derived"]["baseline_only_uncertain"] == 90
    assert report["derived"]["baseline_only_verified_benign"] == 110
    assert report["detector_bearing"]["ccd_bearing"]["lower_bound_reviewed_item_ppv"] == pytest.approx(3150 / 3355)
    assert report["detector_bearing"]["regex_waf_bearing"]["lower_bound_reviewed_item_ppv"] == pytest.approx(2300 / 2700)
    assert report["detector_bearing"]["ccd_bearing"]["verified_benign_per_day"] == pytest.approx(110 / 92)
    assert report["detector_bearing"]["regex_waf_bearing"]["verified_benign_per_day"] == pytest.approx(220 / 92)
    assert report["case_study_boundary"]["compromise_claim"] is False


def test_labeled_live_overlap_rows_can_be_checked() -> None:
    report = _run_live_overlap(
        "--labels",
        "live_overlap/fixtures/masked_live_labels.jsonl",
        "--counts",
        "live_overlap/fixtures/expected_fixture_counts.json",
        "--expect-counts",
    )

    assert report["status"] == "pass"
    assert report["source"] == "labeled_live_items"
    assert report["derived"]["ccd_verified_live_positives"] == 2
    assert report["detector_bearing"]["ccd_bearing"]["all_alerts"] == 4
    assert report["detector_bearing"]["regex_waf_bearing"]["all_alerts"] == 5
