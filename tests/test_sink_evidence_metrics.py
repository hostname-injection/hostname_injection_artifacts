from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sink_evidence_metrics_are_recomputable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/recompute_sink_evidence_metrics.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["status"] == "pass"
    assert report["derived"]["n_cases"] == 3
    assert report["derived"]["case_ids"] == ["diagnostic_helper", "alert_action", "analytics_query"]
    assert report["derived"]["controlled_effects"] == [
        "dns_http_callback",
        "macro_template_expansion",
        "delay_failure_bounded_error",
    ]
    assert report["derived"]["all_cases_controlled_replay"] is True
    assert report["derived"]["all_cases_ccd_before_consumption"] is True
    assert report["derived"]["all_cases_non_compromise_claims"] is True
    assert report["scope"]["production_compromise_claim"] is False
    assert report["scope"]["marker_only_strings_not_positive_without_downstream_support"] is True


def test_sink_evidence_rejects_compromise_claim(tmp_path: Path) -> None:
    counts = json.loads((ROOT / "sink_evidence" / "paper_sink_evidence_counts.json").read_text(encoding="utf-8"))
    counts["scope"]["production_compromise_claim"] = True
    counts["cases"][0]["production_compromise_claim"] = True
    bad_counts = tmp_path / "bad_counts.json"
    bad_counts.write_text(json.dumps(counts), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "scripts/recompute_sink_evidence_metrics.py", "--counts", str(bad_counts)],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 1
    assert "scope.production_compromise_claim must be false" in completed.stderr
