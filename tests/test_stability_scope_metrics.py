from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_stability_scope_metrics_are_recomputable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/recompute_stability_scope_metrics.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["status"] == "pass"
    assert report["derived"]["decision_stability"]["edit_budget_k"] == 6
    assert report["derived"]["decision_stability"]["combined_stable_detected_positive_coverage_percent"] == pytest.approx(98.0)
    assert report["derived"]["decision_stability"]["combined_minus_sec_alone_points"] == pytest.approx(38.0)
    assert report["derived"]["decision_stability"]["combined_minus_native_max_points"] == pytest.approx(50.0)
    assert report["derived"]["holdout_depth"]["ccd_gain_over_strongest_non_llm_baseline_points"] == [6.0, 8.0]
    assert report["derived"]["holdout_depth"]["zero_shot_depth5_recall_range"] == [0.59, 0.71]
    assert report["derived"]["holdout_depth"]["seen_macro_recall_drop_depth2_to_depth5_points"] == pytest.approx(9.0)
    assert report["derived"]["drift_refresh"]["refresh_runtime_seconds"] == pytest.approx(40.0)
    assert report["derived"]["drift_refresh"]["max_fpr_deviation_as_target_fraction"] == pytest.approx(0.2)
    assert report["derived"]["public_real_scope"]["public_real_gain_over_synthetic_points"] == pytest.approx(3.8)
    assert report["derived"]["public_real_scope"]["mixed_gain_over_public_real_points"] == pytest.approx(1.7)


def test_stability_scope_rejects_overbroad_certificate_claim(tmp_path: Path) -> None:
    counts = json.loads((ROOT / "stability_scope" / "paper_stability_scope_counts.json").read_text(encoding="utf-8"))
    counts["decision_stability"]["downstream_sink_safety_claim"] = True
    bad_counts = tmp_path / "bad_counts.json"
    bad_counts.write_text(json.dumps(counts), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "scripts/recompute_stability_scope_metrics.py", "--counts", str(bad_counts)],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 1
    assert "downstream_sink_safety_claim must be false" in completed.stderr
