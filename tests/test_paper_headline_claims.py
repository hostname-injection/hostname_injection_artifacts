from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_paper_headline_claims_are_recomputable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/recompute_paper_headline_claims.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    by_id = {claim["item_id"]: claim for claim in report["headline_claims"]}

    assert report["status"] == "pass"
    assert report["derived"]["n_headline_claims"] == 9
    assert report["derived"]["external_required_count"] == 9
    assert by_id["abstract_replay_scale"]["observed"]["replay_rows"] == 200_339_886
    assert by_id["abstract_replay_scale"]["observed"]["tenant_count"] == 835
    assert by_id["abstract_fixed_fpr_recall_latency"]["observed"]["ccd_tpr_at_1e_4"] == pytest.approx(0.935)
    assert by_id["abstract_fixed_fpr_recall_latency"]["observed"]["full_path_median_ms"] == pytest.approx(0.6)
    assert by_id["abstract_live_added_value"]["observed"]["ccd_only_verified_positive"] == 850
    assert by_id["decision_stability_k6"]["observed"]["combined_stable_detected_positive_coverage"] == pytest.approx(0.98)
    assert by_id["synthetic_training_gap"]["observed"]["real_training_gain_over_llm_pp"] == pytest.approx(2.4)
    assert by_id["scope_public_and_static"]["observed"]["public_reports_mapped"] == 118
    assert by_id["hostile_mimicry_range"]["observed"]["min_ccd_recall_1e_4"] == pytest.approx(0.915)
    assert by_id["production_throughput_tail"]["observed"]["single_host_throughput_per_s"] == 48_000


def test_paper_headline_claims_reject_stale_expected_value(tmp_path: Path) -> None:
    counts = json.loads((ROOT / "paper_headline_claims" / "paper_headline_claims.json").read_text(encoding="utf-8"))
    for claim in counts["headline_claims"]:
        if claim["item_id"] == "abstract_live_added_value":
            claim["expected"]["ccd_only_verified_positive"] = 851
            break
    bad_counts = tmp_path / "bad_headline_claims.json"
    bad_counts.write_text(json.dumps(counts), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/recompute_paper_headline_claims.py",
            "--counts",
            str(bad_counts),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 1
    assert "abstract_live_added_value.ccd_only_verified_positive expected 851, observed 850" in completed.stderr
