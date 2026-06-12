from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_paper_metric_tables_are_recomputable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/recompute_paper_metric_tables.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["status"] == "pass"
    assert report["derived"]["table5"]["best_tpr_method"] == "CCD"
    assert report["derived"]["table5"]["ccd_tpr_lead_over_best_non_ccd"] == pytest.approx(0.039)
    assert report["derived"]["table6"]["largest_whitebox_drop_variant"] == "Linear head instead of cone sketch"
    assert report["derived"]["table6"]["largest_whitebox_drop_points"] == pytest.approx(37.4)
    assert report["derived"]["table10"]["n_llm_checkpoints"] == 4
    assert report["derived"]["table10"]["checkpoint_providers"] == ["Qwen", "meta-llama"]
    assert report["table10_llm_baselines"][0]["checkpoint_identifier"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert report["derived"]["table12"]["weakest_ccd_1e_5_suite"] == "Cone-escape optimization"
    assert report["derived"]["table12"]["weakest_ccd_1e_5_recall"] == pytest.approx(0.842)
    assert report["derived"]["table12"]["minimum_ccd_margin_over_best_baseline_1e_4"] == pytest.approx(0.138)
    assert report["derived"]["generator_comparison"]["real_training_gain_over_llm"] == pytest.approx(0.024)
    assert report["appendix_f_synthetic_validity"]["public_text_overlap_count"] == 127
