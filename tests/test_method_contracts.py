from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_method_contracts_are_recomputable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/recompute_method_contracts.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["status"] == "pass"
    assert report["derived"]["n_table1_contracts"] == 3
    assert report["derived"]["ccd_defaults"]["num_cones"] == 4096
    assert report["derived"]["ccd_defaults"]["active_cones"] == 8
    assert report["derived"]["code_path_evidence"]["eq1_logsumexp_score_path_available"] is True
    assert report["derived"]["code_path_evidence"]["normalizer_decodes_utf8_percent_runs"] is True
    assert report["derived"]["code_path_evidence"]["score_paths_normalize_unit_embeddings"] is True
    assert report["derived"]["code_path_evidence"]["exact_score_bypasses_lsh_by_default"] is True
    assert report["derived"]["code_path_evidence"]["grouped_split_conformal_calibration_available"] is True
    assert report["derived"]["code_path_evidence"]["tenant_window_threshold_resolution_available"] is True
    assert report["derived"]["code_path_evidence"]["group_metadata_rejects_empty_values"] is True
    assert report["derived"]["code_path_evidence"]["model_predict_grouped_thresholds_available"] is True
    assert report["derived"]["code_path_evidence"]["grouped_decision_explanations_available"] is True
    assert report["derived"]["code_path_evidence"]["explanations_use_normalized_cone_evidence"] is True
    assert report["derived"]["code_path_evidence"]["calibrated_margin_certificate_available"] is True
    assert report["derived"]["code_path_evidence"]["combined_cmc_then_enumeration_certificate_available"] is True
    assert report["derived"]["code_path_evidence"]["certificate_records_normalizer_and_threshold_scope"] is True
    assert report["derived"]["code_path_evidence"]["exact_full_axis_scan_for_deployed_top_r_statistic"] is True
    assert report["derived"]["code_path_evidence"]["benign_reference_refresh_available"] is True
    assert report["derived"]["code_path_evidence"]["grouped_threshold_refresh_available"] is True
    assert report["derived"]["code_path_evidence"]["grouped_refresh_requires_groups_or_explicit_drop"] is True
    assert report["derived"]["code_path_evidence"]["grouped_refresh_rolls_back_on_calibration_failure"] is True
    assert report["derived"]["code_path_evidence"]["refresh_updates_pb_and_threshold_only"] is True
    assert report["derived"]["edit_manifest"]["deterministic_closure_available"] is True
    assert report["derived"]["edit_manifest"]["utf8_percent_run_closure"] is True
    assert report["derived"]["caho_training_support"]["binary_auxiliary_head_supported"] is True
    assert report["derived"]["caho_training_support"]["binary_auxiliary_head_trains_both_views"] is True
    assert report["derived"]["caho_training_support"]["binary_gradcache_fails_closed"] is True
    assert report["derived"]["caho_training_support"]["contrastive_loss_supported"] is True
    assert report["derived"]["caho_training_support"]["supervised_orbit_contrastive_supported"] is True
    assert report["derived"]["caho_training_support"]["general_supcon_view_alignment"] is True
    assert report["derived"]["caho_training_support"]["benign_orbit_labels_preserve_diversity"] is True
    assert report["derived"]["caho_training_support"]["l2_normalized_binary_inputs"] is True
    assert report["derived"]["caho_training_support"]["deterministic_training_seed_supported"] is True
    assert report["derived"]["caho_training_support"]["adamw_weight_decay_default"] == 0.01
    assert report["derived"]["caho_training_support"]["benchmark_binary_training_script_defaults"] == {
        "batch_size": 256,
        "device": "auto",
        "epochs": 50,
        "lr": 0.0001,
        "seed": 13,
        "weight_decay": 0.01,
    }
    assert report["derived"]["caho_training_support"]["benchmark_binary_training_script_has_cuda_gate"] is True
    assert report["derived"]["bundle_contracts"]["optional_calibrated_threshold_serialization"] is True
    assert report["derived"]["bundle_contracts"]["optional_grouped_calibrated_threshold_serialization"] is True


def test_method_contracts_reject_missing_edit_manifest_prefix(tmp_path: Path) -> None:
    counts = json.loads((ROOT / "method_contracts" / "paper_method_contracts.json").read_text(encoding="utf-8"))
    counts["expected_edit_manifest"]["required_prefixes"].append("E99")
    bad_counts = tmp_path / "bad_counts.json"
    bad_counts.write_text(json.dumps(counts), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "scripts/recompute_method_contracts.py", "--counts", str(bad_counts)],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 1
    assert "edit manifest missing required prefixes" in completed.stderr
