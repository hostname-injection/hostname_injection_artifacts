from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recompute_metrics import compute_metrics


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _row(
    row_id: str,
    *,
    split: str,
    label: str,
    ccd_score: float | None,
    ccd_flag: bool | None,
    calibration_group: str | None = None,
) -> dict:
    outputs = {
        "ccd_flag": ccd_flag,
        "ccd_score_bin": "public_score" if ccd_score is not None else "not_recomputed",
    }
    if ccd_score is not None:
        outputs["ccd_score"] = ccd_score
    if calibration_group is not None:
        outputs["public_calibration_group"] = calibration_group
    return {
        "public_row_id": row_id,
        "released_artifact": f"{row_id}.example",
        "released_canonical_artifact": f"{row_id}.example",
        "source_family": "dns_host",
        "time_bucket": "withheld",
        "split": split,
        "label": label,
        "evidence_tier": "none",
        "sink_family": "none",
        "obfuscation_family": "none",
        "released_length_bucket": "withheld",
        "character_class_mask": "withheld",
        "detector_outputs": outputs,
        "row_integrity_hash": row_id,
    }


def test_recompute_metrics_replays_conformal_threshold_and_excludes_calibration_from_rates(tmp_path: Path) -> None:
    release = tmp_path / "release.jsonl"
    _write_jsonl(
        release,
        [
            _row("cal-1", split="calibration", label="resolved_benign", ccd_score=0.1, ccd_flag=False),
            _row("cal-2", split="calibration", label="resolved_benign", ccd_score=0.2, ccd_flag=False),
            _row("cal-3", split="calibration", label="resolved_benign", ccd_score=0.3, ccd_flag=False),
            _row("pos-1", split="test", label="verified_executable_semantics", ccd_score=0.5, ccd_flag=True),
            _row("pos-2", split="test", label="verified_executable_semantics", ccd_score=0.2, ccd_flag=False),
            _row("neg-1", split="test", label="resolved_benign", ccd_score=0.4, ccd_flag=True),
            _row("neg-2", split="test", label="resolved_benign", ccd_score=0.1, ccd_flag=False),
            _row("unk-1", split="test", label="unresolved", ccd_score=0.9, ccd_flag=True),
        ],
    )

    metrics = compute_metrics(release, alpha=0.25)

    assert metrics["calibration"]["threshold"] == 0.3
    assert metrics["calibration"]["order_statistic_rank"] == 3
    assert metrics["label_accounting"]["metric_positive_rows"] == 2
    assert metrics["label_accounting"]["metric_negative_rows"] == 2
    assert metrics["fixed_fpr_replay"]["status"] == "available"
    assert metrics["fixed_fpr_replay"]["tp"] == 1
    assert metrics["fixed_fpr_replay"]["fn"] == 1
    assert metrics["fixed_fpr_replay"]["fp"] == 1
    assert metrics["fixed_fpr_replay"]["tn"] == 1
    assert metrics["fixed_fpr_replay"]["tpr"] == 0.5
    assert metrics["fixed_fpr_replay"]["fpr"] == 0.5
    assert metrics["resolved_row_replay"]["ccd_flag_metrics"]["tpr"] == 0.5
    assert metrics["ccd_output_counts"]["effective_ccd_flag_sources"]["released_ccd_flag"] == 8


def test_recompute_metrics_reports_missing_public_scores(tmp_path: Path) -> None:
    release = tmp_path / "release.jsonl"
    _write_jsonl(
        release,
        [
            _row("cal-1", split="calibration", label="resolved_benign", ccd_score=None, ccd_flag=None),
            _row("neg-1", split="test", label="resolved_benign", ccd_score=None, ccd_flag=None),
        ],
    )

    metrics = compute_metrics(release)

    assert metrics["calibration"]["threshold_source"] == "not_recomputed_no_public_scores"
    assert metrics["fixed_fpr_replay"]["status"] == "not_available"
    assert metrics["fixed_fpr_replay"]["reason"]
    assert metrics["resolved_row_replay"]["ccd_tpr"] is None


def test_recompute_metrics_replays_grouped_public_calibration_thresholds(tmp_path: Path) -> None:
    release = tmp_path / "release.jsonl"
    _write_jsonl(
        release,
        [
            _row("g1-cal-1", split="calibration", label="resolved_benign", ccd_score=0.1, ccd_flag=False, calibration_group="public_g1"),
            _row("g1-cal-2", split="calibration", label="resolved_benign", ccd_score=0.2, ccd_flag=False, calibration_group="public_g1"),
            _row("g2-cal-1", split="calibration", label="resolved_benign", ccd_score=0.8, ccd_flag=False, calibration_group="public_g2"),
            _row("g2-cal-2", split="calibration", label="resolved_benign", ccd_score=0.9, ccd_flag=False, calibration_group="public_g2"),
            _row("g1-pos", split="test", label="verified_executable_semantics", ccd_score=0.3, ccd_flag=True, calibration_group="public_g1"),
            _row("g1-neg", split="test", label="resolved_benign", ccd_score=0.3, ccd_flag=True, calibration_group="public_g1"),
            _row("g2-pos", split="test", label="verified_executable_semantics", ccd_score=0.85, ccd_flag=False, calibration_group="public_g2"),
            _row("g2-neg", split="test", label="resolved_benign", ccd_score=0.85, ccd_flag=False, calibration_group="public_g2"),
        ],
    )

    metrics = compute_metrics(release, alpha=0.25)

    assert metrics["calibration"]["threshold_source"] == "recomputed_from_public_grouped_calibration_scores"
    assert metrics["calibration"]["n_calibration_groups"] == 2
    assert metrics["calibration"]["grouped_thresholds"]["public_g1"]["threshold"] == 0.2
    assert metrics["calibration"]["grouped_thresholds"]["public_g2"]["threshold"] == 0.9
    assert metrics["fixed_fpr_replay"]["threshold"] is None
    assert metrics["fixed_fpr_replay"]["grouped_thresholds"]["public_g1"]["threshold"] == 0.2
    assert metrics["fixed_fpr_replay"]["tp"] == 1
    assert metrics["fixed_fpr_replay"]["fp"] == 1
    assert metrics["fixed_fpr_replay"]["fn"] == 1
    assert metrics["fixed_fpr_replay"]["tn"] == 1


def test_recompute_metrics_cli_writes_output(tmp_path: Path) -> None:
    release = tmp_path / "release.jsonl"
    out = tmp_path / "metrics.json"
    _write_jsonl(
        release,
        [
            _row("cal-1", split="calibration", label="resolved_benign", ccd_score=0.1, ccd_flag=False),
            _row("neg-1", split="test", label="resolved_benign", ccd_score=0.2, ccd_flag=True),
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "deidentification_release/scripts/recompute_metrics.py",
            "--public-release",
            str(release),
            "--out",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )

    assert json.loads(out.read_text(encoding="utf-8"))["n_rows"] == 2
