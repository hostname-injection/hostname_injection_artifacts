from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_artifact_claim_checks_runner_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_artifact_claim_checks.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    by_id = {check["id"]: check for check in report["checks"]}

    assert report["status"] == "pass"
    assert report["n_checks"] == 14
    assert report["n_failed"] == 0
    assert by_id["readiness_audit"]["summary"]["checks"]["public_release_gate"] == "pass"
    assert by_id["paper_claim_coverage"]["summary"]["reported_status"] == "pass"
    assert by_id["paper_headline_claims"]["summary"]["reported_status"] == "pass"
    assert by_id["public_sample_replay_metrics"]["summary"]["n_rows"] == 150
    assert by_id["public_sample_replay_metrics"]["summary"]["fixed_fpr_status"] == "available"
    assert by_id["public_sample_replay_metrics"]["summary"]["fixed_fpr_confusion"] == {
        "fn": 4,
        "fp": 6,
        "tn": 85,
        "tp": 41,
    }


def test_artifact_claim_checks_runner_reports_child_failures() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_artifact_claim_checks.py",
            "--python",
            str(ROOT / "does-not-exist-python"),
            "--fail-fast",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert report["status"] == "fail"
    assert report["n_checks"] == 1
    assert "could not execute command" in report["failures"][0]
