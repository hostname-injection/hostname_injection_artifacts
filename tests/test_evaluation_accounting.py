from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_accounting_tables_are_recomputable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/recompute_evaluation_accounting.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["status"] == "pass"
    assert report["derived"]["n_evidence_units"] == 6
    assert report["derived"]["resolved_replay_denominator_units"] == ["replay_entry", "verified_positive"]
    assert report["derived"]["live_stream_units"] == [
        "live_comparison_item",
        "composite_alert",
        "tenant_visible_alert",
    ]
    assert report["derived"]["n_reproducibility_boundary_rows"] == 4
    assert report["manifest_alignment"]["checked"] is True
    assert report["manifest_alignment"]["matched_external_boundaries"] == [
        "replay_accuracy",
        "baselines",
        "live_overlap",
        "sink_evidence",
    ]


def test_evaluation_accounting_rejects_unresolved_denominator_leak(tmp_path: Path) -> None:
    counts = json.loads((ROOT / "evaluation_accounting" / "paper_evaluation_accounting.json").read_text(encoding="utf-8"))
    for unit in counts["table2_evidence_units"]:
        if unit["unit_id"] == "unresolved":
            unit["resolved_replay_denominator"] = True
    bad_counts = tmp_path / "bad_counts.json"
    bad_counts.write_text(json.dumps(counts), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/recompute_evaluation_accounting.py",
            "--counts",
            str(bad_counts),
            "--no-manifest",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 1
    assert "only replay_entry and verified_positive" in completed.stderr
