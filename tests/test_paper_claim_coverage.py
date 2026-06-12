from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paper_claim_coverage_is_recomputable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/recompute_paper_claim_coverage.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["status"] == "pass"
    assert report["derived"]["n_coverage_items"] == 33
    assert report["derived"]["n_figures"] == 7
    assert report["derived"]["n_tables"] == 12
    assert report["derived"]["n_appendices"] == 6
    assert report["derived"]["n_equation_or_formal_items"] == 3
    assert "appendix_b" in report["derived"]["external_required_items"]
    assert "table3" in report["derived"]["external_required_items"]
    assert report["derived"]["coverage_status_counts"]["executable_public"] >= 7


def test_paper_claim_coverage_rejects_unmapped_full_replay_claim(tmp_path: Path) -> None:
    counts = json.loads((ROOT / "paper_claim_coverage" / "paper_claim_coverage.json").read_text(encoding="utf-8"))
    for item in counts["coverage_items"]:
        if item["item_id"] == "appendix_b":
            item["external_required"] = False
            item["external_completion_keyword"] = ""
            break
    bad_counts = tmp_path / "bad_coverage.json"
    bad_counts.write_text(json.dumps(counts), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/recompute_paper_claim_coverage.py",
            "--counts",
            str(bad_counts),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 1
    assert "external_required item mismatch" in completed.stderr
