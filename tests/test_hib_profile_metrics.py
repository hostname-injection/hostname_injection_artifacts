from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_hib_profile_metrics_are_recomputable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/recompute_hib_profile_metrics.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["status"] == "pass"
    assert report["derived"]["replay"]["source_total_rows"] == 200_339_886
    assert report["derived"]["replay"]["resolved_replay_denominator"] == 200_252_900
    assert report["derived"]["replay"]["unresolved_rows"] == 86_986
    assert report["derived"]["replay"]["source_percentages"]["user_login_hostnames"] == pytest.approx(56.25, abs=0.01)
    assert report["derived"]["replay"]["source_percentages"]["dns_resolution_hostnames"] == pytest.approx(43.75, abs=0.01)
    assert report["derived"]["replay"]["event_count_residuals"] == {
        "user_login_hostnames": 1,
        "dns_resolution_hostnames": 1,
    }
    assert report["derived"]["verified_positive_profile"]["attack_family_total"] == 363_401
    assert report["derived"]["verified_positive_profile"]["largest_attack_family"] == "JNDI lookup"
    assert report["derived"]["verified_positive_profile"]["largest_attack_family_share_percent"] == pytest.approx(16.0)
    assert report["derived"]["verified_positive_profile"]["non_none_obfuscation_rate_percent"] == pytest.approx(72.8, abs=0.1)


def test_hib_profile_rejects_inconsistent_attack_family_total(tmp_path: Path) -> None:
    counts = json.loads((ROOT / "hib_profile" / "paper_hib_profile_counts.json").read_text(encoding="utf-8"))
    counts["verified_positive_profile"]["attack_family_distribution"][0]["count"] += 1
    bad_counts = tmp_path / "bad_counts.json"
    bad_counts.write_text(json.dumps(counts), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "scripts/recompute_hib_profile_metrics.py", "--counts", str(bad_counts)],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 1
    assert "share_percent" in completed.stderr or "attack-family counts" in completed.stderr
