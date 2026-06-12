from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_production_latency_metrics_are_recomputable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/recompute_production_latency_metrics.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["status"] == "pass"
    assert report["derived"]["full_path_latency"]["median_ms"] == pytest.approx(0.60)
    assert report["derived"]["full_path_latency"]["p99_ms"] == pytest.approx(3.92)
    assert report["derived"]["full_path_latency"]["p999_ms"] == pytest.approx(7.80)
    assert report["derived"]["full_path_latency"]["single_host_throughput_k_per_s"] == pytest.approx(48.0)
    assert report["derived"]["scoring_kernel_latency"]["p50_ms"] == pytest.approx(0.04)
    assert report["derived"]["scoring_kernel_latency"]["p99_ms"] == pytest.approx(0.11)
    assert report["derived"]["table5_alignment"]["aligned_with_full_path_tails"] is True
    assert report["derived"]["baseline_context"]["llm_slowdown_range_x"] == [100.0, 1000.0]
    assert report["local_smoke_boundary"]["expected_to_reproduce_production_latency"] is False


def test_production_latency_rejects_misaligned_table5_tails(tmp_path: Path) -> None:
    counts = json.loads((ROOT / "production_latency" / "paper_production_latency_counts.json").read_text(encoding="utf-8"))
    counts["table5_alignment"]["ccd"]["p99_ms"] = 4.0
    bad_counts = tmp_path / "bad_counts.json"
    bad_counts.write_text(json.dumps(counts), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "scripts/recompute_production_latency_metrics.py", "--counts", str(bad_counts)],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 1
    assert "Table 5 CCD p99 does not match" in completed.stderr
