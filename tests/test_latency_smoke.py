from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_latency_smoke_scoring_kernel_path_runs() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_artifact_latency.py",
            "--skip-encoder",
            "--num-samples",
            "16",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["status"] == "pass"
    assert report["hardware_dependent"] is True
    assert report["encoder"] is None
    assert report["scoring_kernel"]["samples"] == 16
    assert report["scoring_kernel"]["path"] == "ccd_scores_logpriors_topk"
    assert report["scoring_kernel"]["ms_per_sample_median"] >= 0.0
