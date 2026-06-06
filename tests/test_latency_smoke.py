from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_latency_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_artifact_latency.py"
    spec = importlib.util.spec_from_file_location("_test_benchmark_artifact_latency", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latency_smoke_requires_checkpoint_arg(monkeypatch) -> None:
    module = _load_latency_module()
    monkeypatch.setattr(module.sys, "argv", ["benchmark_artifact_latency.py", "--num-samples", "1"])

    with pytest.raises(SystemExit):
        module.main()


def test_latency_smoke_runs_only_with_trained_caho_checkpoint(tmp_path, monkeypatch) -> None:
    module = _load_latency_module()
    checkpoint = tmp_path / "caho_encoder"
    checkpoint.mkdir()
    (checkpoint / "modules.json").write_text("[]", encoding="utf-8")
    inputs = tmp_path / "queries.txt"
    inputs.write_text("alpha.example\nbeta.example\n", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeEncoder:
        def __init__(self, config):
            captured["model_name"] = config.model_name
            captured["device"] = config.device

        def device_type(self):
            return "cpu"

        def encode_torch(self, hostnames, batch_size=32, normalize=True):
            captured["hostnames"] = list(hostnames)
            captured["batch_size"] = batch_size
            captured["normalize"] = normalize
            return np.ones((len(hostnames), 384), dtype=np.float32)

    monkeypatch.setattr(module, "CahoEncoder", FakeEncoder)

    args = SimpleNamespace(
        checkpoint=str(checkpoint),
        input=str(inputs),
        num_samples=16,
        batch_size=4,
        repeats=1,
        warmup=0,
        device="cpu",
        dim=384,
        num_cones=256,
        active_cones=8,
        temperature=10.0,
        seed=13,
    )

    report = module.build_report(args)

    assert captured["model_name"] == str(checkpoint)
    assert captured["device"] == "cpu"
    assert captured["hostnames"] == ["alpha.example", "beta.example"] * 8
    assert captured["batch_size"] == 4
    assert captured["normalize"] is True
    assert report["status"] == "pass"
    assert report["hardware_dependent"] is True
    assert report["encoder"]["checkpoint"] == str(checkpoint)
    assert report["encoder"]["path"] == "CahoEncoder.encode_torch"
    assert report["scoring_kernel"]["samples"] == 16
    assert report["scoring_kernel"]["path"] == "ccd_scores_logpriors_topk"
    assert report["scoring_kernel"]["ms_per_sample_median"] >= 0.0
