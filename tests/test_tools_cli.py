import json
import sys

import numpy as np

import ccd.diagnostics as diagnostics
import ccd.explain as explain
import ccd.score_cli as score_cli


def test_score_cli_writes_output(tmp_path, monkeypatch):
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "out.csv"
    input_path.write_text("alpha.example\nbeta.example\n")

    seen = {}

    class DummyModel:
        threshold = 0.5

        def score(self, hostnames, batch_size=32, normalize=True, approximate=False, approximate_k=None):
            seen["approximate"] = approximate
            seen["approximate_k"] = approximate_k
            return np.array([0.1, 0.9], dtype=np.float32)

    monkeypatch.setattr(score_cli, "load_model", lambda _: DummyModel())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ccd-score",
            "--model",
            "model.npz",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--approximate",
            "--approximate-k",
            "2",
        ],
    )

    assert score_cli.main() == 0
    assert output_path.exists()
    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert seen["approximate"] is True
    assert seen["approximate_k"] == 2


def test_score_cli_applies_grouped_thresholds(tmp_path, monkeypatch):
    input_path = tmp_path / "input.txt"
    groups_path = tmp_path / "groups.txt"
    output_path = tmp_path / "out.csv"
    input_path.write_text("alpha.example\nbeta.example\n", encoding="utf-8")
    groups_path.write_text("tenant-a\ntenant-b\n", encoding="utf-8")

    class DummyModel:
        threshold = 0.5
        grouped_thresholds = {
            "tenant-a": {"threshold": 0.7},
            "tenant-b": {"threshold": 0.4},
        }

        def score(self, hostnames, batch_size=32, normalize=True, approximate=False, approximate_k=None):
            return np.array([0.6, 0.6], dtype=np.float32)

    monkeypatch.setattr(score_cli, "load_model", lambda _: DummyModel())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ccd-score",
            "--model",
            "model.npz",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--groups",
            str(groups_path),
            "--require-group-thresholds",
        ],
    )

    assert score_cli.main() == 0
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "hostname,calibration_group,threshold,score,prediction"
    assert lines[1].endswith("tenant-a,0.700000,0.600000,0")
    assert lines[2].endswith("tenant-b,0.400000,0.600000,1")


def test_explain_cli_outputs_json(tmp_path, monkeypatch):
    input_path = tmp_path / "input.txt"
    groups_path = tmp_path / "groups.txt"
    output_path = tmp_path / "explain.json"
    input_path.write_text("alpha.example\nbeta.example\n", encoding="utf-8")
    groups_path.write_text("tenant-a\ntenant-b\n", encoding="utf-8")

    seen = {}

    class DummyModel:
        threshold = 0.5
        grouped_thresholds = {
            "tenant-a": {"threshold": 0.7},
            "tenant-b": {"threshold": 0.4},
        }

        def explain(
            self,
            hostnames,
            batch_size=32,
            normalize=True,
            top_k=3,
            calibration_groups=None,
            missing_group_threshold="default",
            approximate=False,
            approximate_k=None,
        ):
            seen["calibration_groups"] = calibration_groups
            seen["missing_group_threshold"] = missing_group_threshold
            return [
                {
                    "index": 0,
                    "hostname": hostnames[0],
                    "score": 0.25,
                    "prediction": 1,
                    "threshold": 0.7,
                    "calibration_group": calibration_groups[0],
                    "top_cones": [],
                }
            ]

    monkeypatch.setattr(explain, "load_model", lambda _: DummyModel())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ccd-explain",
            "--model",
            "model.npz",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--groups",
            str(groups_path),
            "--require-group-thresholds",
        ],
    )

    assert explain.main() == 0
    data = json.loads(output_path.read_text())
    assert seen["calibration_groups"] == ["tenant-a", "tenant-b"]
    assert seen["missing_group_threshold"] == "error"
    assert data["count"] == 1
    assert data["grouped_thresholds_used"] is True
    assert data["explanations"][0]["hostname"] == "alpha.example"
    assert data["explanations"][0]["calibration_group"] == "tenant-a"


def test_diagnostics_cli_runs(monkeypatch):
    class DummyEncoder:
        def __init__(self, config):
            self.config = config

        def device_type(self):
            return "cpu"

        def encode_torch(self, texts, batch_size=32, normalize=True):
            return texts

    monkeypatch.setattr(diagnostics, "CahoEncoder", DummyEncoder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ccd-diagnose",
            "--num-samples",
            "10",
            "--batch-size",
            "4",
        ],
    )

    assert diagnostics.main() == 0
