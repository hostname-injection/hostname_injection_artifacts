import json
import sys

import pytest

import ccd.diagnostics as diagnostics
import ccd.explain as explain


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
            "ccd.explain",
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
    assert data["threshold"] == 0.5
    assert data["threshold_source"] == "model_bundle_threshold"
    assert data["grouped_thresholds_source"] == "model_bundle_grouped_thresholds"
    assert data["grouped_thresholds_used"] is True
    assert data["decision_rule"] == "score > threshold"
    assert data["score_path"] == {
        "exact_all_cones": True,
        "score_statistic": "deployed_top_r_cone_sketch",
        "normalized_inputs": True,
    }
    assert data["normalizer"]["function"] == "ccd.preprocess.normalize_hostname"
    assert data["explanations"][0]["hostname"] == "alpha.example"
    assert data["explanations"][0]["calibration_group"] == "tenant-a"
    assert data["explanations"][0]["decision_rule"] == "score > threshold"
    assert data["explanations"][0]["threshold_source"] == "model_bundle_grouped_thresholds"


def test_explain_cli_rejects_non_finite_model_bundle_threshold_before_explaining(tmp_path, monkeypatch):
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "explain.json"
    input_path.write_text("alpha.example\n", encoding="utf-8")

    class FailingModel:
        threshold = float("nan")
        grouped_thresholds = None

        def explain(self, hostnames, **kwargs):
            raise AssertionError("explain should not run with invalid model threshold")

    monkeypatch.setattr(explain, "load_model", lambda _: FailingModel())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ccd.explain",
            "--model",
            "model.npz",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(ValueError, match="model bundle threshold.*finite"):
        explain.main()


def test_explain_cli_rejects_uncalibrated_model_bundle_before_explaining(tmp_path, monkeypatch):
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "explain.json"
    input_path.write_text("alpha.example\n", encoding="utf-8")

    class FailingModel:
        threshold = None
        grouped_thresholds = None

        def explain(self, hostnames, **kwargs):
            raise AssertionError("explain should not run without an embedded calibrated threshold")

    monkeypatch.setattr(explain, "load_model", lambda _: FailingModel())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ccd.explain",
            "--model",
            "model.npz",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(ValueError, match="requires a calibrated model bundle"):
        explain.main()


def test_diagnostics_cli_runs(tmp_path, monkeypatch):
    checkpoint = tmp_path / "caho_encoder"
    checkpoint.mkdir()
    (checkpoint / "modules.json").write_text("[]", encoding="utf-8")

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
            "ccd.diagnostics",
            "--checkpoint",
            str(checkpoint),
            "--num-samples",
            "10",
            "--batch-size",
            "4",
        ],
    )

    assert diagnostics.main() == 0
