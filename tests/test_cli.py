import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

import ccd.cli as cli_module
from ccd.certify import DecisionCertificate
from ccd.config import CCDConfig, ConeConfig
from ccd.cli import build_parser


def test_train_caho_parser_smoke():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-caho",
            "--benign",
            "benign.txt",
            "--malicious",
            "malicious.csv",
            "--out",
            "caho_encoder",
            "--loss",
            "contrastive",
            "--augmenter",
            "weighted",
            "--grad-cache",
        ]
    )
    assert args.command == "train-caho"
    assert args.loss == "contrastive"
    assert args.augmenter == "weighted"
    assert args.grad_cache is True
    assert args.seed == 13


def test_train_user_logins_parser_smoke():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-user-logins",
            "--train-caho",
            "--output",
            "model.npz",
            "--caho-loss",
            "contrastive",
            "--caho-augmenter",
            "weighted",
            "--caho-grad-cache",
        ]
    )
    assert args.command == "train-user-logins"
    assert args.train_caho is True
    assert args.caho_loss == "contrastive"
    assert args.caho_augmenter == "weighted"
    assert args.caho_grad_cache is True


def test_train_priors_parser_smoke():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-priors",
            "--benign",
            "benign.txt",
            "--malicious",
            "malicious.csv",
            "--output",
            "ccd_model.npz",
        ]
    )
    assert args.command == "train-priors"


def test_train_caho_corpus_parser_smoke():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train-caho-corpus",
            "--benign-dir",
            "benign_dir",
            "--malicious-jsonl-dir",
            "jsonl_dir",
            "--malicious-txt-dir",
            "txt_dir",
            "--out",
            "caho_encoder",
            "--loss",
            "contrastive",
            "--augmenter",
            "weighted",
            "--grad-cache",
            "--contrastive-loss",
            "learnable",
            "--save-best",
            "--no-save-final",
        ]
    )
    assert args.command == "train-caho-corpus"
    assert args.loss == "contrastive"
    assert args.grad_cache is True


def test_score_parser_smoke():
    parser = build_parser()
    args = parser.parse_args(
        [
            "score",
            "--model",
            "ccd_model.npz",
            "--input",
            "queries.txt",
            "--output",
            "scores.csv",
            "--groups",
            "groups.txt",
        ]
    )
    assert args.command == "score"
    assert str(args.groups) == "groups.txt"


def test_read_malicious_csv_preserves_quoted_raw_hostname_fields(tmp_path):
    path = tmp_path / "malicious.csv"
    path.write_text('hostname,family\n"cmd,one""two.example",cmd\n', encoding="utf-8")

    assert cli_module._read_malicious_csv(path) == {"cmd": ['cmd,one"two.example']}


def test_calibrate_parser_smoke():
    parser = build_parser()
    args = parser.parse_args(
        [
            "calibrate",
            "--model",
            "ccd_model.npz",
            "--benign",
            "benign.txt",
            "--output",
            "calibration.json",
            "--groups",
            "groups.txt",
            "--save-model",
            "calibrated_model.npz",
        ]
    )
    assert args.command == "calibrate"
    assert str(args.groups) == "groups.txt"
    assert str(args.save_model) == "calibrated_model.npz"


def test_calibrate_can_embed_threshold_in_model_bundle(tmp_path, monkeypatch):
    benign = tmp_path / "benign.txt"
    benign.write_text("alpha.example\nbeta.example\n")
    groups = tmp_path / "groups.txt"
    groups.write_text("tenant-a\ntenant-b\n")
    calibration = tmp_path / "calibration.json"
    calibrated_model = tmp_path / "calibrated_model.npz"

    config = CCDConfig()
    config.cone = ConeConfig(dim=2, num_cones=2, active_cones=1, use_lsh=False)
    dummy_model = SimpleNamespace(
        config=config,
        cones=SimpleNamespace(axes=np.eye(2, dtype=np.float32)),
        benign_prior=np.array([0.7, 0.3], dtype=np.float32),
        malicious_priors={"fam": np.array([0.2, 0.8], dtype=np.float32)},
        threshold=None,
        grouped_thresholds=None,
        score=lambda hostnames, **_kwargs: np.array([0.1, 0.4], dtype=np.float32),
    )
    saved = {}

    monkeypatch.setattr(cli_module, "load_model", lambda _path: dummy_model)
    monkeypatch.setattr(cli_module, "save_model", lambda path, bundle: saved.update(path=path, bundle=bundle))

    parser = build_parser()
    args = parser.parse_args(
        [
            "calibrate",
            "--model",
            "model.npz",
            "--benign",
            str(benign),
            "--output",
            str(calibration),
            "--groups",
            str(groups),
            "--save-model",
            str(calibrated_model),
            "--alpha",
            "0.5",
            "--no-normalize",
        ]
    )
    args.func(args)

    assert abs(dummy_model.threshold - 0.4) < 1e-6
    assert saved["path"] == calibrated_model
    assert abs(saved["bundle"].threshold - 0.4) < 1e-6
    assert saved["bundle"].grouped_thresholds is not None
    assert abs(saved["bundle"].grouped_thresholds["tenant-b"]["threshold"] - 0.4) < 1e-6
    assert saved["bundle"].config.to_dict() == config.to_dict()
    payload = json.loads(calibration.read_text(encoding="utf-8"))
    assert payload["threshold"] == 0.4000000059604645
    assert payload["order_statistic_rank"] == 2
    assert payload["decision_rule"] == "score > threshold"
    assert payload["calibration_scores"] == "benign_only"
    assert "tenant-a" in payload["grouped_thresholds"]
    assert payload["grouped_thresholds"]["tenant-b"]["decision_rule"] == "score > threshold"
    assert payload["n_calibration_groups"] == 2
    assert payload["score_path"]["normalized_inputs"] is False


def test_score_applies_grouped_thresholds(tmp_path, monkeypatch):
    queries = tmp_path / "queries.txt"
    queries.write_text("alpha.example\nbeta.example\n", encoding="utf-8")
    groups = tmp_path / "groups.txt"
    groups.write_text("tenant-a\ntenant-b\n", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "grouped_thresholds": {
                    "tenant-a": {"threshold": 0.7},
                    "tenant-b": {"threshold": 0.4},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "scores.csv"

    dummy_model = SimpleNamespace(
        threshold=None,
        grouped_thresholds=None,
        score=lambda hostnames, **_kwargs: np.array([0.6, 0.6], dtype=np.float32),
    )
    monkeypatch.setattr(cli_module, "load_model", lambda _path: dummy_model)

    parser = build_parser()
    args = parser.parse_args(
        [
            "score",
            "--model",
            "model.npz",
            "--input",
            str(queries),
            "--output",
            str(output),
            "--calibration",
            str(calibration),
            "--groups",
            str(groups),
            "--require-group-thresholds",
            "--no-normalize",
        ]
    )
    args.func(args)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "hostname,calibration_group,threshold,score,prediction"
    assert lines[1].endswith("tenant-a,0.700000,0.600000,0")
    assert lines[2].endswith("tenant-b,0.400000,0.600000,1")


def test_score_csv_preserves_raw_artifact_commas(tmp_path, monkeypatch):
    queries = tmp_path / "queries.txt"
    queries.write_text("alpha,one.example\n", encoding="utf-8")
    output = tmp_path / "scores.csv"

    dummy_model = SimpleNamespace(
        threshold=0.5,
        grouped_thresholds=None,
        score=lambda hostnames, **_kwargs: np.array([0.6], dtype=np.float32),
    )
    monkeypatch.setattr(cli_module, "load_model", lambda _path: dummy_model)

    parser = build_parser()
    args = parser.parse_args(
        [
            "score",
            "--model",
            "model.npz",
            "--input",
            str(queries),
            "--output",
            str(output),
            "--no-normalize",
        ]
    )
    args.func(args)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows == [["hostname", "threshold", "score", "prediction"], ["alpha,one.example", "0.500000", "0.600000", "1"]]


def test_score_uses_grouped_thresholds_from_model_bundle(tmp_path, monkeypatch):
    queries = tmp_path / "queries.txt"
    queries.write_text("alpha.example\nbeta.example\n", encoding="utf-8")
    groups = tmp_path / "groups.txt"
    groups.write_text("tenant-a\ntenant-b\n", encoding="utf-8")
    output = tmp_path / "scores.csv"

    dummy_model = SimpleNamespace(
        threshold=0.5,
        grouped_thresholds={
            "tenant-a": {"threshold": 0.7},
            "tenant-b": {"threshold": 0.4},
        },
        score=lambda hostnames, **_kwargs: np.array([0.6, 0.6], dtype=np.float32),
    )
    monkeypatch.setattr(cli_module, "load_model", lambda _path: dummy_model)

    parser = build_parser()
    args = parser.parse_args(
        [
            "score",
            "--model",
            "model.npz",
            "--input",
            str(queries),
            "--output",
            str(output),
            "--groups",
            str(groups),
            "--require-group-thresholds",
            "--no-normalize",
        ]
    )
    args.func(args)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[1].endswith("tenant-a,0.700000,0.600000,0")
    assert lines[2].endswith("tenant-b,0.400000,0.600000,1")


def test_group_files_reject_empty_group_ids(tmp_path):
    groups = tmp_path / "groups.txt"
    groups.write_text("tenant-a\n\n tenant-b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="groups file contains empty values"):
        cli_module._read_parallel_lines(groups, 2, field_name="groups")


def test_refresh_benign_parser_smoke():
    parser = build_parser()
    args = parser.parse_args(
        [
            "refresh-benign",
            "--model",
            "ccd_model.npz",
            "--benign",
            "benign_window.txt",
            "--output",
            "ccd_model.refreshed.npz",
            "--report",
            "refresh.json",
            "--groups",
            "groups.txt",
            "--drop-grouped-thresholds",
        ]
    )
    assert args.command == "refresh-benign"
    assert str(args.output) == "ccd_model.refreshed.npz"
    assert str(args.report) == "refresh.json"
    assert str(args.groups) == "groups.txt"
    assert args.drop_grouped_thresholds is True


def test_refresh_benign_saves_updated_model_bundle(tmp_path, monkeypatch):
    benign = tmp_path / "benign_window.txt"
    benign.write_text("alpha.example\nbeta.example\nbeta2.example\n", encoding="utf-8")
    groups = tmp_path / "groups.txt"
    groups.write_text("tenant-a\ntenant-b\ntenant-b\n", encoding="utf-8")
    report_path = tmp_path / "refresh.json"
    refreshed_model = tmp_path / "refreshed.npz"

    class DummyEncoder:
        def encode(self, hostnames, batch_size=64, normalize=True):
            rows = []
            for host in hostnames:
                rows.append([1.0, 0.0] if host.startswith("alpha") else [0.0, 1.0])
            return np.array(rows, dtype=np.float32)

    config = CCDConfig()
    config.cone = ConeConfig(dim=2, num_cones=2, active_cones=1, temperature=1.0, use_lsh=False)
    axes = np.eye(2, dtype=np.float32)
    from ccd.cone import ConePartition
    from ccd.model import CCDModel

    model = CCDModel(
        config=config,
        encoder=DummyEncoder(),
        cones=ConePartition.build(config.cone, axes=axes),
        benign_prior=np.array([0.95, 0.05], dtype=np.float32),
        malicious_priors={"fam": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=7.0,
        grouped_thresholds={"old": {"threshold": 7.0}},
    )
    saved = {}

    monkeypatch.setattr(cli_module, "load_model", lambda _path: model)
    monkeypatch.setattr(cli_module, "save_model", lambda path, bundle: saved.update(path=path, bundle=bundle))

    parser = build_parser()
    args = parser.parse_args(
        [
            "refresh-benign",
            "--model",
            "model.npz",
            "--benign",
            str(benign),
            "--output",
            str(refreshed_model),
            "--report",
            str(report_path),
            "--groups",
            str(groups),
            "--alpha",
            "0.5",
            "--no-normalize",
        ]
    )
    args.func(args)

    assert saved["path"] == refreshed_model
    assert saved["bundle"].benign_prior[1] > saved["bundle"].benign_prior[0]
    assert np.allclose(saved["bundle"].malicious_priors["fam"], np.array([0.1, 0.9], dtype=np.float32))
    assert saved["bundle"].threshold == model.threshold
    assert saved["bundle"].grouped_thresholds is not None
    assert set(saved["bundle"].grouped_thresholds) == {"tenant-a", "tenant-b"}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["refresh_scope"]["malicious_priors_fixed"] is True
    assert report["refresh_scope"]["grouped_thresholds_updated"] is True
    assert report["threshold_source"] == "grouped_benign_refresh_scores"
    assert report["order_statistic_rank"] == 2
    assert report["decision_rule"] == "score > threshold"
    assert report["calibration_scores"] == "benign_only"
    assert report["n_calibration_groups"] == 2
    assert report["refresh_scope"]["encoder_config_fixed"] is True
    assert report["score_path"]["normalized_inputs"] is False
    assert report["input"]["num_hostnames"] == 3


def test_certify_parser_smoke():
    parser = build_parser()
    args = parser.parse_args(
        [
            "certify",
            "--model",
            "ccd_model.npz",
            "--input",
            "queries.txt",
            "--output",
            "certificates.json",
            "--radius",
            "1",
            "--groups",
            "groups.txt",
            "--edits",
            "E3_delimiter,E5_case",
            "--cert-method",
            "combined",
            "--sketch-lipschitz",
            "0.1",
            "--embedding-rotation-bound",
            "0.2",
        ]
    )
    assert args.command == "certify"
    assert args.radius == 1
    assert str(args.groups) == "groups.txt"
    assert args.edits == "E3_delimiter,E5_case"
    assert args.cert_method == "combined"
    assert args.sketch_lipschitz == 0.1
    assert args.embedding_rotation_bound == 0.2


def test_certify_writes_scope_and_combined_method_args(tmp_path, monkeypatch):
    input_path = tmp_path / "queries.txt"
    input_path.write_text("HTTP://WWW.Example.COM/path\nbeta.example\n", encoding="utf-8")
    groups_path = tmp_path / "groups.txt"
    groups_path.write_text("tenant-a\ntenant-b\n", encoding="utf-8")
    output_path = tmp_path / "certificates.json"
    seen = {"thresholds": []}

    class DummyModel:
        threshold = 0.25
        grouped_thresholds = {
            "tenant-a": {"threshold": 0.7},
            "tenant-b": {"threshold": 0.4},
        }

        def certify(self, hostname, **kwargs):
            seen["hostname"] = hostname
            seen["thresholds"].append(kwargs["threshold"])
            seen["kwargs"] = kwargs
            return DecisionCertificate(
                certified=True,
                prediction=True,
                method="calibrated_margin",
                radius=kwargs["radius"],
                threshold=kwargs["threshold"],
                base_score=1.0,
                margin=0.75,
                checked=0,
                max_score_movement=0.1,
            )

    monkeypatch.setattr(cli_module, "load_model", lambda _path: DummyModel())

    parser = build_parser()
    args = parser.parse_args(
        [
            "certify",
            "--model",
            "model.npz",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--radius",
            "2",
            "--groups",
            str(groups_path),
            "--require-group-thresholds",
            "--cert-method",
            "combined",
            "--sketch-lipschitz",
            "0.1",
            "--embedding-rotation-bound",
            "0.2",
        ]
    )
    args.func(args)

    assert seen["hostname"] == "beta.example"
    assert seen["thresholds"] == [0.7, 0.4]
    assert seen["kwargs"]["method"] == "combined"
    assert seen["kwargs"]["sketch_lipschitz"] == 0.1
    assert seen["kwargs"]["embedding_rotation_bound"] == 0.2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["cert_method"] == "combined"
    assert payload["grouped_thresholds_used"] is True
    assert payload["threshold_source"] == "model_bundle_threshold"
    assert payload["grouped_thresholds_source"] == "model_bundle_grouped_thresholds"
    assert payload["score_path"]["exact_all_cones"] is True
    assert "all cone axes are scanned exactly" in payload["score_path"]["exact_all_cones_meaning"]
    assert payload["score_path"]["score_statistic"] == "deployed_top_r_cone_sketch"
    assert payload["score_path"]["lsh_bypassed"] is True
    assert payload["normalizer"] == {
        "enabled": True,
        "function": "ccd.preprocess.normalize_hostname",
        "unicode_form": "NFKC",
        "decode_percent": True,
        "decode_utf8_percent_runs": True,
        "idna_roundtrip": True,
        "per_certificate_trace": True,
    }
    assert payload["edit_manifest"]["version"] == "Eraw-public-v2"
    assert payload["certificates"][0]["normalized_hostname"] == "www.example.com"
    assert payload["certificates"][0]["normalization_trace"]["segmentation"]["scheme"] == "http"
    assert payload["certificates"][0]["normalization_trace"]["segmentation"]["path_present"] is True
    assert payload["certificates"][0]["calibration_group"] == "tenant-a"
    assert payload["certificates"][0]["threshold_source"] == "model_bundle_grouped_thresholds"
    assert payload["certificates"][0]["method"] == "calibrated_margin"


def test_eval_caho_parser_smoke():
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval-caho",
            "--input",
            "queries.txt",
            "--output",
            "embeddings.npz",
        ]
    )
    assert args.command == "eval-caho"
