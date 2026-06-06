import csv
import importlib.util
import json
from pathlib import Path

import pytest


def _load_export_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_hib_release_pipeline_inputs.py"
    spec = importlib.util.spec_from_file_location("_test_export_hib_release_pipeline_inputs", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_release(path: Path) -> None:
    rows = [
        {
            "public_row_id": "train-benign-1",
            "released_artifact": "benign.example",
            "split": "train",
            "label": "resolved_benign",
            "source_family": "dns_host",
            "sink_family": "none",
            "ccd_outputs": {"public_calibration_group": "public_a"},
        },
        {
            "public_row_id": "train-benign-2",
            "released_artifact": "benign.example",
            "split": "train",
            "label": "resolved_benign",
            "source_family": "dns_host",
            "sink_family": "none",
            "ccd_outputs": {"public_calibration_group": "public_a"},
        },
        {
            "public_row_id": "train-positive",
            "released_artifact": "probe)AND%20pg_sleep(5)--.invalid",
            "split": "train",
            "label": "verified_executable_semantics",
            "source_family": "dns_host",
            "sink_family": "query",
            "ccd_outputs": {"public_calibration_group": "public_a"},
        },
        {
            "public_row_id": "cal-benign",
            "released_artifact": "calibration.example",
            "split": "calibration",
            "label": "resolved_benign",
            "source_family": "dns_host",
            "sink_family": "none",
            "ccd_outputs": {"public_calibration_group": "public_a"},
        },
        {
            "public_row_id": "test-positive",
            "released_artifact": "test)AND%20pg_sleep(5)--.invalid",
            "split": "test",
            "label": "verified_executable_semantics",
            "source_family": "dns_host",
            "sink_family": "query",
            "ccd_outputs": {"public_calibration_group": "public_b"},
        },
        {
            "public_row_id": "validation-benign",
            "released_artifact": "recent.example",
            "split": "validation",
            "label": "resolved_benign",
            "source_family": "dns_host",
            "sink_family": "none",
            "ccd_outputs": {"public_calibration_group": "public_a"},
        },
        {
            "public_row_id": "test-unresolved",
            "released_artifact": "unknown.example",
            "split": "test",
            "label": "unresolved",
            "source_family": "dns_host",
            "sink_family": "none",
            "ccd_outputs": {"public_calibration_group": "public_b"},
        },
    ]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_export_public_release_pipeline_inputs_preserves_rows_and_groups(tmp_path: Path) -> None:
    module = _load_export_module()
    release = tmp_path / "hib_release.jsonl"
    out = tmp_path / "pipeline"
    _write_release(release)

    summary = module.export_pipeline_inputs(release, out)

    assert (out / "benign.txt").read_text(encoding="utf-8").splitlines() == [
        "benign.example",
        "benign.example",
    ]
    with (out / "malicious.csv").open(newline="", encoding="utf-8") as handle:
        malicious_rows = list(csv.DictReader(handle))
    assert malicious_rows == [{"hostname": "probe)AND%20pg_sleep(5)--.invalid", "family": "query"}]
    assert (out / "benign_calibration.txt").read_text(encoding="utf-8").splitlines() == [
        "calibration.example",
    ]
    assert (out / "benign_calibration_groups.txt").read_text(encoding="utf-8").splitlines() == [
        "public_a",
    ]
    assert (out / "queries.txt").read_text(encoding="utf-8").splitlines() == [
        "test)AND%20pg_sleep(5)--.invalid",
        "recent.example",
    ]
    assert (out / "query_groups.txt").read_text(encoding="utf-8").splitlines() == ["public_b", "public_a"]
    with (out / "query_labels.csv").open(newline="", encoding="utf-8") as handle:
        label_rows = list(csv.DictReader(handle))
    assert [row["public_row_id"] for row in label_rows] == ["test-positive", "validation-benign"]
    assert summary["files"]["benign.txt"] == 2
    assert summary["policy"]["row_multiplicity_preserved"] is True
    manifest = json.loads((out / "pipeline_inputs_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["query_labels_by_label"] == {
        "resolved_benign": 1,
        "verified_executable_semantics": 1,
    }
    assert manifest["policy"]["unresolved_rows_excluded_from_training_and_queries"] is True


def test_export_requires_benign_and_malicious_query_labels(tmp_path: Path) -> None:
    module = _load_export_module()
    release = tmp_path / "hib_release.jsonl"
    _write_release(release)
    rows = [json.loads(line) for line in release.read_text(encoding="utf-8").splitlines()]

    no_benign_query = tmp_path / "no_benign_query.jsonl"
    no_benign_query.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
            if row["public_row_id"] != "validation-benign"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="query_benign"):
        module.export_pipeline_inputs(no_benign_query, tmp_path / "no_benign_out")

    no_malicious_query = tmp_path / "no_malicious_query.jsonl"
    no_malicious_query.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
            if row["public_row_id"] != "test-positive"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="query_malicious"):
        module.export_pipeline_inputs(no_malicious_query, tmp_path / "no_malicious_out")
