from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hib_deid import PrivateConfig, row_integrity_hash, anonymize_csv, canonicalize_artifact, read_jsonl, verify_release
from repair_public_release_fields import repair, repair_row


def _config() -> PrivateConfig:
    return PrivateConfig(row_id_secret="row-secret", artifact_secret="artifact-secret", shuffle_secret="shuffle-secret")


def _write_private_csv(path: Path) -> None:
    rows = []
    for i in range(60):
        label = "M" if i % 3 == 0 else "B"
        content = "fd-dspos-101)AND%20pg_sleep(20)--.customer.internal" if label == "M" else "benign-service.customer.internal"
        rows.append(
            {
                "ROW_ID": f"row-{i:03d}",
                "CDB": "CUSTOMER_PRIVATE",
                "CONTENT": content,
                "HOSTNAME": content,
                "USERNAME": "",
                "CREATED_TIME": "2025-08-01T00:00:00Z",
                "RESOLVED_LABEL_BOTH_M": label,
                "DATASET_FAMILY": "dns_hostnames",
                "CONTENT_TYPE": "HOSTNAME",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_label_preservation_and_public_schema(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    rows = read_jsonl(public_jsonl)
    report = verify_release(private_csv, public_jsonl, audit_dir, _config(), min_k=1)

    assert report["anonymization"]["utility_reproducibility_checks"]["label_preservation_rate"] == 1.0
    assert report["anonymization"]["llm_label_reason_handling"]["labels_preserved"] is True
    assert set(rows[0]) == {
        "public_row_id",
        "released_artifact",
        "released_canonical_artifact",
        "source_family",
        "time_bucket",
        "split",
        "label",
        "evidence_tier",
        "sink_family",
        "obfuscation_family",
        "released_length_bucket",
        "character_class_mask",
        "detector_outputs",
        "row_integrity_hash",
    }


def test_payload_markers_delimiters_and_encoding_preserved(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    malicious = [row for row in read_jsonl(public_jsonl) if row["label"] == "verified_executable_semantics"]

    assert malicious
    assert all("%20" in row["released_artifact"] for row in malicious)
    assert all("--" in row["released_artifact"] for row in malicious)
    assert all(row["sink_family"] == "query" for row in malicious)


def test_optional_public_detector_scores_are_replayable(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    rows = [
        {
            "ROW_ID": f"score-{i:03d}",
            "CDB": "CUSTOMER_PRIVATE",
            "CONTENT": f"svc-{i:03d})AND%20pg_sleep(20)--.customer.internal",
            "HOSTNAME": f"svc-{i:03d})AND%20pg_sleep(20)--.customer.internal",
            "USERNAME": "",
            "CREATED_TIME": "2025-08-01T00:00:00Z",
            "RESOLVED_LABEL_BOTH_M": "M" if i >= 4 else "B",
            "DATASET_FAMILY": "dns_hostnames",
            "CONTENT_TYPE": "HOSTNAME",
            "PUBLIC_CCD_SCORE": "0.95" if i >= 4 else f"0.{i + 1}",
            "PUBLIC_CCD_FLAG": "true" if i >= 4 else "false",
            "PUBLIC_CALIBRATION_GROUP": "public_group_a" if i % 2 == 0 else "public_group_b",
        }
        for i in range(8)
    ]
    with private_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    public_rows = read_jsonl(public_jsonl)

    assert all(row["detector_outputs"]["ccd_score_bin"] == "public_score" for row in public_rows)
    assert all("ccd_score_public" in row["detector_outputs"] for row in public_rows)
    assert all(row["detector_outputs"].get("public_calibration_group") in {"public_group_a", "public_group_b"} for row in public_rows)
    assert any(row["detector_outputs"]["ccd_flag"] is True for row in public_rows)
    assert all(set(row["detector_outputs"]) <= {"ccd_score_bin", "ccd_flag", "ccd_score_public", "public_calibration_group"} for row in public_rows)


def test_public_calibration_group_rejects_tenant_identifying_terms(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    row = {
        "ROW_ID": "score-001",
        "CDB": "CUSTOMER_PRIVATE",
        "CONTENT": "svc-001.example.invalid",
        "HOSTNAME": "svc-001.example.invalid",
        "USERNAME": "",
        "CREATED_TIME": "2025-08-01T00:00:00Z",
        "RESOLVED_LABEL_BOTH_M": "B",
        "DATASET_FAMILY": "dns_hostnames",
        "CONTENT_TYPE": "HOSTNAME",
        "PUBLIC_CCD_SCORE": "0.1",
        "PUBLIC_CALIBRATION_GROUP": "tenant-123",
    }
    with private_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match="tenant-identifying"):
        anonymize_csv(private_csv, public_jsonl, audit_dir, _config())


def test_non_sensitive_spans_preserved_exactly(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    rows = [
        {
            "ROW_ID": f"safe-{i:03d}",
            "CDB": "CUSTOMER_PRIVATE",
            "CONTENT": "api.prod.customer.internal",
            "HOSTNAME": "api.prod.customer.internal",
            "USERNAME": "",
            "CREATED_TIME": "2025-08-01T00:00:00Z",
            "RESOLVED_LABEL_BOTH_M": "B",
            "DATASET_FAMILY": "dns_hostnames",
            "CONTENT_TYPE": "HOSTNAME",
        }
        for i in range(60)
    ]
    with private_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())

    assert all(row["released_artifact"].startswith("api.prod.") for row in read_jsonl(public_jsonl))


def test_sensitive_replacements_preserve_shape(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    report = verify_release(private_csv, public_jsonl, audit_dir, _config(), min_k=1)

    assert report["anonymization"]["utility_reproducibility_checks"]["sensitive_span_length_preservation_or_nearest_safe_rate"] >= 0.99
    assert report["anonymization"]["utility_reproducibility_checks"]["delimiter_position_preservation_rate"] >= 0.995


def test_encoded_sensitive_spans_keep_encoding_style(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    report = verify_release(private_csv, public_jsonl, audit_dir, _config(), min_k=1)

    assert report["anonymization"]["utility_reproducibility_checks"]["encoding_style_preservation_rate"] == 1.0


def test_audit_reports_and_data_card_written(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    verify_release(private_csv, public_jsonl, audit_dir, _config(), min_k=1)

    assert (audit_dir / "anonymization_audit_report.json").exists()
    assert (audit_dir / "anonymization_audit_report.md").exists()
    assert (audit_dir / "nonlinkability_audit_report.json").exists()
    assert (audit_dir / "nonlinkability_audit_report.md").exists()
    assert (audit_dir / "anonymization_shortcut_audit_report.json").exists()
    assert (audit_dir / "anonymization_shortcut_audit_report.md").exists()
    assert (audit_dir / "release_data_card.md").exists()
    data_card = (audit_dir / "release_data_card.md").read_text(encoding="utf-8")
    assert "Synthetic Hostname Realism" in data_card
    assert "intent-signaling generated hostname" in data_card


def test_release_sha256_sidecar_is_sha256sum_compatible(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    sidecar = public_jsonl.with_suffix(public_jsonl.suffix + ".sha256").read_text(encoding="utf-8").strip()
    digest, filename = sidecar.split()

    assert len(digest) == 64
    assert filename == public_jsonl.name


def test_repair_row_preserves_label_and_recomputes_public_fields() -> None:
    row = {
        "public_row_id": "row_ABC",
        "released_artifact": "evil-hacker-service.internal",
        "released_canonical_artifact": "evil-hacker-service.internal",
        "source_family": "dns_host",
        "time_bucket": "2025-W31",
        "split": "test",
        "label": "verified_executable_semantics",
        "evidence_tier": "artifact_supported",
        "sink_family": "url_fetch",
        "obfuscation_family": "none",
        "released_length_bucket": "16-31",
        "character_class_mask": "aaaa",
        "detector_outputs": {"ccd_score_bin": "not_recomputed", "ccd_flag": None},
        "row_integrity_hash": "old",
    }

    repaired = repair_row(row)

    assert repaired["label"] == "verified_executable_semantics"
    assert repaired["sink_family"] == "none"
    assert repaired["evidence_tier"] == "syntax_only"
    assert "evil" not in repaired["released_artifact"].lower()
    assert "hacker" not in repaired["released_artifact"].lower()
    assert repaired["time_bucket"] == "withheld"
    assert repaired["released_canonical_artifact"] == canonicalize_artifact(repaired["released_artifact"])
    assert repaired["row_integrity_hash"] == row_integrity_hash(repaired)


def test_repair_writes_schema_and_sha256_sidecar(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "input.jsonl"
    output_jsonl = tmp_path / "release.v3.jsonl"
    row = {
        "public_row_id": "row_ABC",
        "released_artifact": "evil-hacker-service.internal",
        "released_canonical_artifact": "evil-hacker-service.internal",
        "source_family": "dns_host",
        "time_bucket": "2025-W31",
        "split": "test",
        "label": "verified_executable_semantics",
        "evidence_tier": "artifact_supported",
        "sink_family": "url_fetch",
        "obfuscation_family": "none",
        "released_length_bucket": "16-31",
        "character_class_mask": "aaaa",
        "detector_outputs": {"ccd_score_bin": "not_recomputed", "ccd_flag": None},
        "row_integrity_hash": "old",
    }
    input_jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert repair(input_jsonl, output_jsonl) == 1
    assert output_jsonl.exists()
    assert output_jsonl.with_suffix(".schema.json").exists()
    sidecar = output_jsonl.with_suffix(output_jsonl.suffix + ".sha256")
    digest, filename = sidecar.read_text(encoding="utf-8").strip().split()

    assert len(digest) == 64
    assert filename == output_jsonl.name


def test_anonymization_shortcut_classifier_near_chance_on_balanced_shape_fixture(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    rows = []
    for i in range(80):
        label = "M" if i % 2 == 0 else "B"
        rows.append(
            {
                "ROW_ID": f"balanced-{i:03d}",
                "CDB": "CUSTOMER_PRIVATE",
                "CONTENT": "api-prod-001.customer.internal",
                "HOSTNAME": "api-prod-001.customer.internal",
                "USERNAME": "",
                "CREATED_TIME": "2025-08-01T00:00:00Z",
                "RESOLVED_LABEL_BOTH_M": label,
                "DATASET_FAMILY": "dns_hostnames",
                "CONTENT_TYPE": "HOSTNAME",
            }
        )
    with private_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    verify_release(private_csv, public_jsonl, audit_dir, _config(), min_k=1)
    shortcut = json.loads((audit_dir / "anonymization_shortcut_audit_report.json").read_text(encoding="utf-8"))

    assert shortcut["status"] == "pass"
    assert shortcut["estimated_auroc_from_anonymizer_artifacts"] <= 0.55
