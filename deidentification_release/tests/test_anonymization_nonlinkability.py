from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hib_deid import PrivateConfig, anonymize_csv, anonymize_csv_files, canonicalize_artifact, public_row_id, read_jsonl, verify_release, verify_release_streaming, write_jsonl


def _write_private_csv(path: Path) -> None:
    rows = []
    for i in range(60):
        rows.append(
            {
                "ROW_ID": f"dup-{i:03d}",
                "CDB": "PRIVATE_TENANT",
                "CONTENT": "acme-prod.service.customer.example.com",
                "HOSTNAME": "acme-prod.service.customer.example.com",
                "USERNAME": "",
                "CREATED_TIME": "2025-08-01T00:00:00Z",
                "RESOLVED_LABEL_BOTH_M": "B",
                "DATASET_FAMILY": "dns_hostnames",
                "CONTENT_TYPE": "HOSTNAME",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _config() -> PrivateConfig:
    return PrivateConfig(row_id_secret="row-secret", artifact_secret="artifact-secret", shuffle_secret="shuffle-secret")


def test_duplicate_private_hostnames_map_to_distinct_public_artifacts(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    rows = read_jsonl(public_jsonl)
    artifacts = [row["released_artifact"] for row in rows]
    canonicals = [row["released_canonical_artifact"] for row in rows]

    assert len(artifacts) == len(set(artifacts))
    assert len(canonicals) == len(set(canonicals))
    assert "dedup_hostname_id" not in rows[0]
    assert "unique_host_hash" not in rows[0]
    assert "tenant_surrogate" not in rows[0]


def test_duplicate_private_hostnames_map_to_distinct_public_canonical_artifacts(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    rows = read_jsonl(public_jsonl)

    assert len({row["released_canonical_artifact"] for row in rows}) == len(rows)


def test_no_public_stable_hostname_identifier_fields(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    rows = read_jsonl(public_jsonl)

    forbidden = {"dedup_hostname_id", "unique_host_hash", "stable_hostname_hash", "tenant_surrogate"}
    assert all(forbidden.isdisjoint(row) for row in rows)


def test_public_row_ids_not_derived_from_hostname(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    ids = [row["public_row_id"] for row in read_jsonl(public_jsonl)]

    assert all("acme" not in row_id.lower() for row_id in ids)
    assert all(row_id.startswith("row_") for row_id in ids)


def test_nonlinkability_audit_passes_for_duplicate_fixture(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    report = verify_release(private_csv, public_jsonl, audit_dir, _config(), min_k=1)

    private_linkage = report["nonlinkability"]["private_origin_linkage_checks"]
    assert private_linkage["status"] == "pass"
    assert private_linkage["raw_hostname_group_counts_released"] is False
    assert private_linkage["raw_hostname_group_existence_released"] is False
    assert report["nonlinkability"]["website_access_pattern_audit"]["status"] == "pass"


def test_duplicate_private_hostnames_have_coarsened_public_fingerprints(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    rows = read_jsonl(public_jsonl)

    assert {row["released_length_bucket"] for row in rows} == {"withheld"}
    assert {row["character_class_mask"] for row in rows} == {"withheld"}
    assert {row["obfuscation_family"] for row in rows} == {"none"}


def test_no_public_rows_sorted_by_private_time_tenant_or_hostname(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    observed_ids = [row["public_row_id"] for row in read_jsonl(public_jsonl)]
    original_ids = [public_row_id(f"dup-{i:03d}", _config()) for i in range(60)]

    assert observed_ids != original_ids
    assert observed_ids != sorted(observed_ids)


def test_sparse_public_time_source_label_combinations_fail_closed(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    report = verify_release(private_csv, public_jsonl, audit_dir, _config(), min_k=100)

    assert report["status"] == "fail"
    assert report["nonlinkability"]["access_pattern_checks"]["n_sparse_public_combinations"] > 0


def test_stage_manifests_are_written(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    stage_dir = audit_dir / "stage_manifests"
    manifests = sorted(stage_dir.glob("*.json"))

    assert len(manifests) >= 9
    assert all(json.loads(path.read_text(encoding="utf-8"))["status"] == "complete" for path in manifests)


def test_streaming_chunk_directory_mode_shuffles_and_preserves_schema(tmp_path: Path) -> None:
    input_dir = tmp_path / "chunks"
    input_dir.mkdir()
    all_original_ids = []
    for chunk in range(3):
        path = input_dir / f"chunk_{chunk:02d}.csv"
        rows = []
        for i in range(20):
            row_id = f"stream-{chunk:02d}-{i:03d}"
            all_original_ids.append(public_row_id(row_id, _config()))
            rows.append(
                {
                    "ROW_ID": row_id,
                    "CDB": "PRIVATE_TENANT",
                    "CONTENT": "streaming-duplicate.customer.internal",
                    "HOSTNAME": "streaming-duplicate.customer.internal",
                    "USERNAME": "",
                    "CREATED_TIME": "2025-08-01T00:00:00Z",
                    "RESOLVED_LABEL_BOTH_M": "B",
                    "DATASET_FAMILY": "dns_hostnames",
                    "CONTENT_TYPE": "HOSTNAME",
                }
            )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"

    manifest = anonymize_csv_files(sorted(input_dir.glob("*.csv")), public_jsonl, audit_dir, _config(), shuffle_buckets=8)
    rows = read_jsonl(public_jsonl)

    assert manifest["n_public_rows"] == 60
    assert len(rows) == 60
    assert [row["public_row_id"] for row in rows] != all_original_ids
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
        "ccd_outputs",
        "row_integrity_hash",
    }


def test_streaming_verifier_passes_chunk_directory_release(tmp_path: Path) -> None:
    input_dir = tmp_path / "chunks"
    input_dir.mkdir()
    paths = []
    for chunk in range(2):
        path = input_dir / f"chunk_{chunk:02d}.csv"
        paths.append(path)
        rows = []
        for i in range(30):
            rows.append(
                {
                    "ROW_ID": f"verify-stream-{chunk:02d}-{i:03d}",
                    "CDB": "PRIVATE_TENANT",
                    "CONTENT": "streaming-duplicate.customer.internal",
                    "HOSTNAME": "streaming-duplicate.customer.internal",
                    "USERNAME": "",
                    "CREATED_TIME": "2025-08-01T00:00:00Z",
                    "RESOLVED_LABEL_BOTH_M": "B",
                    "DATASET_FAMILY": "dns_hostnames",
                    "CONTENT_TYPE": "HOSTNAME",
                }
            )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"

    anonymize_csv_files(paths, public_jsonl, audit_dir, _config(), shuffle_buckets=8)
    report = verify_release_streaming(paths, public_jsonl, audit_dir, _config(), min_k=50, buckets=8)

    assert report["status"] == "pass"
    assert report["anonymization"]["public_expected_row_checks"]["status"] == "pass"
    assert report["nonlinkability"]["private_origin_linkage_checks"]["raw_hostname_group_counts_released"] is False


def test_streaming_verifier_fails_closed_on_label_change(tmp_path: Path) -> None:
    input_dir = tmp_path / "chunks"
    input_dir.mkdir()
    path = input_dir / "chunk_00.csv"
    rows = []
    for i in range(60):
        rows.append(
            {
                "ROW_ID": f"tamper-stream-{i:03d}",
                "CDB": "PRIVATE_TENANT",
                "CONTENT": "tamper-duplicate.customer.internal",
                "HOSTNAME": "tamper-duplicate.customer.internal",
                "USERNAME": "",
                "CREATED_TIME": "2025-08-01T00:00:00Z",
                "RESOLVED_LABEL_BOTH_M": "B",
                "DATASET_FAMILY": "dns_hostnames",
                "CONTENT_TYPE": "HOSTNAME",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"

    anonymize_csv_files([path], public_jsonl, audit_dir, _config(), shuffle_buckets=8)
    public_rows = read_jsonl(public_jsonl)
    public_rows[0]["label"] = "verified_executable_semantics"
    public_rows[0]["row_integrity_hash"] = "tampered"
    write_jsonl(public_jsonl, public_rows)
    report = verify_release_streaming([path], public_jsonl, audit_dir, _config(), min_k=1, buckets=8)

    assert report["status"] == "fail"
    assert report["anonymization"]["public_expected_row_checks"]["n_label_mismatches"] == 1
    assert report["anonymization"]["public_expected_row_checks"]["n_integrity_failures"] == 1


def test_canonicalization_computed_from_released_artifact() -> None:
    assert canonicalize_artifact(" ExAmPlE.TEST. ") == "example.test"
