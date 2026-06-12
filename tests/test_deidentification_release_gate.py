from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deidentification_release" / "scripts"))

from validate_release_gate import find_private_duplicate_audit_key_leaks, validate_release_gate  # noqa: E402


def test_release_gate_rejects_private_duplicate_group_audit_keys(tmp_path: Path) -> None:
    audit_dir = tmp_path / "audits"
    shutil.copytree(ROOT / "deidentification_release" / "data" / "audits", audit_dir)
    nonlink_path = audit_dir / "nonlinkability_audit_report.json"
    nonlink = json.loads(nonlink_path.read_text(encoding="utf-8"))
    nonlink["private_origin_linkage_checks"]["n_raw_hostname_duplicate_groups"] = 3
    nonlink_path.write_text(json.dumps(nonlink), encoding="utf-8")

    result = validate_release_gate(
        ROOT / "deidentification_release" / "data" / "release" / "hib_release.jsonl",
        audit_dir,
        count_rows=True,
    )

    assert result["status"] == "fail"
    assert any("private raw-hostname duplicate/group key" in failure for failure in result["failures"])


def test_private_duplicate_key_scan_allows_public_duplicate_counts_only() -> None:
    nonlink = {
        "public_uniqueness_checks": {
            "n_duplicate_released_artifact_values": 0,
            "n_duplicate_released_canonical_values": 0,
        },
        "private_origin_linkage_checks": {
            "raw_hostname_group_counts_released": False,
            "raw_hostname_group_existence_released": False,
            "raw_hostname_multiplicity_released": False,
        },
    }

    assert find_private_duplicate_audit_key_leaks(nonlink) == []
    nonlink["private_origin_linkage_checks"]["raw_hostname_group_count"] = 2

    assert find_private_duplicate_audit_key_leaks(nonlink) == [
        "private_origin_linkage_checks.raw_hostname_group_count"
    ]
