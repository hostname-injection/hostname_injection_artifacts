#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_ROOT = Path(os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark"))


def stable_key(dataset_family: str, content_type: str, content: str, original_time: str) -> str:
    raw = "\x1f".join([dataset_family, content_type, content, original_time])
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()


def fallback_key(dataset_family: str, source_file: str, source_row_number: str) -> str:
    raw = "\x1f".join([dataset_family, source_file, source_row_number])
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()


def parse_created(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iter_chunks(root: Path, chunks: List[Dict[str, object]]) -> Iterable[Tuple[Dict[str, object], Path]]:
    for chunk in chunks:
        yield chunk, root / str(chunk["path"])


def audit_family(root: Path, family: str, dataset: Dict[str, object], schema: List[str]) -> Dict[str, object]:
    chunk_count = 0
    rows = 0
    data_rows_by_manifest = 0
    missing_files: List[str] = []
    header_errors: List[str] = []
    byte_mismatches: List[str] = []
    row_count_mismatches: List[str] = []
    row_id_errors: List[str] = []
    duplicate_key_examples: List[Dict[str, str]] = []
    sort_order_errors: List[Dict[str, str]] = []
    bad_family_examples: List[Dict[str, str]] = []
    column_semantic_errors: List[Dict[str, str]] = []
    timestamp_errors: List[Dict[str, str]] = []
    parse_status_counts: Counter[str] = Counter()
    label_agreement_counts: Counter[str] = Counter()
    resolved_label_counts: Counter[str] = Counter()
    gpt_5_5_counts: Counter[str] = Counter()
    opus_4_8_counts: Counter[str] = Counter()
    first_created = None
    last_created = None
    first_row_id = None
    last_row_id = None
    previous_key = None
    previous_created = None
    min_chunk_rows = None
    max_chunk_rows = 0

    chunks = list(dataset["chunks"])
    for chunk, path in iter_chunks(root, chunks):
        chunk_count += 1
        data_rows_by_manifest += int(chunk["rows"])
        if not path.exists():
            missing_files.append(str(path))
            continue
        if path.stat().st_size != int(chunk["bytes"]):
            byte_mismatches.append(str(path))

        actual_rows = 0
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != schema:
                header_errors.append(str(path))
                continue
            for row in reader:
                rows += 1
                actual_rows += 1
                expected_row_id = f"{family}-{rows:012d}"
                row_id = row.get("ROW_ID", "")
                if row_id != expected_row_id and len(row_id_errors) < 20:
                    row_id_errors.append(f"{path.name}:{actual_rows}: expected {expected_row_id}, saw {row_id}")
                if first_row_id is None:
                    first_row_id = row_id
                last_row_id = row_id

                if row.get("DATASET_FAMILY") != family and len(bad_family_examples) < 20:
                    bad_family_examples.append({"row_id": row_id, "dataset_family": row.get("DATASET_FAMILY", "")})

                if len(column_semantic_errors) < 50:
                    if family == "dns_hostnames" and row.get("CONTENT", "") != row.get("HOSTNAME", ""):
                        column_semantic_errors.append(
                            {
                                "row_id": row_id,
                                "field": "CONTENT/HOSTNAME",
                                "content": row.get("CONTENT", ""),
                                "hostname": row.get("HOSTNAME", ""),
                            }
                        )
                    if family == "user_logins" and row.get("CONTENT", "") != row.get("USERNAME", ""):
                        column_semantic_errors.append(
                            {
                                "row_id": row_id,
                                "field": "CONTENT/USERNAME",
                                "content": row.get("CONTENT", ""),
                                "username": row.get("USERNAME", ""),
                            }
                        )
                    cdb = row.get("CDB", "")
                    if cdb and "_CDB_" not in cdb and not cdb.startswith(("PRODN_CDB_", "EUPRODN_CDB_", "APRODUS", "AUPROD")):
                        column_semantic_errors.append({"row_id": row_id, "field": "CDB", "value": cdb})
                    for field in [
                        "GPT_5_5_IS_DNS_CMD_INJECTION",
                        "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION",
                        "RESOLVED_LABEL_BOTH_M",
                    ]:
                        value = row.get(field, "")
                        if value and value not in {"B", "M", "U"}:
                            column_semantic_errors.append({"row_id": row_id, "field": field, "value": value})

                parse_status = row.get("PARSE_STATUS", "")
                parse_status_counts[parse_status] += 1
                label_agreement_counts[row.get("LABEL_AGREEMENT", "")] += 1
                resolved_label_counts[row.get("RESOLVED_LABEL_BOTH_M", "")] += 1
                gpt_5_5_counts[row.get("GPT_5_5_IS_DNS_CMD_INJECTION", "")] += 1
                opus_4_8_counts[row.get("CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION", "")] += 1

                if parse_status == "missing_content_or_original_time":
                    key = fallback_key(family, row.get("SOURCE_FILE", ""), row.get("SOURCE_ROW_NUMBER", ""))
                else:
                    key = stable_key(
                        family,
                        row.get("CONTENT_TYPE", ""),
                        row.get("CONTENT", ""),
                        row.get("ORIGINAL_CREATED_TIME", ""),
                    )
                if previous_key is not None:
                    if key == previous_key and len(duplicate_key_examples) < 20:
                        duplicate_key_examples.append({"row_id": row_id, "key": key})
                    if key < previous_key and len(sort_order_errors) < 20:
                        sort_order_errors.append({"row_id": row_id, "previous_key": previous_key, "key": key})
                previous_key = key

                try:
                    created = parse_created(row.get("CREATED_TIME", ""))
                except Exception:
                    if len(timestamp_errors) < 20:
                        timestamp_errors.append({"row_id": row_id, "created_time": row.get("CREATED_TIME", "")})
                    continue
                if first_created is None:
                    first_created = row.get("CREATED_TIME", "")
                last_created = row.get("CREATED_TIME", "")
                if previous_created is not None and created <= previous_created and len(timestamp_errors) < 20:
                    timestamp_errors.append(
                        {
                            "row_id": row_id,
                            "created_time": row.get("CREATED_TIME", ""),
                            "previous_created_time": previous_created.isoformat(),
                        }
                    )
                previous_created = created

        if actual_rows != int(chunk["rows"]):
            row_count_mismatches.append(f"{path.name}: manifest {chunk['rows']} actual {actual_rows}")
        min_chunk_rows = actual_rows if min_chunk_rows is None else min(min_chunk_rows, actual_rows)
        max_chunk_rows = max(max_chunk_rows, actual_rows)

    return {
        "family": family,
        "manifest_rows": int(dataset["rows"]),
        "actual_rows": rows,
        "manifest_chunk_rows_sum": data_rows_by_manifest,
        "manifest_chunk_count": int(dataset["chunk_count"]),
        "actual_chunk_count": chunk_count,
        "min_chunk_rows": min_chunk_rows,
        "max_chunk_rows": max_chunk_rows,
        "first_row_id": first_row_id,
        "last_row_id": last_row_id,
        "first_created_time": first_created,
        "last_created_time": last_created,
        "missing_files": missing_files,
        "header_errors": header_errors,
        "byte_mismatches": byte_mismatches,
        "row_count_mismatches": row_count_mismatches,
        "row_id_errors": row_id_errors,
        "duplicate_dedupe_key_examples": duplicate_key_examples,
        "dedupe_key_sort_order_errors": sort_order_errors,
        "bad_family_examples": bad_family_examples,
        "column_semantic_errors": column_semantic_errors,
        "timestamp_errors": timestamp_errors,
        "parse_status_counts": dict(parse_status_counts),
        "label_agreement_counts": dict(label_agreement_counts),
        "resolved_label_both_m_counts": dict(resolved_label_counts),
        "gpt_5_5_label_counts": dict(gpt_5_5_counts),
        "opus_4_8_label_counts": dict(opus_4_8_counts),
        "passes": {
            "manifest_rows_match_actual_rows": rows == int(dataset["rows"]) == data_rows_by_manifest,
            "chunk_count_matches_manifest": chunk_count == int(dataset["chunk_count"]),
            "all_files_exist": not missing_files,
            "headers_match_schema": not header_errors,
            "file_sizes_match_manifest": not byte_mismatches,
            "chunk_row_counts_match_manifest": not row_count_mismatches,
            "row_ids_are_sequential_unique": not row_id_errors,
            "dedupe_keys_are_unique_in_sorted_output": not duplicate_key_examples and not sort_order_errors,
            "dataset_family_column_matches": not bad_family_examples,
            "core_columns_have_expected_semantics": not column_semantic_errors,
            "created_time_is_parseable_and_strictly_increasing": not timestamp_errors,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = list(manifest["schema"])
    report = {
        "benchmark_root": str(root),
        "manifest_path": str(manifest_path),
        "audit_started_note": "Full streaming audit over generated CSV chunks.",
        "build_artifacts_present": (root / "_build").exists(),
        "families": {},
    }
    for family, dataset in manifest["datasets"].items():
        print(f"auditing {family}", flush=True)
        report["families"][family] = audit_family(root, family, dataset, schema)

    report["overall_pass"] = (
        not report["build_artifacts_present"]
        and all(all(result["passes"].values()) for result in report["families"].values())
    )
    output = args.output or (root / "quality_report.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"overall_pass": report["overall_pass"], "output": str(output)}, indent=2), flush=True)
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
