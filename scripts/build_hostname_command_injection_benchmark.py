#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


ROOT = Path(os.environ.get("HIB_SOURCE_ROOT", "data/private"))
USER_LOGIN_SRC = ROOT / "hostname_injection_dataset" / "hostname_injection_benchmark" / "user_logins"
DNS_SRC = ROOT / "hostname_injection_dataset_2"
OUT = ROOT / "HostnameCommandInjectionBenchmark"

START = datetime(2025, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
END = datetime(2026, 2, 6, 23, 59, 59, tzinfo=timezone.utc)

GPT_5_5_LABEL = "GPT_5_5_IS_DNS_CMD_INJECTION"
GPT_5_5_CONF = "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE"
GPT_5_5_REASON = "GPT_5_5_DNS_CMD_INJECTION_REASON"
OPUS_LABEL = "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION"
OPUS_CONF = "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE"
OPUS_REASON = "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_REASON"
SOURCE_GPT_5_5_LABEL = "CLAUDE_SONNET_4_5_IS_DNS_CMD_INJECTION"
SOURCE_GPT_5_5_CONF = "CLAUDE_SONNET_4_5_DNS_CMD_INJECTION_CONFIDENCE"
SOURCE_GPT_5_5_REASON = "CLAUDE_SONNET_4_5_DNS_CMD_INJECTION_REASON"
SOURCE_OPUS_LABEL = "CLAUDE_OPUS_4_5_IS_DNS_CMD_INJECTION"
SOURCE_OPUS_CONF = "CLAUDE_OPUS_4_5_DNS_CMD_INJECTION_CONFIDENCE"
SOURCE_OPUS_REASON = "CLAUDE_OPUS_4_5_DNS_CMD_INJECTION_REASON"

FIELDNAMES = [
    "ROW_ID",
    "CDB",
    "USERNAME",
    "HOSTNAME",
    "OS",
    "MID",
    "IP_ADDR",
    "CREATED_TIME",
    "ORIGINAL_CREATED_TIME",
    "SUCCESSFUL_LOGIN",
    "LOGIN_PORT",
    "ERROR_CODE",
    "SUCCESSFUL_QUERY",
    GPT_5_5_LABEL,
    GPT_5_5_CONF,
    GPT_5_5_REASON,
    OPUS_LABEL,
    OPUS_CONF,
    OPUS_REASON,
    "RESOLVED_LABEL_BOTH_M",
    "LABEL_AGREEMENT",
    "DATASET_FAMILY",
    "CONTENT_TYPE",
    "CONTENT",
    "SOURCE_FILE",
    "SOURCE_ROW_NUMBER",
    "PARSE_STATUS",
    "month",
    "year",
]

COMMON_KEYS = [
    "CDB",
    "USERNAME",
    "HOSTNAME",
    "OS",
    "MID",
    "IP_ADDR",
    "SUCCESSFUL_LOGIN",
    "LOGIN_PORT",
    "ERROR_CODE",
    "SUCCESSFUL_QUERY",
    GPT_5_5_LABEL,
    GPT_5_5_CONF,
    GPT_5_5_REASON,
    OPUS_LABEL,
    OPUS_CONF,
    OPUS_REASON,
]

SOURCE_TO_OUTPUT_KEYS = {
    SOURCE_GPT_5_5_LABEL: GPT_5_5_LABEL,
    SOURCE_GPT_5_5_CONF: GPT_5_5_CONF,
    SOURCE_GPT_5_5_REASON: GPT_5_5_REASON,
    SOURCE_OPUS_LABEL: OPUS_LABEL,
    SOURCE_OPUS_CONF: OPUS_CONF,
    SOURCE_OPUS_REASON: OPUS_REASON,
}


def norm(value: object) -> str:
    return "" if value is None else str(value)


def normalize_label(value: str) -> str:
    text = norm(value).strip().upper()
    if text.startswith("B"):
        return "B"
    if text.startswith("M"):
        return "M"
    return "U"


def resolved_both_m(sonnet: str, opus: str) -> str:
    s = normalize_label(sonnet)
    o = normalize_label(opus)
    if s == "M" and o == "M":
        return "M"
    if s == "B" and o == "B":
        return "B"
    return ""


def label_agreement(sonnet: str, opus: str) -> str:
    s = normalize_label(sonnet)
    o = normalize_label(opus)
    if s in {"B", "M"} and s == o:
        return "agree"
    if s in {"B", "M"} and o in {"B", "M"} and s != o:
        return "conflict"
    return "unknown"


def stable_key(dataset_family: str, content_type: str, content: str, original_time: str) -> str:
    raw = "\x1f".join([dataset_family, content_type, content, original_time])
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()


def fallback_key(dataset_family: str, path: Path, row_number: int) -> str:
    raw = "\x1f".join([dataset_family, source_rel(path), str(row_number)])
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()


def source_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _looks_like_cdb(value: str) -> bool:
    text = norm(value)
    return "_CDB_" in text or text.startswith(("PRODN_CDB_", "EUPRODN_CDB_", "APRODUS", "AUPROD"))


def _parse_physical_csv_line(line: str) -> List[str]:
    try:
        return next(csv.reader([line]))
    except Exception:
        return [line.rstrip("\r\n")]


def _repair_overwide_row(fields: List[str], expected: int) -> List[str]:
    if len(fields) <= expected:
        return fields + [""] * (expected - len(fields))
    tail_count = expected - 2
    content = ",".join(fields[1 : len(fields) - tail_count])
    return [fields[0], content] + fields[-tail_count:]


def _repair_missing_cdb_row(fields: List[str], expected: int) -> List[str]:
    repaired = [""] + fields
    return repaired[:expected] + [""] * max(expected - len(repaired), 0)


def iter_robust_source_csv(path: Path) -> Iterator[Tuple[int, Dict[str, str], str]]:
    """Yield logical source rows while preserving attack payload newlines.

    In both source families, the potentially malicious text is column 2:
    USERNAME for user logins and HOSTNAME for DNS. Several DNS payloads contain
    literal newlines and commas that were not CSV-quoted in the source files.
    Standard DictReader treats the continuation line as a new row, shifting
    columns. This parser reconstructs those rows by treating non-CDB physical
    lines as continuation payload text until the remaining tail columns appear.
    """
    with path.open("r", newline="", errors="ignore") as handle:
        header_line = handle.readline()
        if not header_line:
            return
        fieldnames = _parse_physical_csv_line(header_line)
        expected = len(fieldnames)
        tail_count = expected - 2
        logical_row = 0
        pending_cdb: Optional[str] = None
        pending_content_parts: List[str] = []
        pending_start_line = 0

        def make_row(fields: List[str]) -> Dict[str, str]:
            row = {name: "" for name in fieldnames}
            for name, value in zip(fieldnames, fields):
                row[name] = norm(value)
            return row

        def emit(fields: List[str], start_line: int, status: str) -> Tuple[int, Dict[str, str], str]:
            nonlocal logical_row
            logical_row += 1
            return logical_row, make_row(fields), status

        for physical_line, line in enumerate(handle, start=2):
            fields = _parse_physical_csv_line(line)
            if not fields:
                continue

            if pending_cdb is None:
                if _looks_like_cdb(fields[0]):
                    if len(fields) == expected:
                        yield emit(fields, physical_line, "ok")
                    elif len(fields) > expected:
                        yield emit(
                            _repair_overwide_row(fields, expected),
                            physical_line,
                            "repaired_overwide_content",
                        )
                    else:
                        pending_cdb = fields[0]
                        pending_content_parts = [",".join(fields[1:])]
                        pending_start_line = physical_line
                    continue

                if len(fields) >= expected - 1:
                    yield emit(
                        _repair_missing_cdb_row(fields, expected),
                        physical_line,
                        "repaired_missing_cdb",
                    )
                    continue

                yield emit(
                    _repair_overwide_row(fields, expected),
                    physical_line,
                    "unexpected_non_cdb_start",
                )
                continue

            if len(fields) >= tail_count + 1:
                content_fields = fields[: len(fields) - tail_count]
                tail = fields[-tail_count:]
                content = "\n".join(
                    part for part in pending_content_parts + [",".join(content_fields)] if part
                )
                yield emit(
                    [pending_cdb, content] + tail,
                    pending_start_line,
                    "repaired_multiline_content",
                )
                pending_cdb = None
                pending_content_parts = []
                pending_start_line = 0
            else:
                pending_content_parts.append(",".join(fields))

        if pending_cdb is not None:
            fields = [pending_cdb, "\n".join(pending_content_parts)]
            yield emit(
                fields + [""] * (expected - len(fields)),
                pending_start_line,
                "unterminated_multiline_content",
            )


def canonical_payload(
    *,
    dataset_family: str,
    content_type: str,
    content_key: str,
    row: Dict[str, str],
    path: Path,
    row_number: int,
    source_parse_status: str,
) -> Optional[Tuple[str, Dict[str, str]]]:
    content = norm(row.get(content_key))
    original_time = norm(row.get("CREATED_TIME"))
    out = {name: "" for name in FIELDNAMES}
    for key in COMMON_KEYS:
        out[key] = norm(row.get(key))
    for source_key, output_key in SOURCE_TO_OUTPUT_KEYS.items():
        if not out[output_key]:
            out[output_key] = norm(row.get(source_key))
    out["DATASET_FAMILY"] = dataset_family
    out["CONTENT_TYPE"] = content_type
    out["CONTENT"] = content
    out["ORIGINAL_CREATED_TIME"] = original_time
    out["SOURCE_FILE"] = source_rel(path)
    out["SOURCE_ROW_NUMBER"] = str(row_number)
    if content and original_time:
        out["PARSE_STATUS"] = source_parse_status
        key = stable_key(dataset_family, content_type, content, original_time)
    else:
        out["PARSE_STATUS"] = "missing_content_or_original_time"
        key = fallback_key(dataset_family, path, row_number)
    out["RESOLVED_LABEL_BOTH_M"] = resolved_both_m(out[GPT_5_5_LABEL], out[OPUS_LABEL])
    out["LABEL_AGREEMENT"] = label_agreement(out[GPT_5_5_LABEL], out[OPUS_LABEL])
    return key, out


def emit_raw_tsv(
    *,
    raw_path: Path,
    paths: Iterable[Path],
    dataset_family: str,
    content_type: str,
    content_key: str,
) -> Dict[str, int]:
    stats = {"source_rows": 0, "rows_with_missing_content_or_original_time": 0}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8", newline="") as out:
        for path in paths:
            file_rows = 0
            file_kept = 0
            for row_number, row, source_parse_status in iter_robust_source_csv(path):
                stats["source_rows"] += 1
                file_rows += 1
                item = canonical_payload(
                    dataset_family=dataset_family,
                    content_type=content_type,
                    content_key=content_key,
                    row=row,
                    path=path,
                    row_number=row_number,
                    source_parse_status=source_parse_status,
                )
                key, payload = item
                if payload["PARSE_STATUS"] != "ok":
                    if payload["PARSE_STATUS"] == "missing_content_or_original_time":
                        stats["rows_with_missing_content_or_original_time"] += 1
                    stats[f"source_parse_status.{payload['PARSE_STATUS']}"] = (
                        stats.get(f"source_parse_status.{payload['PARSE_STATUS']}", 0) + 1
                    )
                out.write(key)
                out.write("\t")
                out.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
                out.write("\n")
                file_kept += 1
            print(f"staged {path}: rows={file_rows} kept={file_kept}", flush=True)
    return stats


def sort_tsv(raw_path: Path, sorted_path: Path, tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["TMPDIR"] = str(tmp_dir)
    cmd = [
        "sort",
        "-T",
        str(tmp_dir),
        "-S",
        "50%",
        "-t",
        "\t",
        "-k1,1",
        str(raw_path),
        "-o",
        str(sorted_path),
    ]
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def sorted_groups(sorted_path: Path) -> Iterator[List[Dict[str, str]]]:
    current_key = None
    group: List[Dict[str, str]] = []
    with sorted_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            key, payload_json = line.rstrip("\n").split("\t", 1)
            if current_key is None:
                current_key = key
            if key != current_key:
                yield group
                group = []
                current_key = key
            group.append(json.loads(payload_json))
    if group:
        yield group


def merge_group(group: List[Dict[str, str]]) -> Dict[str, str]:
    merged = dict(group[0])
    sources = []
    for payload in group:
        source = norm(payload.get("SOURCE_FILE"))
        if source and source not in sources:
            sources.append(source)
        for col in COMMON_KEYS:
            value = norm(payload.get(col))
            if value and not norm(merged.get(col)):
                merged[col] = value
    merged["SOURCE_FILE"] = ";".join(sources)
    merged["RESOLVED_LABEL_BOTH_M"] = resolved_both_m(merged[GPT_5_5_LABEL], merged[OPUS_LABEL])
    merged["LABEL_AGREEMENT"] = label_agreement(merged[GPT_5_5_LABEL], merged[OPUS_LABEL])
    return merged


def iso_at(index: int, total: int) -> str:
    if total <= 1:
        dt = START
    else:
        span = (END - START).total_seconds()
        dt = datetime.fromtimestamp(START.timestamp() + span * index / (total - 1), tz=timezone.utc)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def clean_chunks(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.glob("*.csv"):
        child.unlink()


def count_unique(sorted_path: Path) -> Tuple[int, int]:
    raw = 0
    unique = 0
    for group in sorted_groups(sorted_path):
        raw += len(group)
        unique += 1
    return raw, unique


def write_chunks(
    *,
    out_root: Path,
    family: str,
    content_type: str,
    sorted_path: Path,
    chunk_rows: int,
    unique_rows: int,
) -> Dict[str, object]:
    chunk_dir = out_root / "data" / family / "chunks"
    clean_chunks(chunk_dir)
    chunks: List[Dict[str, object]] = []
    writer = None
    handle = None
    chunk_index = -1
    chunk_count = 0
    row_index = 0
    for group in sorted_groups(sorted_path):
        if row_index % chunk_rows == 0:
            if handle:
                handle.close()
                chunks[-1]["rows"] = chunk_count
                chunks[-1]["bytes"] = os.path.getsize(out_root / chunks[-1]["path"])
            chunk_index += 1
            chunk_count = 0
            chunk_path = chunk_dir / f"{family}_{chunk_index:05d}.csv"
            handle = chunk_path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            chunks.append({"path": str(chunk_path.relative_to(out_root)), "rows": 0, "bytes": 0})
        payload = merge_group(group)
        created = iso_at(row_index, unique_rows)
        payload["ROW_ID"] = f"{family}-{row_index + 1:012d}"
        payload["CREATED_TIME"] = created
        dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        payload["month"] = f"{dt.month:02d}"
        payload["year"] = str(dt.year)
        writer.writerow({name: payload.get(name, "") for name in FIELDNAMES})
        row_index += 1
        chunk_count += 1
    if handle:
        handle.close()
        chunks[-1]["rows"] = chunk_count
        chunks[-1]["bytes"] = os.path.getsize(out_root / chunks[-1]["path"])
    return {
        "content_type": content_type,
        "rows": unique_rows,
        "chunk_rows_target": chunk_rows,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def write_docs(out_root: Path, manifests: Dict[str, Dict[str, object]], stats: Dict[str, int]) -> None:
    manifest = {
        "name": "HostnameCommandInjectionBenchmark",
        "version": "1.0",
        "created_by": Path(__file__).name,
        "source_roots": [str(USER_LOGIN_SRC), str(DNS_SRC)],
        "time_correction": {
            "created_time": "Uniformly reassigned over the known sampling interval.",
            "start_utc": START.isoformat().replace("+00:00", "Z"),
            "end_utc": END.isoformat().replace("+00:00", "Z"),
            "original_timestamp_column": "ORIGINAL_CREATED_TIME",
        },
        "deduplication": {
            "rule": "Rows are unique within each dataset family by DATASET_FAMILY, CONTENT_TYPE, CONTENT, and ORIGINAL_CREATED_TIME.",
            "fallback_rule": "Rows missing CONTENT or ORIGINAL_CREATED_TIME are preserved and keyed by source file plus source row number.",
            "user_logins_content": "USERNAME",
            "dns_hostnames_content": "HOSTNAME",
        },
        "source_csv_repairs": {
            "rule": "The builder repairs unquoted newlines and unquoted commas in the second source column, where USERNAME/HOSTNAME payloads live.",
            "parse_status_column": "PARSE_STATUS records ok, repaired_multiline_content, repaired_overwide_content, unexpected_non_cdb_start, or unterminated_multiline_content.",
        },
        "partitioning": "No train/test/eval split is included. Split downstream in training code.",
        "schema": FIELDNAMES,
        "datasets": manifests,
        "build_stats": stats,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_root / "README.md").write_text(
        "\n".join(
            [
                "# HostnameCommandInjectionBenchmark",
                "",
                "Canonical, deduplicated benchmark data for hostname command-injection experiments.",
                "",
                "## Layout",
                "",
                "- `data/user_logins/chunks/*.csv`: user-login records. `CONTENT` is copied from `USERNAME`.",
                "- `data/dns_hostnames/chunks/*.csv`: DNS hostname records. `CONTENT` is copied from `HOSTNAME`.",
                "- `manifest.json`: machine-readable schema, chunk list, source paths, row counts, and build stats.",
                "",
                "The first columns are the original operational fields (`CDB`, `USERNAME`, `HOSTNAME`, `MID`, IP and login/query fields, labels, reasons). "
                "Benchmark helper fields (`DATASET_FAMILY`, `CONTENT_TYPE`, `CONTENT`, `SOURCE_FILE`, `SOURCE_ROW_NUMBER`, `PARSE_STATUS`) follow the label columns.",
                "",
                "The dataset is intentionally not split into train, validation, or test subsets. Training scripts should perform any split they need.",
                "",
                "## PyTorch Loading",
                "",
                "This repository includes `ccd.benchmark_dataset.HostnameCommandInjectionBenchmarkDataset`, a map-style PyTorch-compatible dataset "
                "for these chunks. It supports selecting `user_logins`, `dns_hostnames`, or both; resolving labels from GPT 5.5, Claude Opus 4.8, or both; "
                "dropping or retaining unknown labels; returning explanations; and optionally returning full row metadata.",
                "",
                "Example:",
                "",
                "```python",
                "from ccd.benchmark_dataset import HostnameCommandInjectionBenchmarkDataset",
                "",
                "dataset = HostnameCommandInjectionBenchmarkDataset(",
                "    'HostnameCommandInjectionBenchmark',",
                "    family='both',",
                "    label_method='both-disagree-unknown',",
                "    drop_unknown=True,",
                "    include_explanations=False,",
                ")",
                "```",
                "",
                "## Time Correction",
                "",
                "The source timestamps were known to be incorrectly recorded. `CREATED_TIME` has been reassigned uniformly from "
                f"`{START.isoformat().replace('+00:00', 'Z')}` through `{END.isoformat().replace('+00:00', 'Z')}`. "
                "`ORIGINAL_CREATED_TIME` preserves the source timestamp used for deduplication.",
                "",
                "## Deduplication",
                "",
                "Rows are deduplicated within each dataset family by `(DATASET_FAMILY, CONTENT_TYPE, CONTENT, ORIGINAL_CREATED_TIME)`. "
                "For DNS rows split across GPT-5.5-only and Opus-4.8-only files, labels are merged into one row when the content-time key matches.",
                "Rows with an empty content field or missing original timestamp are preserved with `PARSE_STATUS=missing_content_or_original_time` "
                "and keyed by source file plus source row number.",
                "",
                "## Source CSV Repairs",
                "",
                "Some DNS payloads contain literal newlines or commas in the source `HOSTNAME` column without CSV quoting. "
                "The builder reconstructs those logical records before writing benchmark CSVs. Repaired rows keep the payload in `HOSTNAME` and `CONTENT`, "
                "and `PARSE_STATUS` records the repair type. A small number of rows with source lines missing the CDB prefix are preserved with blank `CDB` "
                "and `PARSE_STATUS=repaired_missing_cdb`. Use a real CSV parser when loading chunks because some valid fields contain embedded newlines.",
                "",
                "## Labels",
                "",
                "`RESOLVED_LABEL_BOTH_M` follows the existing training default: `M` only when both model labels are malicious, "
                "`B` only when both are benign, and blank otherwise. `LABEL_AGREEMENT` records `agree`, `conflict`, or `unknown`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_family(
    *,
    out_root: Path,
    build_dir: Path,
    family: str,
    content_type: str,
    content_key: str,
    paths: Iterable[Path],
    chunk_rows: int,
) -> Tuple[Dict[str, object], Dict[str, int]]:
    raw_path = build_dir / f"{family}.raw.tsv"
    sorted_path = build_dir / f"{family}.sorted.tsv"
    stats = emit_raw_tsv(
        raw_path=raw_path,
        paths=paths,
        dataset_family=family,
        content_type=content_type,
        content_key=content_key,
    )
    sort_tsv(raw_path, sorted_path, build_dir / "sort_tmp")
    raw_rows, unique_rows = count_unique(sorted_path)
    stats["raw_rows_after_staging"] = raw_rows
    stats["unique_rows"] = unique_rows
    stats["duplicates_by_content_time"] = raw_rows - unique_rows
    manifest = write_chunks(
        out_root=out_root,
        family=family,
        content_type=content_type,
        sorted_path=sorted_path,
        chunk_rows=chunk_rows,
        unique_rows=unique_rows,
    )
    return manifest, {f"{family}.{k}": v for k, v in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--keep-build-files", action="store_true")
    parser.add_argument(
        "--families",
        choices=["both", "user_logins", "dns_hostnames"],
        default="both",
        help="Build all families or only one family. Single-family builds reuse the existing manifest for the other family.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    build_dir = args.output / "_build"
    if args.reset:
        shutil.rmtree(build_dir, ignore_errors=True)
        shutil.rmtree(args.output / "data", ignore_errors=True)
        for name in ["manifest.json", "README.md"]:
            path = args.output / name
            if path.exists():
                path.unlink()
    build_dir.mkdir(parents=True, exist_ok=True)

    existing_manifest = {}
    manifest_path = args.output / "manifest.json"
    if args.families != "both" and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifests: Dict[str, Dict[str, object]] = dict(existing_manifest.get("datasets", {}))
    all_stats: Dict[str, int] = dict(existing_manifest.get("build_stats", {}))
    if args.families in {"both", "user_logins"}:
        user_manifest, user_stats = build_family(
            out_root=args.output,
            build_dir=build_dir,
            family="user_logins",
            content_type="USERNAME",
            content_key="USERNAME",
            paths=sorted(USER_LOGIN_SRC.glob("*.csv")),
            chunk_rows=args.chunk_rows,
        )
        manifests["user_logins"] = user_manifest
        all_stats = {k: v for k, v in all_stats.items() if not k.startswith("user_logins.")}
        all_stats.update(user_stats)

    if args.families in {"both", "dns_hostnames"}:
        dns_manifest, dns_stats = build_family(
            out_root=args.output,
            build_dir=build_dir,
            family="dns_hostnames",
            content_type="HOSTNAME",
            content_key="HOSTNAME",
            paths=sorted(DNS_SRC.glob("**/*.csv")),
            chunk_rows=args.chunk_rows,
        )
        manifests["dns_hostnames"] = dns_manifest
        all_stats = {k: v for k, v in all_stats.items() if not k.startswith("dns_hostnames.")}
        all_stats.update(dns_stats)

    write_docs(args.output, manifests, all_stats)
    if not args.keep_build_files:
        shutil.rmtree(build_dir, ignore_errors=True)
    print(f"wrote benchmark to {args.output}", flush=True)


if __name__ == "__main__":
    main()
