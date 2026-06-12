#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict


DEFAULT_ROOT = Path(os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark"))
GPT_LABEL = "GPT_5_5_IS_DNS_CMD_INJECTION"
GPT_REASON = "GPT_5_5_DNS_CMD_INJECTION_REASON"
OPUS_LABEL = "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION"
OPUS_REASON = "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_REASON"

SKECHERS_SQLI_SERVICE_RE = re.compile(
    r"^skechers-(?:dev|prod)\.sqli\.cloud4\.incorta\.com(?:\.us-west-2\.compute\.internal)?$",
    re.I,
)
BENIGN_SQLI_REASON_RE = re.compile(
    r"(valid|standard|hostname|cloud|service|not injection|not sql|not malicious|part of subdomain|service identifier)",
    re.I,
)


def normalize_label(value: str) -> str:
    text = (value or "").strip().upper()
    if text.startswith("M"):
        return "M"
    if text.startswith("B"):
        return "B"
    return "U"


def resolve_both_m(gpt: str, opus: str) -> str:
    g = normalize_label(gpt)
    o = normalize_label(opus)
    if g == "M" and o == "M":
        return "M"
    if g == "B" and o == "B":
        return "B"
    return ""


def label_agreement(gpt: str, opus: str) -> str:
    g = normalize_label(gpt)
    o = normalize_label(opus)
    if g in {"B", "M"} and g == o:
        return "agree"
    if g in {"B", "M"} and o in {"B", "M"} and g != o:
        return "conflict"
    return "unknown"


def should_fix(content: str, label: str, reason: str) -> bool:
    return (
        normalize_label(label) == "M"
        and SKECHERS_SQLI_SERVICE_RE.fullmatch(content or "") is not None
        and BENIGN_SQLI_REASON_RE.search(reason or "") is not None
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log_path = root / ("sqli_service_label_corrections_dry_run.tsv" if args.dry_run else "sqli_service_label_corrections.tsv")
    stats: Counter[str] = Counter()

    with log_path.open("w", newline="", encoding="utf-8") as log_handle:
        log_writer = csv.DictWriter(
            log_handle,
            fieldnames=[
                "row_id",
                "family",
                "content",
                "model",
                "old_label",
                "new_label",
                "reason",
                "source_file",
                "source_row_number",
            ],
            delimiter="\t",
        )
        log_writer.writeheader()
        for family, dataset in manifest["datasets"].items():
            for chunk in dataset["chunks"]:
                path = root / chunk["path"]
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                changed = False
                with path.open("r", newline="", encoding="utf-8") as src:
                    reader = csv.DictReader(src)
                    if args.dry_run:
                        for row in reader:
                            changed |= maybe_fix(row, family, log_writer, stats, dry_run=True)
                        continue
                    with tmp_path.open("w", newline="", encoding="utf-8") as dst:
                        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                        writer.writeheader()
                        for row in reader:
                            if maybe_fix(row, family, log_writer, stats, dry_run=False):
                                row["RESOLVED_LABEL_BOTH_M"] = resolve_both_m(row.get(GPT_LABEL, ""), row.get(OPUS_LABEL, ""))
                                row["LABEL_AGREEMENT"] = label_agreement(row.get(GPT_LABEL, ""), row.get(OPUS_LABEL, ""))
                                changed = True
                            writer.writerow(row)
                if args.dry_run:
                    continue
                if changed:
                    os.replace(tmp_path, path)
                    chunk["bytes"] = path.stat().st_size
                else:
                    tmp_path.unlink(missing_ok=True)

    report = {
        "dry_run": args.dry_run,
        "assignments_corrected": stats["assignments_corrected"],
        "rows_with_corrections": stats["rows_with_corrections"],
        "model_counts": {k.removeprefix("model."): v for k, v in sorted(stats.items()) if k.startswith("model.")},
        "log_file": str(log_path),
    }
    if not args.dry_run:
        manifest.setdefault("postprocessing", {})
        manifest["postprocessing"]["sqli_service_label_corrections"] = {
            "script": Path(__file__).name,
            "assignments_corrected": report["assignments_corrected"],
            "rows_with_corrections": report["rows_with_corrections"],
            "model_counts": report["model_counts"],
            "log_file": str(log_path),
            "reason": "Corrected labels for skechers-*.sqli.cloud4.incorta.com hostnames where reasons explicitly describe sqli as a benign service/subdomain identifier.",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path = root / ("sqli_service_label_corrections_dry_run.json" if args.dry_run else "sqli_service_label_corrections_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def maybe_fix(row: Dict[str, str], family: str, writer: csv.DictWriter, stats: Counter[str], *, dry_run: bool) -> bool:
    content = row.get("CONTENT", "")
    row_changed = False
    assignments = 0
    for model, label_col, reason_col in [
        ("gpt_5_5", GPT_LABEL, GPT_REASON),
        ("opus_4_8", OPUS_LABEL, OPUS_REASON),
    ]:
        if should_fix(content, row.get(label_col, ""), row.get(reason_col, "")):
            writer.writerow(
                {
                    "row_id": row.get("ROW_ID", ""),
                    "family": family,
                    "content": content,
                    "model": model,
                    "old_label": row.get(label_col, ""),
                    "new_label": "B",
                    "reason": row.get(reason_col, ""),
                    "source_file": row.get("SOURCE_FILE", ""),
                    "source_row_number": row.get("SOURCE_ROW_NUMBER", ""),
                }
            )
            if not dry_run:
                row[label_col] = "B"
            stats["assignments_corrected"] += 1
            stats[f"model.{model}"] += 1
            assignments += 1
            row_changed = True
    if assignments:
        stats["rows_with_corrections"] += 1
    return row_changed


if __name__ == "__main__":
    main()
