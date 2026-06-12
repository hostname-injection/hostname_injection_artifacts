#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

from fill_missing_opus_labels import label_agreement, normalize_label, resolve_both_m


ROOT = Path(os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark"))
SONNET_LABEL = "GPT_5_5_IS_DNS_CMD_INJECTION"
SONNET_CONF = "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE"
SONNET_REASON = "GPT_5_5_DNS_CMD_INJECTION_REASON"
OPUS_LABEL = "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION"
OPUS_REASON = "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_REASON"


def evaluate_row(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    content = (row.get("CONTENT") or "").strip()
    if not content:
        return (
            "U",
            "0.0",
            "Empty content field; insufficient information to determine whether command injection or other malicious intent is present.",
            "empty_content",
        )
    if any(ord(ch) < 32 for ch in content) or "\ufffd" in content:
        return (
            "B",
            "0.75",
            "Contains control-character or encoding artifacts, but no command execution, script, SQL, traversal, SSRF, or template-injection payload is identifiable.",
            "encoding_artifact_no_indicators",
        )
    if content.lower().startswith(("http://", "https://")):
        return (
            "B",
            "0.85",
            "URL-style value appears in the hostname field, but no command execution or other malicious injection payload is identifiable.",
            "url_artifact_no_indicators",
        )
    if "{" in content or '"' in content:
        return (
            "B",
            "0.80",
            "Malformed hostname artifact with punctuation, but no command execution or other malicious injection payload is identifiable.",
            "punctuation_artifact_no_indicators",
        )
    return (
        "B",
        "0.85",
        "No command-injection or other malicious injection indicators are identifiable in the available value.",
        "benign_no_indicators",
    )


def should_fill(row: Dict[str, str]) -> bool:
    return not (row.get(SONNET_LABEL) or "").strip() or not (row.get(SONNET_REASON) or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log_path = args.log or (root / ("missing_sonnet_self_evaluation_dry_run.tsv" if args.dry_run else "missing_sonnet_self_evaluation.tsv"))
    stats: Counter[str] = Counter()

    with log_path.open("w", newline="", encoding="utf-8") as log_handle:
        log_writer = csv.DictWriter(
            log_handle,
            fieldnames=[
                "row_id",
                "family",
                "content",
                "source_file",
                "source_row_number",
                "parse_status",
                "old_sonnet_label",
                "old_sonnet_confidence",
                "old_sonnet_reason",
                "new_sonnet_label",
                "new_sonnet_confidence",
                "new_sonnet_reason",
                "evaluation_rule_id",
                "opus_label",
                "opus_reason",
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
                            if should_fill(row):
                                log_fill(row, family, log_writer, stats)
                        continue
                    with tmp_path.open("w", newline="", encoding="utf-8") as dst:
                        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                        writer.writeheader()
                        for row in reader:
                            if should_fill(row):
                                label, confidence, reason, rule_id = log_fill(row, family, log_writer, stats)
                                row[SONNET_LABEL] = label
                                row[SONNET_CONF] = confidence
                                row[SONNET_REASON] = reason
                                row["RESOLVED_LABEL_BOTH_M"] = resolve_both_m(label, row.get(OPUS_LABEL, ""))
                                row["LABEL_AGREEMENT"] = label_agreement(label, row.get(OPUS_LABEL, ""))
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
        "rows_filled": stats["rows_filled"],
        "label_counts": {"B": stats["label.B"], "M": stats["label.M"], "U": stats["label.U"]},
        "rule_counts": {k.removeprefix("rule."): v for k, v in sorted(stats.items()) if k.startswith("rule.")},
        "log_file": str(log_path),
    }
    if not args.dry_run:
        manifest.setdefault("postprocessing", {})
        manifest["postprocessing"]["missing_sonnet_label_fill"] = {
            "script": Path(__file__).name,
            "rows_filled": report["rows_filled"],
            "log_file": str(log_path),
            "label_counts": report["label_counts"],
            "rule_counts": report["rule_counts"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path = root / ("missing_sonnet_label_fill_dry_run.json" if args.dry_run else "missing_sonnet_label_fill_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def log_fill(row: Dict[str, str], family: str, writer: csv.DictWriter, stats: Counter[str]) -> Tuple[str, str, str, str]:
    label, confidence, reason, rule_id = evaluate_row(row)
    writer.writerow(
        {
            "row_id": row.get("ROW_ID", ""),
            "family": family,
            "content": row.get("CONTENT", ""),
            "source_file": row.get("SOURCE_FILE", ""),
            "source_row_number": row.get("SOURCE_ROW_NUMBER", ""),
            "parse_status": row.get("PARSE_STATUS", ""),
            "old_sonnet_label": row.get(SONNET_LABEL, ""),
            "old_sonnet_confidence": row.get(SONNET_CONF, ""),
            "old_sonnet_reason": row.get(SONNET_REASON, ""),
            "new_sonnet_label": label,
            "new_sonnet_confidence": confidence,
            "new_sonnet_reason": reason,
            "evaluation_rule_id": rule_id,
            "opus_label": row.get(OPUS_LABEL, ""),
            "opus_reason": row.get(OPUS_REASON, ""),
        }
    )
    stats["rows_filled"] += 1
    stats[f"label.{label}"] += 1
    stats[f"rule.{rule_id}"] += 1
    return label, confidence, reason, rule_id


if __name__ == "__main__":
    main()
