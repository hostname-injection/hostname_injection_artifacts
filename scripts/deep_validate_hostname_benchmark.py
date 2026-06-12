#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple


DEFAULT_ROOT = Path(os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark"))
GPT_LABEL = "GPT_5_5_IS_DNS_CMD_INJECTION"
GPT_CONF = "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE"
GPT_REASON = "GPT_5_5_DNS_CMD_INJECTION_REASON"
OPUS_LABEL = "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION"
OPUS_CONF = "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE"
OPUS_REASON = "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_REASON"

MALICIOUS_TEXT_RE = re.compile(
    r"""
    (\$\(|`|\$\{\s*jndi|jndi\s*:|/bin/(?:sh|bash|zsh|cmd)|\b(?:curl|wget|nc|ncat|netcat|whoami|uname|nslookup|dig)\b)
    |(pg_sleep|dbms_pipe|waitfor\s+delay|union\s+select|'\s*or\s+\d+=\d+|"\s*or\s+\d+=\d+)
    |(<\s*script|javascript\s*:|onerror\s*=|onload\s*=|<\s*svg|alert\s*\(|document\.(?:cookie|domain))
    |(\.\./|%2e%2e|%252e%252e|/etc/passwd|web-inf/web\.xml)
    |(169\.254\.169\.254|metadata\.azure|latest/meta-data|ssrf)
    |(<!entity|<!doctype|xml external entity|(?<![a-z0-9])xxe(?![a-z0-9]))
    """,
    re.I | re.X,
)
BENIGN_REASON_RE = re.compile(r"\b(valid|legitimate|standard|normal|benign|no .*injection|no .*indicators|no malicious|not malicious)\b", re.I)
MALICIOUS_REASON_RE = re.compile(r"\b(malicious|command injection|shell|remote code execution|sql injection|xss|path traversal|ssrf|xxe|log4shell|jndi|payload|attack|probe)\b", re.I)


def normalize_label(value: str) -> str:
    text = (value or "").strip().upper()
    if text in {"B", "M", "U"}:
        return text
    return ""


def resolve_both_m(gpt: str, opus: str) -> str:
    if gpt == "M" and opus == "M":
        return "M"
    if gpt == "B" and opus == "B":
        return "B"
    return ""


def label_agreement(gpt: str, opus: str) -> str:
    if gpt in {"B", "M"} and gpt == opus:
        return "agree"
    if gpt in {"B", "M"} and opus in {"B", "M"} and gpt != opus:
        return "conflict"
    return "unknown"


def add_example(examples: Dict[str, list], key: str, row: Dict[str, str], detail: str, max_examples: int) -> None:
    bucket = examples.setdefault(key, [])
    if len(bucket) < max_examples:
        bucket.append(
            {
                "row_id": row.get("ROW_ID", ""),
                "family": row.get("DATASET_FAMILY", ""),
                "content": row.get("CONTENT", "")[:240],
                "detail": detail[:500],
            }
        )


def validate_row(row: Dict[str, str], counters: Counter[str], examples: Dict[str, list], max_examples: int) -> None:
    family = row.get("DATASET_FAMILY", "")
    content = row.get("CONTENT", "")
    counters["rows"] += 1
    if family == "dns_hostnames":
        if content != row.get("HOSTNAME", ""):
            counters["content_hostname_mismatch"] += 1
            add_example(examples, "content_hostname_mismatch", row, "CONTENT != HOSTNAME", max_examples)
        if row.get("USERNAME", ""):
            counters["dns_username_nonempty"] += 1
            add_example(examples, "dns_username_nonempty", row, "DNS row has USERNAME", max_examples)
    elif family == "user_logins":
        if content != row.get("USERNAME", ""):
            counters["content_username_mismatch"] += 1
            add_example(examples, "content_username_mismatch", row, "CONTENT != USERNAME", max_examples)
    else:
        counters["bad_family"] += 1
        add_example(examples, "bad_family", row, family, max_examples)

    gpt = normalize_label(row.get(GPT_LABEL, ""))
    opus = normalize_label(row.get(OPUS_LABEL, ""))
    counters[f"gpt_label.{gpt or '<invalid>'}"] += 1
    counters[f"opus_label.{opus or '<invalid>'}"] += 1
    for model, label, conf_col, reason_col in [
        ("gpt_5_5", gpt, GPT_CONF, GPT_REASON),
        ("opus_4_8", opus, OPUS_CONF, OPUS_REASON),
    ]:
        reason = (row.get(reason_col) or "").strip()
        conf = (row.get(conf_col) or "").strip()
        if label not in {"B", "M", "U"}:
            counters[f"{model}.invalid_label"] += 1
            add_example(examples, f"{model}.invalid_label", row, row.get(reason_col, ""), max_examples)
        if not reason:
            counters[f"{model}.blank_reason"] += 1
            add_example(examples, f"{model}.blank_reason", row, "", max_examples)
        try:
            parsed = float(conf)
            if parsed < 0 or parsed > 1:
                raise ValueError
        except Exception:
            counters[f"{model}.bad_confidence"] += 1
            add_example(examples, f"{model}.bad_confidence", row, conf, max_examples)
        if label == "M" and BENIGN_REASON_RE.search(reason) and not MALICIOUS_REASON_RE.search(reason):
            counters[f"{model}.malicious_label_benign_reason"] += 1
            add_example(examples, f"{model}.malicious_label_benign_reason", row, reason, max_examples)
        if label == "B" and MALICIOUS_TEXT_RE.search(content) and not BENIGN_REASON_RE.search(reason):
            counters[f"{model}.benign_label_suspicious_content_reason_not_explanatory"] += 1
            add_example(examples, f"{model}.benign_label_suspicious_content_reason_not_explanatory", row, reason, max_examples)

    expected_resolved = resolve_both_m(gpt, opus)
    if row.get("RESOLVED_LABEL_BOTH_M", "") != expected_resolved:
        counters["resolved_label_mismatch"] += 1
        add_example(examples, "resolved_label_mismatch", row, f"expected {expected_resolved}", max_examples)
    expected_agreement = label_agreement(gpt, opus)
    if row.get("LABEL_AGREEMENT", "") != expected_agreement:
        counters["label_agreement_mismatch"] += 1
        add_example(examples, "label_agreement_mismatch", row, f"expected {expected_agreement}", max_examples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=25)
    args = parser.parse_args()

    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    counters: Counter[str] = Counter()
    examples: Dict[str, list] = {}
    for family, dataset in manifest["datasets"].items():
        print(f"deep-validating {family}", flush=True)
        for chunk in dataset["chunks"]:
            path = args.root / chunk["path"]
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != manifest["schema"]:
                    counters["header_mismatch"] += 1
                    continue
                for row in reader:
                    validate_row(row, counters, examples, args.max_examples)

    hard_error_keys = [
        key
        for key in counters
        if key.endswith("mismatch")
        or key.endswith("blank_reason")
        or key.endswith("bad_confidence")
        or key.endswith("invalid_label")
        or key in {"bad_family", "content_hostname_mismatch", "content_username_mismatch", "header_mismatch"}
    ]
    report = {
        "overall_pass": not hard_error_keys,
        "hard_error_keys": hard_error_keys,
        "counters": dict(sorted(counters.items())),
        "examples": examples,
    }
    output = args.output or (args.root / "deep_quality_report.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"overall_pass": report["overall_pass"], "output": str(output), "hard_error_keys": hard_error_keys}, indent=2), flush=True)
    if hard_error_keys:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
