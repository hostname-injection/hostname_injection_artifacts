#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Tuple


ROOT = Path(os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark"))
SONNET_LABEL = "GPT_5_5_IS_DNS_CMD_INJECTION"
SONNET_REASON = "GPT_5_5_DNS_CMD_INJECTION_REASON"
OPUS_LABEL = "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION"
OPUS_CONF = "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE"
OPUS_REASON = "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_REASON"


MALICIOUS_PATTERNS: Tuple[Tuple[str, re.Pattern[str], str, str], ...] = (
    (
        "jndi_log4shell",
        re.compile(r"\$\{\s*jndi\s*:|jndi\s*:\s*(ldap|rmi|dns|iiop|http)", re.I),
        "M",
        "JNDI/Log4Shell-style lookup expression indicates an injection payload capable of remote code execution or callback behavior.",
    ),
    (
        "shell_command_substitution",
        re.compile(r"(`[^`]+`|\$\([^)]{1,240}\))", re.I | re.S),
        "M",
        "Shell command-substitution syntax is present in the hostname, which is a command-injection indicator.",
    ),
    (
        "shell_execution_command",
        re.compile(
            r"""
            (?:[;&|`]|\$\(|\$\{|[({\[]|/bin/)
            (?:/bin/(?:ba|z|c|k)?sh|bash|sh|zsh|cmd(?:\.exe)?|powershell|pwsh|curl|wget|nc|ncat|netcat|telnet|python(?:3)?|perl|ruby|php|lua|node|openssl|busybox)
            (?:$|[;&|`$({}\[\]\s./:-])
            """,
            re.I | re.X,
        ),
        "M",
        "Hostname contains shell metacharacter context with a command/interpreter token, consistent with command injection.",
    ),
    (
        "shell_probe_command",
        re.compile(
            r"""
            (?:[;&|`]|\$\(|\$\{|[({\[]|\s)
            (?:whoami|id|uname|hostname|sleep|ping|nslookup|dig|cat|ls|pwd|env|printenv|chmod|chown|rm|touch)
            (?:$|[;&|`$({}\[\]\s./:-])
            """,
            re.I | re.X,
        ),
        "M",
        "Hostname contains shell-style separators or substitutions around a common command/probe token, consistent with command injection.",
    ),
    (
        "sql_injection",
        re.compile(
            r"\b(sql\s*injection|sqli|union\s+select|pg_sleep|dbms_pipe|waitfor\s+delay|benchmark\s*\(|sleep\s*\(\s*\d+\)|(?:'\s*or\s+\d+=\d+)|(?:\"\s*or\s+\d+=\d+)|(?:--\s*(?:$|[.])))",
            re.I,
        ),
        "M",
        "SQL-injection syntax or timing probe is present in the hostname, indicating malicious injection intent.",
    ),
    (
        "xss",
        re.compile(r"(<\s*script\b|%3c\s*script|javascript\s*:|onerror\s*=|onload\s*=|<\s*svg\b|%3csvg|alert\s*\(|document\.cookie|document\.domain)", re.I),
        "M",
        "XSS/script-injection payload syntax is present in the hostname, indicating malicious injection intent.",
    ),
    (
        "template_injection",
        re.compile(r"(\{\{\s*[^}]{1,80}\s*\}\}|\$\{\s*[^}]{1,120}\}|<%=?\s*[^%]{1,120}%>|\[\[\s*[^]]{1,80}\s*\]\])", re.I),
        "M",
        "Template-expression syntax is present in the hostname, indicating a template-injection probe.",
    ),
    (
        "path_traversal",
        re.compile(r"(\.\./|\.\.\\|%2e%2e|%252e%252e|/etc/passwd|/etc/shadow|/proc/self|c:\\windows\\win\.ini|web-inf/web\.xml)", re.I),
        "M",
        "Path-traversal or sensitive-file probe syntax is present in the hostname, indicating malicious intent.",
    ),
    (
        "ssrf_metadata",
        re.compile(r"(169\.254\.169\.254|metadata\.azure|instance-data|latest/meta-data|ssrf)", re.I),
        "M",
        "Cloud metadata or SSRF probe indicator is present in the hostname, indicating malicious intent.",
    ),
    (
        "xxe_xml",
        re.compile(r"(<!entity|<!doctype|%3c!entity|%3c!doctype|xml external entity|(?<![a-z0-9])xxe(?![a-z0-9]))", re.I),
        "M",
        "XML external entity or XML markup probe syntax is present in the hostname, indicating malicious intent.",
    ),
)

SUSPICIOUS_PATTERNS: Tuple[Tuple[str, re.Pattern[str], str, str], ...] = (
    (
        "unclosed_shell_substitution",
        re.compile(r"(\$\(|`|\$\{)", re.I),
        "U",
        "Contains incomplete variable or substitution syntax, but the available hostname is not enough to determine malicious command execution intent.",
    ),
    (
        "url_in_hostname",
        re.compile(r"^https?://|://|%3a%2f%2f", re.I),
        "U",
        "Contains URL syntax inside a hostname field, which is unusual, but no clear malicious injection pattern is present.",
    ),
    (
        "odd_metacharacters",
        re.compile(r"[;&|<>\"'{}()[\]\\]", re.I),
        "U",
        "Contains unusual metacharacters for a hostname, but no clear command, script, SQL, traversal, SSRF, or template-injection payload is identifiable.",
    ),
)

BENIGN_INFRA_RE = re.compile(
    r"""
    (^|[.])
    (amazonaws\.com|googleapis\.com|gcr\.io|pkg\.dev|azure\.com|windows\.net|cloudfront\.net|akamaihd\.net|fastly\.net|cloudflare\.com|docker\.com|docker\.io|k8s\.io|kubernetes\.io|svc\.cluster\.local|cluster\.local|compute\.internal|google\.internal|lacework\.net)
    ($|[.])
    """,
    re.I | re.X,
)

HOSTNAME_SHAPE_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


def normalize_label(value: str) -> str:
    text = (value or "").strip().upper()
    if text.startswith("M"):
        return "M"
    if text.startswith("B"):
        return "B"
    return "U"


def resolve_both_m(sonnet: str, opus: str) -> str:
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


def evaluate_content(content: str, family: str) -> Tuple[str, str, str, str]:
    text = (content or "").strip()
    if not text:
        return (
            "U",
            "0.0",
            "Empty content field; insufficient information to determine whether command injection or other malicious intent is present.",
            "empty_content",
        )

    for rule_id, pattern, label, reason in MALICIOUS_PATTERNS:
        if pattern.search(text):
            return label, "0.90", reason, rule_id

    for rule_id, pattern, label, reason in SUSPICIOUS_PATTERNS:
        if pattern.search(text):
            return label, "0.45", reason, rule_id

    if family == "dns_hostnames":
        if BENIGN_INFRA_RE.search(text):
            return (
                "B",
                "0.95",
                "Hostname matches a normal cloud, infrastructure, Kubernetes, or service-domain pattern and contains no injection indicators.",
                "benign_infra_hostname",
            )
        if HOSTNAME_SHAPE_RE.fullmatch(text):
            return (
                "B",
                "0.95",
                "Hostname uses ordinary DNS-compatible characters and contains no command, script, SQL, traversal, SSRF, or template-injection indicators.",
                "benign_dns_shape",
            )

    return (
        "B",
        "0.80",
        "No command-injection or other malicious injection indicators are identifiable in the available value.",
        "benign_no_indicators",
    )


def should_fill(row: Mapping[str, str]) -> bool:
    return not (row.get(OPUS_LABEL) or "").strip() or not (row.get(OPUS_REASON) or "").strip()


def fill_missing(root: Path, *, dry_run: bool, log_path: Path) -> Dict[str, object]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats: Counter[str] = Counter()

    log_handle = log_path.open("w", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(
        log_handle,
        fieldnames=[
            "row_id",
            "family",
            "content",
            "source_file",
            "source_row_number",
            "parse_status",
            "old_opus_label",
            "old_opus_confidence",
            "old_opus_reason",
            "new_opus_label",
            "new_opus_confidence",
            "new_opus_reason",
            "evaluation_rule_id",
            "sonnet_label",
            "sonnet_reason",
        ],
        delimiter="\t",
    )
    log_writer.writeheader()

    try:
        for family, dataset in manifest["datasets"].items():
            for chunk in dataset["chunks"]:
                path = root / chunk["path"]
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                changed = False
                if dry_run:
                    src_handle = path.open("r", newline="", encoding="utf-8")
                    dst_handle = None
                else:
                    src_handle = path.open("r", newline="", encoding="utf-8")
                    dst_handle = tmp_path.open("w", newline="", encoding="utf-8")
                with src_handle as src:
                    reader = csv.DictReader(src)
                    writer = None
                    if dst_handle is not None:
                        with dst_handle as dst:
                            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                            writer.writeheader()
                            for row in reader:
                                if should_fill(row):
                                    apply_fill(row, family, log_writer, stats)
                                    changed = True
                                writer.writerow(row)
                    else:
                        for row in reader:
                            if should_fill(row):
                                apply_fill(row, family, log_writer, stats)
                                changed = True
                if dry_run:
                    continue
                if changed:
                    os.replace(tmp_path, path)
                    chunk["bytes"] = path.stat().st_size
                else:
                    tmp_path.unlink(missing_ok=True)
        if not dry_run:
            manifest.setdefault("postprocessing", {})
            manifest["postprocessing"]["missing_opus_label_fill"] = {
                "script": Path(__file__).name,
                "rows_filled": stats["rows_filled"],
                "log_file": str(log_path),
                "label_counts": {
                    "B": stats["label.B"],
                    "M": stats["label.M"],
                    "U": stats["label.U"],
                },
                "rule_counts": {k.removeprefix("rule."): v for k, v in sorted(stats.items()) if k.startswith("rule.")},
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finally:
        log_handle.close()

    return {
        "dry_run": dry_run,
        "rows_filled": stats["rows_filled"],
        "label_counts": {"B": stats["label.B"], "M": stats["label.M"], "U": stats["label.U"]},
        "family_counts": {k.removeprefix("family."): v for k, v in sorted(stats.items()) if k.startswith("family.")},
        "rule_counts": {k.removeprefix("rule."): v for k, v in sorted(stats.items()) if k.startswith("rule.")},
        "log_file": str(log_path),
    }


def apply_fill(row: Dict[str, str], family: str, log_writer: csv.DictWriter, stats: Counter[str]) -> None:
    old_label = row.get(OPUS_LABEL, "")
    old_conf = row.get(OPUS_CONF, "")
    old_reason = row.get(OPUS_REASON, "")
    label, confidence, reason, rule_id = evaluate_content(row.get("CONTENT", ""), family)
    log_writer.writerow(
        {
            "row_id": row.get("ROW_ID", ""),
            "family": family,
            "content": row.get("CONTENT", ""),
            "source_file": row.get("SOURCE_FILE", ""),
            "source_row_number": row.get("SOURCE_ROW_NUMBER", ""),
            "parse_status": row.get("PARSE_STATUS", ""),
            "old_opus_label": old_label,
            "old_opus_confidence": old_conf,
            "old_opus_reason": old_reason,
            "new_opus_label": label,
            "new_opus_confidence": confidence,
            "new_opus_reason": reason,
            "evaluation_rule_id": rule_id,
            "sonnet_label": row.get(SONNET_LABEL, ""),
            "sonnet_reason": row.get(SONNET_REASON, ""),
        }
    )
    row[OPUS_LABEL] = label
    row[OPUS_CONF] = confidence
    row[OPUS_REASON] = reason
    row["RESOLVED_LABEL_BOTH_M"] = resolve_both_m(row.get(SONNET_LABEL, ""), label)
    row["LABEL_AGREEMENT"] = label_agreement(row.get(SONNET_LABEL, ""), label)
    stats["rows_filled"] += 1
    stats[f"label.{label}"] += 1
    stats[f"family.{family}"] += 1
    stats[f"rule.{rule_id}"] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    log_path = args.log or (args.root / ("missing_opus_self_evaluation_dry_run.tsv" if args.dry_run else "missing_opus_self_evaluation.tsv"))
    result = fill_missing(args.root, dry_run=args.dry_run, log_path=log_path)
    report_path = args.root / ("missing_opus_label_fill_dry_run.json" if args.dry_run else "missing_opus_label_fill_report.json")
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
