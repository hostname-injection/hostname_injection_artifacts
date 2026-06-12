#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple


DEFAULT_ROOT = Path(os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark"))
GPT_LABEL = "GPT_5_5_IS_DNS_CMD_INJECTION"
GPT_CONF = "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE"
GPT_REASON = "GPT_5_5_DNS_CMD_INJECTION_REASON"
OPUS_LABEL = "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION"


CONSUL_TEMPLATE_RE = re.compile(
    r"^mesh-gateway\.service\.\$\(CONSUL_DATACENTER\)\.consul(?:\.|$)",
    re.I,
)
SHEIN_STATIC_TEMPLATE_RE = re.compile(
    r"^img\.ltwebstatic\.com\$\{recommendgoods_\d+_(?:text_)?img\d+\}$",
    re.I,
)
DEVICE_PIPE_RE = re.compile(
    r"""
    (
      ^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ '-]{1,40}\|(?:ASB|BC)-MBA(?:[.]|$)
      |^[A-Za-z0-9À-ÿ -]+-\|-Justaddsugar(?:[.]|$)
      |^[A-Za-z0-9À-ÿ -]+\s+\|\s+[A-Za-z0-9.-]+$
    )
    """,
    re.I | re.X,
)
LOCALHOST_PORT_RE = re.compile(r"^localhost\s*:\s*\d+(?:[.]|$)", re.I)
S3_INTERNAL_RE = re.compile(r"^[a-z0-9._-]+[.]s3[.][a-z0-9-]+[.]amazonaws[.]com[.]google[.]internal$", re.I)
S3_GLOBAL_INTERNAL_RE = re.compile(r"^[a-z0-9._-]+[.]s3[.]amazonaws[.]com[.]google[.]internal$", re.I)
METADATA_SEARCH_SUFFIX_RE = re.compile(r"^metadata[.]google[.]internal[.][a-z0-9.-]+$", re.I)
UNIX_DAEMON_PATH_RE = re.compile(r"^/usr/(?:sbin|libexec)/[A-Za-z0-9_.-]+$", re.I)
ENCODING_ARTIFACT_RE = re.compile(r"[\x00-\x08\x0b-\x1f\ufffd]")
SINGLE_DOLLAR_RE = re.compile(r"^\$$")

JNDI_RE = re.compile(r"\$\{\s*jndi\b|jndi\s*:", re.I)
SQL_INJECTION_RE = re.compile(
    r"waitfor\s+delay|pg_sleep\s*\(|dbms_pipe\.receive_message|union\s+select|benchmark\s*\(|sleep\s*\(\s*\d+\)|xor\s*\(\s*if\s*\(\s*now\(\)\s*=\s*sysdate\(\)",
    re.I,
)
CODE_INJECTION_RE = re.compile(
    r"gethostbyname\s*\(|require[\"']socket|response\.write\s*\(|(?:^|[.])bxss[.]me(?:[.]|$)|(?:^|[.])r87[.]me(?:[.]|$)|(?:^|[.])oastify[.]com(?:[.]|$)|interactsh",
    re.I,
)
RUBY_SLEEP_RE = re.compile(r"(?:^|['\"+])\s*sleep\s*\(\s*\d+[.]to_i\s*\)", re.I)
CHR_HEX_CHAIN_RE = re.compile(r"chr\s*\(\s*(?:hex\s*\(\s*['\"][0-9a-f]+['\"]\s*\)|\d+)\s*\)(?:\s*[+.]\s*chr\s*\(|.*[.]online[.]cmgroep[.]local)", re.I)
SHELL_COMMAND_RE = re.compile(
    r"""
    (
      (?:;|&&|\|\|)\s*(?:curl|wget|sleep|whoami|id|uname|nslookup|dig|powershell|cmd|bash|sh)(?:\b|[.(])
      |[|&]\s*(?:curl|wget|sleep|whoami|id|uname|nslookup|dig)(?:\b|[.(])
      |(?:;|&&|\|\|)\s*(?:cat|type)\s+
      |\$\([^)]*(?:curl|wget|sleep|whoami|id|uname|nslookup|dig|cat|type)[^)]*\)
      |`[^`]*(?:curl|wget|sleep|whoami|id|uname|nslookup|dig|cat|type)[^`]*`
    )
    """,
    re.I | re.X,
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


def classify(row: Dict[str, str]) -> Optional[Tuple[str, str, str, str]]:
    content = (row.get("CONTENT") or "").strip()
    family = row.get("DATASET_FAMILY", "")
    opus = normalize_label(row.get(OPUS_LABEL, ""))

    if family == "dns_hostnames" and CONSUL_TEMPLATE_RE.search(content):
        return (
            "B",
            "0.95",
            "Consul service-discovery hostname with a datacenter template inside a Kubernetes/cloud DNS suffix; no command payload is present.",
            "benign_consul_service_template",
        )
    if family == "dns_hostnames" and SHEIN_STATIC_TEMPLATE_RE.fullmatch(content):
        return (
            "B",
            "0.90",
            "SHEIN static-asset hostname with an application template placeholder; no command, SQL, script, traversal, or SSRF payload is present.",
            "benign_static_asset_template",
        )
    if family == "dns_hostnames" and LOCALHOST_PORT_RE.search(content):
        return (
            "B",
            "0.90",
            "Localhost plus port formatting is malformed for DNS but does not contain an injection payload.",
            "benign_localhost_port_artifact",
        )
    if family == "dns_hostnames" and (S3_INTERNAL_RE.fullmatch(content) or S3_GLOBAL_INTERNAL_RE.fullmatch(content)):
        return (
            "B",
            "0.85",
            "AWS S3 hostname with an internal DNS suffix; unusual routing artifact but no injection payload is present.",
            "benign_s3_internal_dns_artifact",
        )
    if family == "dns_hostnames" and METADATA_SEARCH_SUFFIX_RE.fullmatch(content):
        return (
            "B",
            "0.85",
            "Google metadata hostname with an added DNS search suffix; unusual routing artifact but no command, code, or SQL injection payload is present.",
            "benign_metadata_search_suffix_artifact",
        )
    if family == "dns_hostnames" and DEVICE_PIPE_RE.search(content):
        return (
            "B",
            "0.85",
            "Device or host naming pattern uses a pipe as a separator; no executable command or injection payload is present.",
            "benign_device_name_separator",
        )
    if family == "dns_hostnames" and UNIX_DAEMON_PATH_RE.fullmatch(content):
        return (
            "B",
            "0.85",
            "Unix daemon path appears to be misplaced data in the hostname field, not an injection payload.",
            "benign_unix_path_artifact",
        )
    if family == "user_logins" and SINGLE_DOLLAR_RE.fullmatch(content):
        return (
            "B",
            "0.85",
            "Single dollar-sign placeholder without a variable name, command, or substitution body is not an injection payload.",
            "benign_single_dollar_placeholder",
        )

    if JNDI_RE.search(content):
        return (
            "M",
            "0.90",
            "JNDI/Log4Shell-style lookup fragment indicates malicious exploit intent even if the payload is truncated.",
            "malicious_jndi_log4shell_intent",
        )
    if SQL_INJECTION_RE.search(content):
        return (
            "M",
            "0.90",
            "SQL-injection timing or database-function payload is present, indicating malicious injection intent.",
            "malicious_sql_injection_intent",
        )
    if RUBY_SLEEP_RE.search(content):
        return (
            "M",
            "0.85",
            "Ruby sleep() timing payload embedded with string concatenation indicates code-injection intent.",
            "malicious_ruby_sleep_timing_intent",
        )
    if CODE_INJECTION_RE.search(content):
        return (
            "M",
            "0.90",
            "Code/DNS-callback injection pattern is present, indicating malicious injection or exfiltration intent.",
            "malicious_code_callback_intent",
        )
    if CHR_HEX_CHAIN_RE.search(content):
        return (
            "M",
            "0.85",
            "chr()/hex() string-construction payload indicates code-injection intent even without a shell command token.",
            "malicious_chr_hex_code_construction_intent",
        )
    if SHELL_COMMAND_RE.search(content):
        return (
            "M",
            "0.90",
            "Shell command separator/substitution syntax is paired with a command token, indicating command-injection intent.",
            "malicious_shell_command_intent",
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--name", default="gpt_5_5_unknown_reclassification")
    args = parser.parse_args()

    root = args.root
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log_path = root / (f"{args.name}_dry_run.tsv" if args.dry_run else f"{args.name}.tsv")
    stats: Counter[str] = Counter()

    with log_path.open("w", newline="", encoding="utf-8") as log_handle:
        writer_log = csv.DictWriter(
            log_handle,
            fieldnames=[
                "row_id",
                "family",
                "content",
                "old_gpt_label",
                "old_gpt_confidence",
                "old_gpt_reason",
                "new_gpt_label",
                "new_gpt_confidence",
                "new_gpt_reason",
                "rule_id",
                "opus_label",
                "source_file",
                "source_row_number",
            ],
            delimiter="\t",
        )
        writer_log.writeheader()
        for family, dataset in manifest["datasets"].items():
            for chunk in dataset["chunks"]:
                path = root / chunk["path"]
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                changed = False
                with path.open("r", newline="", encoding="utf-8") as src:
                    reader = csv.DictReader(src)
                    if args.dry_run:
                        for row in reader:
                            maybe_log(row, family, writer_log, stats, dry_run=True)
                        continue
                    with tmp_path.open("w", newline="", encoding="utf-8") as dst:
                        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                        writer.writeheader()
                        for row in reader:
                            if maybe_log(row, family, writer_log, stats, dry_run=False):
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
        "rows_reclassified": stats["rows_reclassified"],
        "label_counts": {"B": stats["label.B"], "M": stats["label.M"]},
        "rule_counts": {k.removeprefix("rule."): v for k, v in sorted(stats.items()) if k.startswith("rule.")},
        "family_counts": {k.removeprefix("family."): v for k, v in sorted(stats.items()) if k.startswith("family.")},
        "log_file": str(log_path),
    }
    if not args.dry_run:
        manifest.setdefault("postprocessing", {})
        manifest["postprocessing"][args.name] = {
            "script": Path(__file__).name,
            "rows_reclassified": report["rows_reclassified"],
            "label_counts": report["label_counts"],
            "rule_counts": report["rule_counts"],
            "family_counts": report["family_counts"],
            "log_file": str(log_path),
            "research_notes": [
                "Consul DNS service lookups use service.service[...].<datacenter>.dc.<domain>-style names; Kubernetes services use service.namespace.svc.cluster.local-style names.",
                "metadata.google.internal is a documented Google Compute Engine metadata hostname; when it appears only as a hostname with an additional DNS suffix and no injection syntax, this pass treats it as a routing/search-suffix artifact rather than injection intent.",
                "Log4Shell/JNDI lookup strings and SQL timing/database-function payloads indicate malicious injection intent.",
                "Ruby sleep() timing payloads and chr()/hex() string-construction fragments indicate malicious code-injection intent.",
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_path = root / (f"{args.name}_dry_run.json" if args.dry_run else f"{args.name}_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def maybe_log(row: Dict[str, str], family: str, writer: csv.DictWriter, stats: Counter[str], *, dry_run: bool) -> bool:
    if normalize_label(row.get(GPT_LABEL, "")) != "U":
        return False
    result = classify(row)
    if result is None:
        return False
    label, confidence, reason, rule_id = result
    writer.writerow(
        {
            "row_id": row.get("ROW_ID", ""),
            "family": family,
            "content": row.get("CONTENT", ""),
            "old_gpt_label": row.get(GPT_LABEL, ""),
            "old_gpt_confidence": row.get(GPT_CONF, ""),
            "old_gpt_reason": row.get(GPT_REASON, ""),
            "new_gpt_label": label,
            "new_gpt_confidence": confidence,
            "new_gpt_reason": reason,
            "rule_id": rule_id,
            "opus_label": row.get(OPUS_LABEL, ""),
            "source_file": row.get("SOURCE_FILE", ""),
            "source_row_number": row.get("SOURCE_ROW_NUMBER", ""),
        }
    )
    stats["rows_reclassified"] += 1
    stats[f"label.{label}"] += 1
    stats[f"rule.{rule_id}"] += 1
    stats[f"family.{family}"] += 1
    if not dry_run:
        row[GPT_LABEL] = label
        row[GPT_CONF] = confidence
        row[GPT_REASON] = reason
        row["RESOLVED_LABEL_BOTH_M"] = resolve_both_m(label, row.get(OPUS_LABEL, ""))
        row["LABEL_AGREEMENT"] = label_agreement(label, row.get(OPUS_LABEL, ""))
    return True


if __name__ == "__main__":
    main()
