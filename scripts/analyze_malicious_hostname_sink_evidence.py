#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


USECOLS = [
    "CONTENT",
    "HOSTNAME",
    "RESOLVED_LABEL_BOTH_M",
    "DATASET_FAMILY",
    "GPT_5_5_IS_DNS_CMD_INJECTION",
    "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION",
]

EXECUTION_TOKEN_RE = re.compile(
    r"(?ix)"
    r"(?:^|[^a-z0-9_])"
    r"("
    r"sh|bash|dash|zsh|ksh|cmd|powershell|pwsh|"
    r"curl|wget|nc|netcat|ncat|socat|telnet|tftp|ftp|"
    r"python|python3|perl|ruby|php|lua|node|java|javac|gcc|cc|"
    r"eval|exec|system|passthru|popen|spawn|"
    r"cat|tac|head|tail|more|less|grep|awk|sed|cut|sort|uniq|"
    r"ls|dir|id|whoami|uname|hostname|pwd|env|printenv|"
    r"sleep|ping|nslookup|dig|host|traceroute|"
    r"rm|mv|cp|chmod|chown|touch|mkdir|mkfifo|"
    r"echo|printf|base64|xxd|od|dd|"
    r"where|whereis|which|certutil|bitsadmin|mshta|regsvr32|rundll32"
    r")"
    r"(?:$|[^a-z0-9_])"
)

SHELL_META_RE = re.compile(r"[$`;&|<>]")
SHELL_CONTROL_RE = re.compile(r"(`[^`]+`|\$\([^)]{1,200}\)|\$\{[^}]{1,200}\}|;|&&|&|\|\||\||<|>|\\n|%0a|%0d)", re.I)
QUOTE_BREAK_RE = re.compile(r"['\"]\s*(?:;|&&|&|\|\||\||>|<|`|\$\()", re.I)
ENCODED_SHELL_RE = re.compile(r"(%24%28|%60|%3b|%26%26|%7c%7c|%7c|%3e|%3c|\\x60|\\x24\\x28)", re.I)
LOG4SHELL_RE = re.compile(r"\$\{\s*jndi\s*:", re.I)
SQLI_RE = re.compile(
    r"(?i)(\bunion\b.{0,40}\bselect\b|\bor\b\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+|"
    r"\band\b\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+|"
    r"\b(?:or|and)\b\s+.{0,50}(?:=|<|>)|"
    r"\bselect\b.{0,120}\bfrom\b|\bfrom\s+dual\b|"
    r"\bconcat\s*\(|\bsleep\s*\(|\bpg_sleep\s*\(|\bbenchmark\s*\(|\bwaitfor\s+delay\b|"
    r"\bdbms_pipe\b|--|/\*)"
)
CODE_EVAL_RE = re.compile(r"(?i)(__import__|require\s*['\"]|gethostbyname|runtime@getruntime|processbuilder|subprocess|popen|exec\s*\(|system\s*\()")
TEMPLATE_EXEC_RE = re.compile(r"(?i)(\{\{.*(?:config|self|class|mro|subprocess|popen|system|exec).*\}\}|\$\{.*(?:script|exec|runtime|processbuilder).*\})")


@dataclass(frozen=True)
class SinkEvidence:
    has_evidence: bool
    category: str
    rationale: str


def classify_sink_evidence(hostname: str) -> SinkEvidence:
    text = hostname.strip()
    lower = text.lower()

    if LOG4SHELL_RE.search(text):
        return SinkEvidence(
            True,
            "jndi_lookup_execution_sink",
            "JNDI/Log4Shell lookup syntax can trigger network lookup and code-loading behavior in vulnerable logging/expression sinks.",
        )

    if re.search(r"`[^`]+`", text) or re.search(r"\$\([^)]{1,200}\)", text):
        matches = re.findall(r"`([^`]+)`|\$\(([^)]{1,200})\)", text)
        inner = " ".join(part for match in matches for part in match if part)
        if EXECUTION_TOKEN_RE.search(inner) or re.search(r"[;&|<>]", inner):
            return SinkEvidence(
                True,
                "shell_command_substitution",
                "Backtick or $() command substitution contains command tokens or shell control operators.",
            )
        return SinkEvidence(
            True,
            "shell_command_substitution",
            "Backtick or $() command substitution would be evaluated by shell-like hostname sinks even when the inner token is environment-specific.",
        )

    if QUOTE_BREAK_RE.search(text) and (EXECUTION_TOKEN_RE.search(text) or SHELL_CONTROL_RE.search(text)):
        return SinkEvidence(
            True,
            "quote_breaking_shell_control",
            "Quote-breaking syntax plus shell control characters can alter a shell command line or force command failure.",
        )

    if ENCODED_SHELL_RE.search(text) and EXECUTION_TOKEN_RE.search(text):
        return SinkEvidence(
            True,
            "encoded_shell_control",
            "URL/hex encoded shell metacharacters accompany executable command tokens.",
        )

    if re.search(r"(?:^|[;&|])\s*(?:/bin/)?(?:sh|bash|dash|zsh)\b", lower):
        return SinkEvidence(
            True,
            "explicit_shell_invocation",
            "Explicit shell interpreter invocation appears in the hostname payload.",
        )

    if SHELL_CONTROL_RE.search(text) and EXECUTION_TOKEN_RE.search(text):
        return SinkEvidence(
            True,
            "shell_control_with_command_token",
            "Shell metacharacters/control operators accompany recognizable command tokens.",
        )

    if re.search(r"[;&|<>]", text) and not EXECUTION_TOKEN_RE.search(text):
        return SinkEvidence(
            True,
            "shell_execution_failure_evidence",
            "Shell metacharacters without a complete command still provide evidence of command-line alteration or execution failure in shell sinks.",
        )

    if TEMPLATE_EXEC_RE.search(text):
        return SinkEvidence(
            True,
            "template_expression_execution_sink",
            "Template/expression syntax references execution-capable objects or APIs.",
        )

    if CODE_EVAL_RE.search(text):
        return SinkEvidence(
            True,
            "code_expression_execution_sink",
            "Code/expression syntax references import, runtime execution, subprocess, or outbound lookup APIs.",
        )

    if SQLI_RE.search(text):
        return SinkEvidence(
            True,
            "query_execution_failure_or_delay_sink",
            "SQL-control syntax can alter database query execution or induce execution delay/failure if used in a query sink.",
        )

    if SHELL_META_RE.search(text):
        return SinkEvidence(
            False,
            "weak_metacharacter_only",
            "Contains shell-like metacharacters but no strong evidence that a sink would execute or fail rather than treat it as data.",
        )

    return SinkEvidence(False, "no_sink_execution_evidence", "No command, expression, query, or shell-control evidence was detected.")


def iter_chunk_paths(root: Path) -> Iterable[Path]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for chunk in manifest["datasets"]["dns_hostnames"]["chunks"]:
        yield root / chunk["path"]


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.4f}%" if d else "n/a"


def fmt_int(n: int) -> str:
    return f"{n:,}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def load_unique_malicious_hostnames(root: Path) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    total_rows = 0
    for index, path in enumerate(iter_chunk_paths(root), start=1):
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("DATASET_FAMILY") != "dns_hostnames":
                    continue
                if row.get("RESOLVED_LABEL_BOTH_M") != "M":
                    continue
                hostname = row.get("CONTENT") or row.get("HOSTNAME") or ""
                counts[hostname] += 1
                total_rows += 1
        if index % 100 == 0:
            print(f"scanned_dns_chunks={index} malicious_rows={total_rows} unique={len(counts)}", flush=True)
    return counts, total_rows


def analyze(root: Path) -> dict[str, Any]:
    unique_counts, total_rows = load_unique_malicious_hostnames(root)
    category_counts_unique: Counter[str] = Counter()
    category_counts_rows: Counter[str] = Counter()
    evidence_unique = 0
    evidence_rows = 0
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for hostname, row_count in unique_counts.items():
        evidence = classify_sink_evidence(hostname)
        category_counts_unique[evidence.category] += 1
        category_counts_rows[evidence.category] += row_count
        if evidence.has_evidence:
            evidence_unique += 1
            evidence_rows += row_count
        if len(samples[evidence.category]) < 10:
            samples[evidence.category].append(
                {"hostname": hostname, "row_count": row_count, "rationale": evidence.rationale}
            )

    return {
        "scope": {
            "family": "dns_hostnames",
            "malicious_definition": "RESOLVED_LABEL_BOTH_M == 'M' (both GPT-5.5 and Claude Opus 4.8 classified as malicious)",
            "dedupe_key": "exact CONTENT/HOSTNAME string",
        },
        "totals": {
            "malicious_rows": total_rows,
            "unique_malicious_hostnames": len(unique_counts),
            "sink_evidence_unique": evidence_unique,
            "sink_evidence_rows": evidence_rows,
            "sink_evidence_unique_share": evidence_unique / len(unique_counts) if unique_counts else None,
            "sink_evidence_row_share": evidence_rows / total_rows if total_rows else None,
            "no_sink_evidence_unique": len(unique_counts) - evidence_unique,
            "no_sink_evidence_rows": total_rows - evidence_rows,
        },
        "category_counts_unique": dict(category_counts_unique),
        "category_counts_rows": dict(category_counts_rows),
        "samples": dict(samples),
    }


def render_markdown_section(result: dict[str, Any]) -> str:
    totals = result["totals"]
    unique_total = totals["unique_malicious_hostnames"]
    row_total = totals["malicious_rows"]
    unique_counts = Counter(result["category_counts_unique"])
    row_counts = Counter(result["category_counts_rows"])

    lines: list[str] = []
    lines.append("## Sink Execution Evidence In Both-Model Malicious DNS Hostnames")
    lines.append("")
    lines.append(
        "Scope: DNS-family hostnames where `RESOLVED_LABEL_BOTH_M == M`, meaning both GPT-5.5 and Claude Opus 4.8 classified the hostname as malicious. "
        "Hostnames were deduplicated by exact `CONTENT`/`HOSTNAME` string before semantic classification."
    )
    lines.append("")
    lines.append(
        "Sink evidence was marked when the hostname contains concrete syntax that could cause unintended command execution, expression/JNDI/query execution behavior, "
        "or command/query execution failure if placed into a matching vulnerable sink. This is an evidence screen, not proof that the source environment actually evaluated the hostname."
    )
    lines.append("")
    lines.append(table(
        ["Metric", "Count", "Share"],
        [
            ["Malicious DNS rows", fmt_int(row_total), "100.0000%"],
            ["Unique malicious DNS hostnames", fmt_int(unique_total), "100.0000%"],
            ["Unique hostnames with sink evidence", fmt_int(totals["sink_evidence_unique"]), pct(totals["sink_evidence_unique"], unique_total)],
            ["Rows represented by hostnames with sink evidence", fmt_int(totals["sink_evidence_rows"]), pct(totals["sink_evidence_rows"], row_total)],
            ["Unique hostnames without sink evidence", fmt_int(totals["no_sink_evidence_unique"]), pct(totals["no_sink_evidence_unique"], unique_total)],
            ["Rows represented by hostnames without sink evidence", fmt_int(totals["no_sink_evidence_rows"]), pct(totals["no_sink_evidence_rows"], row_total)],
        ],
    ))
    lines.append("")
    lines.append("Category breakdown:")
    lines.append("")
    categories = sorted(unique_counts, key=lambda key: (-unique_counts[key], key))
    lines.append(table(
        ["Category", "Unique hostnames", "Unique share", "Rows", "Row share"],
        [[cat, fmt_int(unique_counts[cat]), pct(unique_counts[cat], unique_total), fmt_int(row_counts[cat]), pct(row_counts[cat], row_total)] for cat in categories],
    ))
    lines.append("")
    lines.append("Representative examples by category:")
    lines.append("")
    for cat in categories:
        examples = result["samples"].get(cat, [])[:5]
        if not examples:
            continue
        lines.append(f"### {cat}")
        lines.append("")
        lines.append(table(["Hostname", "Rows", "Rationale"], [[e["hostname"], fmt_int(e["row_count"]), e["rationale"]] for e in examples]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def replace_or_append_section(markdown_path: Path, section: str) -> None:
    heading = "## Sink Execution Evidence In Both-Model Malicious DNS Hostnames"
    text = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else ""
    if heading in text:
        before, rest = text.split(heading, 1)
        next_heading = re.search(r"\n## [^\n]+", rest)
        if next_heading:
            updated = before + section + rest[next_heading.start() :]
        else:
            updated = before.rstrip() + "\n\n" + section
    else:
        updated = text.rstrip() + "\n\n" + section
    markdown_path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sink-execution evidence among both-model malicious DNS hostnames.")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark")))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--update-profile-md", type=Path, default=None)
    args = parser.parse_args()

    result = analyze(args.root)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    section = render_markdown_section(result)
    args.out_md.write_text(section, encoding="utf-8")
    if args.update_profile_md is not None:
        replace_or_append_section(args.update_profile_md, section)
    print(json.dumps(result["totals"], indent=2, sort_keys=True))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    if args.update_profile_md is not None:
        print(f"Updated {args.update_profile_md}")


if __name__ == "__main__":
    main()
