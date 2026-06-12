#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple


ROOT = Path(os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark"))
SONNET_LABEL = "GPT_5_5_IS_DNS_CMD_INJECTION"
SONNET_REASON = "GPT_5_5_DNS_CMD_INJECTION_REASON"
OPUS_LABEL = "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION"
OPUS_REASON = "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_REASON"


MALICIOUS_REASON_RE = re.compile(
    r"""
    (\bsql\s*injection\b|\bsqli\b|\boracle\s+sql\s+injection\b|\bmysql\s+sql\s+injection\b|\bpg_sleep\b|\bdbms_pipe\b|\bwaitfor\s+delay\b)
    |(\bxss\b|\bcross[- ]site\b|\bscript\s+tag\b|\bscript\s+injection\b).{0,80}\b(attempt|probe|test|vector|payload|injection)\b
    |\b(attempt|probe|test|vector|payload)\b.{0,80}(\bxss\b|\bcross[- ]site\b|\bscript\s+tag\b|\bscript\s+injection\b)
    |\b(path|directory)\s+traversal\b.{0,80}\b(attempt|pattern|payload|test|probe)\b
    |\b(template\s+injection|ssti|server[- ]side template)\b
    |\b(ssrf|server[- ]side request forgery)\b.{0,100}\b(attempt|probe|metadata|cloud metadata)\b
    |\b(metadata\.google\.internal|169\.254\.169\.254|metadata\.azure|instance-data)\b.{0,120}\b(ssrf|metadata)\b.{0,80}\b(attempt|probe|probing)\b
    |\b(open\s+redirect|xxe|xml external entity|deserialization|csrf)\b.{0,80}\b(attempt|payload|probe|test|vector|injection)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

BENIGN_CONTEXT_RE = re.compile(
    r"""
    \b(valid|legitimate|standard|normal)\b.{0,80}\b(credentials?\s+(service|api)|iamcredentials|credential\w*\.googleapis|grpc|srv record)\b
    |\b(no|not|without|lacks|absent)\b.{0,80}\b(sql injection|xss|ssrf|path traversal|directory traversal|xxe|template injection|malicious|attack|exploit)\b
    |\b(false positive|benign|logging artifact|just a hostname|not injection syntax|no malicious intent)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

AMBIGUOUS_CONTEXT_RE = re.compile(
    r"""
    \b(could|may be|might be|possibly|unclear|ambiguous|uncertain|potential(?:ly)?)\b
    |\b(likely|possible)\s+(dns\s+)?misconfig(?:uration)?\b
    |\b(typo|misconfig(?:uration)?|malformed input|valid identifier|just test data)\b
    |\b(appended to|as a subdomain of)\s+(a\s+)?legitimate\b
    |\bincomplete syntax\b
    |\b(lacks|not)\s+clear\b
    |\bsuspicious\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

STRONG_MALICIOUS_CONTEXT_RE = re.compile(
    r"""
    \b(sql\s*injection|sqli|oracle\s+sql\s+injection|mysql\s+sql\s+injection|pg_sleep|dbms_pipe|waitfor\s+delay)\b
    |\b(xss|cross[- ]site|script\s+tag|script\s+injection)\b.{0,100}\b(attempt|payload|probe|test|vector|injection)\b
    |\b(attempt|payload|probe|test|vector)\b.{0,100}\b(xss|cross[- ]site|script\s+tag|script\s+injection)\b
    |\b(path|directory)\s+traversal\b.{0,100}\b(attempt|payload|probe|test|pattern)\b
    |\b(double[- ]encoded|url[- ]encoded)\b.{0,100}\b(path|directory)\s+traversal\b
    |\btemplate\s+injection\b.{0,100}\b(attempt|payload|probe|test|pattern)\b
    |\b(ssrf|server[- ]side request forgery)\b.{0,120}\b(attempt|payload|probe|test|metadata|cloud metadata)\b
    |\bmetadata\s+endpoint\b.{0,120}\b(attempt|payload|probe|test|ssrf)\b
    |\b(open\s+redirect|xxe|xml external entity|deserialization|csrf)\b.{0,100}\b(attempt|payload|probe|test|vector|injection)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def normalize_label(value: str) -> str:
    text = (value or "").strip().upper()
    if text.startswith("M"):
        return "M"
    if text.startswith("B"):
        return "B"
    return "U"


def reason_category(reason: str) -> str:
    low = reason.lower()
    if any(token in low for token in ["sql injection", "pg_sleep", "dbms_pipe", "waitfor delay", "sqli"]):
        return "sql_injection"
    if any(token in low for token in ["xss", "cross-site", "script tag", "script injection"]):
        return "xss"
    if "traversal" in low:
        return "path_traversal"
    if "ssrf" in low or "metadata.google.internal" in low or "169.254" in low:
        return "ssrf_or_metadata_probe"
    if "template" in low or "ssti" in low:
        return "template_injection"
    if "xxe" in low or "xml external" in low:
        return "xxe"
    if "deserialization" in low:
        return "deserialization"
    if "open redirect" in low:
        return "open_redirect"
    if "csrf" in low:
        return "csrf"
    return "other_malicious"


def should_relabel(reason: str) -> bool:
    if not reason:
        return False
    if BENIGN_CONTEXT_RE.search(reason):
        return False
    if not MALICIOUS_REASON_RE.search(reason):
        return False
    category = reason_category(reason)
    if AMBIGUOUS_CONTEXT_RE.search(reason) and category != "sql_injection":
        return False
    return True


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


def collect_decisions(root: Path, manifest: Mapping[str, object]) -> Tuple[Dict[Tuple[str, str, str], str], Counter, Dict[Tuple[str, str, str], Tuple[str, str, str, str]]]:
    decisions: Dict[Tuple[str, str, str], str] = {}
    counts: Counter = Counter()
    examples: Dict[Tuple[str, str, str], Tuple[str, str, str, str]] = {}
    cols = [
        ("sonnet", SONNET_LABEL, SONNET_REASON),
        ("opus", OPUS_LABEL, OPUS_REASON),
    ]
    for family, dataset in manifest["datasets"].items():
        for chunk in dataset["chunks"]:
            path = root / chunk["path"]
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    for model, label_col, reason_col in cols:
                        label = normalize_label(row.get(label_col, ""))
                        reason = row.get(reason_col, "").strip()
                        if label == "M" or not should_relabel(reason):
                            continue
                        key = (model, label, reason)
                        decisions[key] = reason_category(reason)
                        counts[(model, label, reason_category(reason))] += 1
                        examples.setdefault(
                            key,
                            (
                                family,
                                row.get("ROW_ID", ""),
                                row.get("HOSTNAME", ""),
                                row.get("USERNAME", ""),
                            ),
                        )
    return decisions, counts, examples


def collect_decisions_from_reason_pairs(path: Path) -> Tuple[Dict[Tuple[str, str, str], str], Counter, Dict[Tuple[str, str, str], Tuple[str, str, str, str]]]:
    decisions: Dict[Tuple[str, str, str], str] = {}
    counts: Counter = Counter()
    examples: Dict[Tuple[str, str, str], Tuple[str, str, str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            model = (row.get("model") or "").strip()
            label = normalize_label(row.get("label", ""))
            reason = (row.get("reason") or "").strip()
            if model not in {"sonnet", "opus"} or label == "M" or not should_relabel(reason):
                continue
            category = reason_category(reason)
            key = (model, label, reason)
            decisions[key] = category
            counts[(model, label, category)] += int(row.get("count") or "0")
            examples.setdefault(
                key,
                (
                    row.get("family", ""),
                    row.get("row_id", ""),
                    row.get("hostname", ""),
                    row.get("username", ""),
                ),
            )
    return decisions, counts, examples


def apply_relabels(root: Path, manifest: Mapping[str, object], decisions: Mapping[Tuple[str, str, str], str]) -> Counter:
    stats: Counter = Counter()
    cols = [
        ("sonnet", SONNET_LABEL, SONNET_REASON),
        ("opus", OPUS_LABEL, OPUS_REASON),
    ]
    for family, dataset in manifest["datasets"].items():
        for chunk in dataset["chunks"]:
            path = root / chunk["path"]
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            changed = False
            with path.open("r", newline="", encoding="utf-8") as src, tmp_path.open("w", newline="", encoding="utf-8") as dst:
                reader = csv.DictReader(src)
                if not reader.fieldnames:
                    continue
                writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    row_changed = False
                    for model, label_col, reason_col in cols:
                        label = normalize_label(row.get(label_col, ""))
                        reason = row.get(reason_col, "").strip()
                        key = (model, label, reason)
                        if key in decisions:
                            row[label_col] = "M"
                            category = decisions[key]
                            stats[f"{model}.{label}->M"] += 1
                            stats[f"category.{category}"] += 1
                            row_changed = True
                    if row_changed:
                        row["RESOLVED_LABEL_BOTH_M"] = resolve_both_m(row.get(SONNET_LABEL, ""), row.get(OPUS_LABEL, ""))
                        row["LABEL_AGREEMENT"] = label_agreement(row.get(SONNET_LABEL, ""), row.get(OPUS_LABEL, ""))
                        stats["rows_changed"] += 1
                        stats[f"family.{family}.rows_changed"] += 1
                        changed = True
                    writer.writerow(row)
            if changed:
                os.replace(tmp_path, path)
                chunk["bytes"] = path.stat().st_size
            else:
                tmp_path.unlink(missing_ok=True)
    return stats


def write_decision_tsv(
    path: Path,
    decisions: Mapping[Tuple[str, str, str], str],
    examples: Mapping[Tuple[str, str, str], Tuple[str, str, str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["model", "old_label", "new_label", "category", "reason", "family", "row_id", "hostname", "username"])
        for key, category in sorted(decisions.items(), key=lambda item: (item[1], item[0])):
            model, old_label, reason = key
            family, row_id, hostname, username = examples[key]
            writer.writerow([model, old_label, "M", category, reason, family, row_id, hostname, username])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--reason-pairs",
        type=Path,
        default=None,
        help="Optional unique reason-label TSV. Defaults to relabel_candidate_reason_pairs.tsv when present.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = args.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reason_pairs_path: Optional[Path] = args.reason_pairs
    if reason_pairs_path is None:
        default_reason_pairs_path = args.root / "relabel_candidate_reason_pairs.tsv"
        reason_pairs_path = default_reason_pairs_path if default_reason_pairs_path.exists() else None
    if reason_pairs_path is not None:
        decisions, decision_counts, examples = collect_decisions_from_reason_pairs(reason_pairs_path)
    else:
        decisions, decision_counts, examples = collect_decisions(args.root, manifest)
    decision_path = args.root / "other_malicious_relabel_decisions.tsv"
    write_decision_tsv(decision_path, decisions, examples)

    report = {
        "dry_run": args.dry_run,
        "unique_reason_label_pairs_relabelled": len(decisions),
        "assignment_counts_by_model_old_label_category": {"|".join(k): v for k, v in sorted(decision_counts.items())},
        "decision_file": str(decision_path),
        "reason_pairs_file": str(reason_pairs_path) if reason_pairs_path is not None else None,
    }
    if not args.dry_run:
        apply_stats = apply_relabels(args.root, manifest, decisions)
        report["applied"] = dict(sorted(apply_stats.items()))
        manifest.setdefault("postprocessing", {})
        manifest["postprocessing"]["other_malicious_relabel"] = report
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path = args.root / "other_malicious_relabel_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
