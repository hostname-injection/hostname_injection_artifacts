#!/usr/bin/env python3
"""Export private original malicious attack-vector inventory with labels/reasons."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


FIELDNAMES = [
    "attack_vector",
    "source_indicator",
    "seen_in_user_login",
    "seen_in_dns_resolution",
    "total_rows",
    "user_login_rows",
    "dns_resolution_rows",
    "content_type_values",
    "resolved_labels",
    "label_agreements",
    "gpt_5_5_labels",
    "gpt_5_5_confidences",
    "gpt_5_5_reasons",
    "claude_opus_4_8_labels",
    "claude_opus_4_8_confidences",
    "claude_opus_4_8_reasons",
    "example_row_ids",
]


@dataclass
class AttackVector:
    attack_vector: str
    source_counts: Counter[str] = field(default_factory=Counter)
    content_types: set[str] = field(default_factory=set)
    resolved_labels: set[str] = field(default_factory=set)
    label_agreements: set[str] = field(default_factory=set)
    gpt_labels: set[str] = field(default_factory=set)
    gpt_confidences: set[str] = field(default_factory=set)
    gpt_reasons: set[str] = field(default_factory=set)
    opus_labels: set[str] = field(default_factory=set)
    opus_confidences: set[str] = field(default_factory=set)
    opus_reasons: set[str] = field(default_factory=set)
    example_row_ids: list[str] = field(default_factory=list)

    def add(self, row: dict[str, str], source: str, max_examples: int) -> None:
        self.source_counts[source] += 1
        self._add_nonblank(self.content_types, row.get("CONTENT_TYPE", ""))
        self._add_nonblank(self.resolved_labels, row.get("RESOLVED_LABEL_BOTH_M", ""))
        self._add_nonblank(self.label_agreements, row.get("LABEL_AGREEMENT", ""))
        self._add_nonblank(self.gpt_labels, row.get("GPT_5_5_IS_DNS_CMD_INJECTION", ""))
        self._add_nonblank(self.gpt_confidences, row.get("GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE", ""))
        self._add_nonblank(self.gpt_reasons, row.get("GPT_5_5_DNS_CMD_INJECTION_REASON", ""))
        self._add_nonblank(self.opus_labels, row.get("CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION", ""))
        self._add_nonblank(self.opus_confidences, row.get("CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE", ""))
        self._add_nonblank(self.opus_reasons, row.get("CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_REASON", ""))
        row_id = row.get("ROW_ID", "")
        if row_id and len(self.example_row_ids) < max_examples:
            self.example_row_ids.append(row_id)

    @staticmethod
    def _add_nonblank(values: set[str], value: str) -> None:
        value = (value or "").strip()
        if value:
            values.add(value)

    def to_row(self) -> dict[str, str | int | bool]:
        user_rows = self.source_counts["user_login"]
        dns_rows = self.source_counts["dns_resolution"]
        if user_rows and dns_rows:
            source_indicator = "both"
        elif user_rows:
            source_indicator = "user_login"
        else:
            source_indicator = "dns_resolution"
        return {
            "attack_vector": self.attack_vector,
            "source_indicator": source_indicator,
            "seen_in_user_login": bool(user_rows),
            "seen_in_dns_resolution": bool(dns_rows),
            "total_rows": user_rows + dns_rows,
            "user_login_rows": user_rows,
            "dns_resolution_rows": dns_rows,
            "content_type_values": json.dumps(sorted(self.content_types), ensure_ascii=False),
            "resolved_labels": json.dumps(sorted(self.resolved_labels), ensure_ascii=False),
            "label_agreements": json.dumps(sorted(self.label_agreements), ensure_ascii=False),
            "gpt_5_5_labels": json.dumps(sorted(self.gpt_labels), ensure_ascii=False),
            "gpt_5_5_confidences": json.dumps(sorted(self.gpt_confidences), ensure_ascii=False),
            "gpt_5_5_reasons": json.dumps(sorted(self.gpt_reasons), ensure_ascii=False),
            "claude_opus_4_8_labels": json.dumps(sorted(self.opus_labels), ensure_ascii=False),
            "claude_opus_4_8_confidences": json.dumps(sorted(self.opus_confidences), ensure_ascii=False),
            "claude_opus_4_8_reasons": json.dumps(sorted(self.opus_reasons), ensure_ascii=False),
            "example_row_ids": json.dumps(self.example_row_ids, ensure_ascii=False),
        }


def iter_chunks(root: Path, family: str) -> Iterable[Path]:
    chunk_dir = root / "data" / family / "chunks"
    yield from sorted(chunk_dir.glob("*.csv"))


def source_name(family: str) -> str:
    if family == "user_logins":
        return "user_login"
    if family == "dns_hostnames":
        return "dns_resolution"
    raise ValueError(f"unknown family: {family}")


def export(root: Path, output: Path, summary_path: Path, max_examples: int) -> None:
    vectors: dict[str, AttackVector] = {}
    scanned_rows = Counter()
    malicious_rows = Counter()
    source_files = Counter()

    for family in ("user_logins", "dns_hostnames"):
        source = source_name(family)
        for path in iter_chunks(root, family):
            source_files[source] += 1
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    scanned_rows[source] += 1
                    if row.get("RESOLVED_LABEL_BOTH_M") != "M":
                        continue
                    malicious_rows[source] += 1
                    vector = (row.get("CONTENT") or row.get("HOSTNAME") or row.get("USERNAME") or "").strip()
                    if not vector:
                        continue
                    record = vectors.get(vector)
                    if record is None:
                        record = AttackVector(attack_vector=vector)
                        vectors[vector] = record
                    record.add(row, source, max_examples=max_examples)
            if sum(source_files.values()) % 25 == 0:
                print(
                    "export_progress "
                    f"files={sum(source_files.values())} "
                    f"rows={sum(scanned_rows.values())} "
                    f"malicious_rows={sum(malicious_rows.values())} "
                    f"unique_vectors={len(vectors)}",
                    flush=True,
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for vector in sorted(vectors):
            writer.writerow(vectors[vector].to_row())

    both_sources = sum(1 for vector in vectors.values() if vector.source_counts["user_login"] and vector.source_counts["dns_resolution"])
    summary = {
        "input_root": str(root),
        "output_csv": str(output),
        "attack_vector_definition": "Rows with RESOLVED_LABEL_BOTH_M == 'M'; this means both model labels were malicious.",
        "deduplication_key": "Exact original CONTENT value, falling back to HOSTNAME then USERNAME only if CONTENT is blank.",
        "source_indicator_values": {
            "user_login": "Observed only in user-login rows.",
            "dns_resolution": "Observed only in DNS-resolution rows.",
            "both": "Observed in both user-login and DNS-resolution rows.",
        },
        "scanned_rows": dict(scanned_rows),
        "malicious_rows": dict(malicious_rows),
        "source_files": dict(source_files),
        "unique_attack_vectors": len(vectors),
        "unique_user_login_only": sum(1 for v in vectors.values() if v.source_counts["user_login"] and not v.source_counts["dns_resolution"]),
        "unique_dns_resolution_only": sum(1 for v in vectors.values() if v.source_counts["dns_resolution"] and not v.source_counts["user_login"]),
        "unique_both_sources": both_sources,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Benchmark root containing data/{user_logins,dns_hostnames}/chunks.")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV for private original attack-vector inventory.")
    parser.add_argument("--summary", type=Path, required=True, help="Destination JSON summary.")
    parser.add_argument("--max-examples", type=int, default=5, help="Maximum example row IDs retained per attack vector.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export(args.root, args.output, args.summary, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
