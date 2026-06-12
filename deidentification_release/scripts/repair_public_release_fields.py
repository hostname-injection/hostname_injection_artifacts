#!/usr/bin/env python3
"""Stream-repair public release fields after policy coarsening changes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hib_deid import (
    PUBLIC_SCHEMA_FIELDS,
    canonicalize_artifact,
    public_length_bucket,
    public_character_class_mask,
    public_release_safety_postprocess,
    public_sink_and_evidence,
    row_integrity_hash,
    write_sha256_sidecar,
)


def repair_row(row: dict) -> dict:
    released = public_release_safety_postprocess(str(row.get("released_artifact", "")))
    label = str(row.get("label", ""))
    sink, evidence = public_sink_and_evidence(label, str(row.get("sink_family", "")), str(row.get("evidence_tier", "")))
    row = dict(row)
    row["released_artifact"] = released
    row["released_canonical_artifact"] = canonicalize_artifact(released)
    row["time_bucket"] = "withheld"
    row["sink_family"] = sink
    row["evidence_tier"] = evidence
    row["obfuscation_family"] = "none" if str(row.get("obfuscation_family", "")) == "none" else "present"
    row["released_length_bucket"] = public_length_bucket(released)
    row["character_class_mask"] = public_character_class_mask(released)
    row["row_integrity_hash"] = row_integrity_hash(row)
    return row


def repair(input_jsonl: Path, output_jsonl: Path) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            count += 1
            row = repair_row(json.loads(line))
            dst.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            if count % 1_000_000 == 0:
                print(f"repair_progress rows={count}", file=sys.stderr, flush=True)
    output_jsonl.with_suffix(".schema.json").write_text(json.dumps({"fields": PUBLIC_SCHEMA_FIELDS}, indent=2), encoding="utf-8")
    write_sha256_sidecar(output_jsonl.with_suffix(output_jsonl.suffix + ".sha256"), output_jsonl)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = repair(args.input, args.output)
    print(json.dumps({"rows": rows, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
