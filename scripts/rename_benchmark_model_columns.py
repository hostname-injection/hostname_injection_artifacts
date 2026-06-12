#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_ROOT = Path(os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark"))

RENAMES: Dict[str, str] = {
    "CLAUDE_SONNET_4_5_IS_DNS_CMD_INJECTION": "GPT_5_5_IS_DNS_CMD_INJECTION",
    "CLAUDE_SONNET_4_5_DNS_CMD_INJECTION_CONFIDENCE": "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE",
    "CLAUDE_SONNET_4_5_DNS_CMD_INJECTION_REASON": "GPT_5_5_DNS_CMD_INJECTION_REASON",
    "CLAUDE_OPUS_4_5_IS_DNS_CMD_INJECTION": "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION",
    "CLAUDE_OPUS_4_5_DNS_CMD_INJECTION_CONFIDENCE": "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE",
    "CLAUDE_OPUS_4_5_DNS_CMD_INJECTION_REASON": "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_REASON",
}


def renamed(fieldnames: Iterable[str]) -> List[str]:
    return [RENAMES.get(name, name) for name in fieldnames]


def rewrite_chunk(path: Path, expected_schema: List[str]) -> bool:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    changed = False
    with path.open("r", newline="", encoding="utf-8") as src:
        reader = csv.reader(src)
        try:
            header = next(reader)
        except StopIteration:
            return False
        new_header = renamed(header)
        changed = new_header != header
        if not changed:
            return False
        if new_header != expected_schema:
            raise ValueError(f"{path}: renamed header does not match manifest schema")
        with tmp_path.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.writer(dst)
            writer.writerow(new_header)
            writer.writerows(reader)
    os.replace(tmp_path, path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = args.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_schema = list(manifest["schema"])
    new_schema = renamed(old_schema)
    if len(set(new_schema)) != len(new_schema):
        raise SystemExit("renamed schema contains duplicate columns")

    chunks_changed = 0
    for dataset in manifest["datasets"].values():
        for chunk in dataset["chunks"]:
            path = args.root / chunk["path"]
            if args.dry_run:
                with path.open("r", newline="", encoding="utf-8") as handle:
                    header = next(csv.reader(handle))
                if renamed(header) != header:
                    chunks_changed += 1
                continue
            if rewrite_chunk(path, new_schema):
                chunks_changed += 1
                chunk["bytes"] = path.stat().st_size

    if not args.dry_run:
        manifest["schema"] = new_schema
        manifest.setdefault("schema_corrections", {})
        manifest["schema_corrections"]["model_column_names_2026_05_29"] = {
            "reason": "Correct mislabeled model names: former Sonnet 4.5 columns were GPT 5.5 labels; former Opus 4.5 columns were Opus 4.8 labels.",
            "renames": RENAMES,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"dry_run": args.dry_run, "chunks_changed": chunks_changed, "renames": RENAMES}, indent=2), flush=True)


if __name__ == "__main__":
    main()
