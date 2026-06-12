#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hib_deid import anonymize_csv, anonymize_csv_files, load_private_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a non-linkable de-identified HIB public release.")
    parser.add_argument("--input-private", type=Path, required=True)
    parser.add_argument("--private-config", type=Path, default=None)
    parser.add_argument("--public-policy", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--shuffle-buckets", type=int, default=256)
    parser.add_argument("--row-id-secret", default=None)
    parser.add_argument("--artifact-secret", default=None)
    parser.add_argument("--shuffle-secret", default=None)
    args = parser.parse_args()

    config = load_private_config(
        args.private_config,
        row_id_secret=args.row_id_secret,
        artifact_secret=args.artifact_secret,
        shuffle_secret=args.shuffle_secret,
    )
    if args.input_private.is_dir():
        input_paths = sorted(args.input_private.glob("*.csv"))
        manifest = anonymize_csv_files(input_paths, args.output, args.audit_dir, config, shuffle_buckets=args.shuffle_buckets)
    else:
        manifest = anonymize_csv(args.input_private, args.output, args.audit_dir, config)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
