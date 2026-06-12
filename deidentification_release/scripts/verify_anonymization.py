#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hib_deid import csv_input_paths, load_private_config, verify_release, verify_release_streaming


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify HIB de-identified release privacy, non-linkability, and utility invariants.")
    parser.add_argument("--private-input", type=Path, required=True)
    parser.add_argument("--public-release", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=False)
    parser.add_argument("--private-config", type=Path, default=None)
    parser.add_argument("--row-id-secret", default=None)
    parser.add_argument("--artifact-secret", default=None)
    parser.add_argument("--shuffle-secret", default=None)
    parser.add_argument("--min-k", type=int, default=50)
    parser.add_argument("--streaming-buckets", type=int, default=512)
    args = parser.parse_args()

    config = load_private_config(
        args.private_config,
        row_id_secret=args.row_id_secret,
        artifact_secret=args.artifact_secret,
        shuffle_secret=args.shuffle_secret,
    )
    if args.private_input.is_dir():
        report = verify_release_streaming(
            csv_input_paths(args.private_input),
            args.public_release,
            args.audit_dir,
            config,
            min_k=args.min_k,
            buckets=args.streaming_buckets,
        )
    else:
        report = verify_release(args.private_input, args.public_release, args.audit_dir, config, min_k=args.min_k)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
