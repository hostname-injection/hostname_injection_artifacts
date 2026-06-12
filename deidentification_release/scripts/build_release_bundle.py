#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hib_deid import build_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the public HIB release bundle.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=None, help="Optional directory used to derive relative archive paths.")
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()

    hashes = build_bundle(args.output, args.paths, base_dir=args.base_dir)
    print(json.dumps({"bundle": str(args.output), "files": hashes}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
