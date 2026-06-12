"""Explain CCD scores by exposing top cone contributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .io import load_model
from .preprocess import normalize_hostname


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]


def read_parallel_lines(path: Path, expected_len: int, field_name: str) -> List[str]:
    values = read_lines(path)
    if len(values) != expected_len:
        raise ValueError(f"{field_name} file has {len(values)} rows; expected {expected_len}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explain CCD predictions.")
    parser.add_argument("--model", required=True, type=Path, help="CCD .npz model bundle")
    parser.add_argument("--input", required=True, type=Path, help="Input file of hostnames (one per line)")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--calibration", type=Path, default=None, help="JSON file from `ccd calibrate`")
    parser.add_argument("--groups", type=Path, default=None, help="Optional one-calibration-group-per-input-row file.")
    parser.add_argument(
        "--require-group-thresholds",
        action="store_true",
        help="Fail if --groups contains a group missing from grouped calibration output.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top cones to show")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-normalize", action="store_true", help="Skip hostname normalization")
    parser.add_argument("--approximate", action="store_true", help="Use fast approximate scoring")
    parser.add_argument("--approximate-k", type=int, default=None, help="Top-k cones for approximate scoring")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    model = load_model(args.model)
    hostnames = read_lines(args.input)
    if not args.no_normalize:
        hostnames = [normalize_hostname(h) for h in hostnames]
    groups = read_parallel_lines(args.groups, len(hostnames), "groups") if args.groups else None
    if args.threshold is not None:
        model.threshold = args.threshold
    if args.calibration:
        calib = json.loads(args.calibration.read_text())
        model.threshold = float(calib.get("threshold", model.threshold or 0.0))
        model.grouped_thresholds = calib.get("grouped_thresholds", getattr(model, "grouped_thresholds", None))

    explanations = model.explain(
        hostnames,
        batch_size=args.batch_size,
        normalize=False,
        top_k=args.top_k,
        calibration_groups=groups,
        missing_group_threshold="error" if args.require_group_thresholds else "default",
        approximate=args.approximate,
        approximate_k=args.approximate_k,
    )

    payload = {
        "model": str(args.model),
        "count": len(explanations),
        "top_k": args.top_k,
        "grouped_thresholds_used": groups is not None,
        "explanations": explanations,
    }

    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Wrote explanations to {args.output}")
        return 0

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
