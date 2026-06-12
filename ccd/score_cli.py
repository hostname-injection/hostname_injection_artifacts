"""Score a list of hostnames with a saved CCD model bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibration import threshold_for_group
from .csv_io import write_score_csv
from .io import load_model
from .line_io import read_nonempty_lines, read_parallel_lines
from .preprocess import normalize_hostname


def read_lines(path: Path):
    return read_nonempty_lines(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score hostnames with CCD.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--calibration", type=Path, default=None, help="JSON file from calibrate")
    parser.add_argument("--groups", type=Path, default=None, help="Optional one-calibration-group-per-input-row file.")
    parser.add_argument(
        "--require-group-thresholds",
        action="store_true",
        help="Fail if --groups contains a group missing from grouped calibration output.",
    )
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

    scores = model.score(
        hostnames,
        batch_size=args.batch_size,
        normalize=False,
        approximate=args.approximate,
        approximate_k=args.approximate_k,
    )
    threshold = args.threshold if args.threshold is not None else (model.threshold or 0.0)
    grouped_thresholds = getattr(model, "grouped_thresholds", None)
    if args.calibration:
        calib = json.loads(args.calibration.read_text())
        threshold = float(calib.get("threshold", threshold))
        grouped_thresholds = calib.get("grouped_thresholds", grouped_thresholds)
    if groups is not None:
        row_thresholds = np.array(
            [
                threshold_for_group(
                    group,
                    threshold,
                    grouped_thresholds,
                    missing="error" if args.require_group_thresholds else "default",
                )
                for group in groups
            ],
            dtype=np.float64,
        )
    else:
        row_thresholds = np.full(len(scores), threshold, dtype=np.float64)
    preds = scores > row_thresholds

    write_score_csv(
        args.output,
        hostnames,
        scores,
        preds,
        groups=groups,
        thresholds=row_thresholds if groups is not None else None,
    )
    print(f"Wrote scores to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
