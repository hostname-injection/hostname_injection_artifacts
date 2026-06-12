#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccd.calibration import calibrate_threshold, calibrate_thresholds_by_group
from ccd.io import ModelBundle, load_model, save_model
from ccd.line_io import read_nonempty_lines, read_parallel_lines
from ccd.preprocess import normalize_hostname


def read_lines(path: Path):
    return read_nonempty_lines(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--benign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups", type=Path, default=None, help="Optional one-calibration-group-per-benign-row file.")
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path for a model bundle with the calibrated threshold embedded.",
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-normalize", action="store_true", help="Skip hostname normalization")
    args = parser.parse_args()

    model = load_model(args.model)
    hostnames = read_lines(args.benign)
    if not args.no_normalize:
        hostnames = [normalize_hostname(h) for h in hostnames]
    groups = read_parallel_lines(args.groups, len(hostnames), field_name="groups") if args.groups else None

    scores = model.score(hostnames, batch_size=args.batch_size, normalize=False)
    alpha = args.alpha if args.alpha is not None else model.config.calibration.alpha
    threshold = calibrate_threshold(scores, alpha)
    grouped_thresholds = calibrate_thresholds_by_group(scores, groups, alpha) if groups is not None else {}
    model.threshold = threshold
    if hasattr(model, "grouped_thresholds"):
        model.grouped_thresholds = grouped_thresholds or None

    output = {
        "alpha": alpha,
        "threshold": threshold,
        "num_samples": len(scores),
        "threshold_source": "grouped_benign_calibration_scores" if groups is not None else "benign_calibration_scores",
        "grouped_thresholds": grouped_thresholds,
        "n_calibration_groups": len(grouped_thresholds),
        "score_path": {
            "approximate": False,
            "approximate_k": None,
            "normalized_inputs": not args.no_normalize,
        },
    }
    args.output.write_text(json.dumps(output, indent=2))
    if args.save_model:
        save_model(
            args.save_model,
            ModelBundle(
                axes=model.cones.axes,
                benign_prior=model.benign_prior,
                malicious_priors=model.malicious_priors,
                config=model.config,
                threshold=threshold,
                grouped_thresholds=grouped_thresholds or None,
            ),
        )
    print(f"Wrote calibration to {args.output}")


if __name__ == "__main__":
    main()
