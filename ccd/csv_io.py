from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Iterator, Sequence


def iter_malicious_csv_rows(path: Path) -> Iterator[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) < 2:
                continue
            hostname = row[0]
            family = row[1].strip()
            if hostname.strip().lower() == "hostname" and family.lower() in {"family", "label"}:
                continue
            if not hostname.strip() or not family:
                continue
            yield hostname, family


def read_malicious_csv_map(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for hostname, family in iter_malicious_csv_rows(path):
        out.setdefault(family, []).append(hostname)
    return out


def write_score_csv(
    path: Path,
    hostnames: Sequence[str],
    scores: Iterable[float],
    predictions: Iterable[object],
    *,
    groups: Sequence[str] | None = None,
    thresholds: Iterable[float] | None = None,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if groups is None:
            if thresholds is None:
                writer.writerow(["hostname", "score", "prediction"])
                for hostname, score, prediction in zip(hostnames, scores, predictions):
                    writer.writerow([hostname, f"{float(score):.6f}", int(prediction)])
            else:
                writer.writerow(["hostname", "threshold", "score", "prediction"])
                for hostname, threshold, score, prediction in zip(hostnames, thresholds, scores, predictions):
                    writer.writerow([hostname, f"{float(threshold):.6f}", f"{float(score):.6f}", int(prediction)])
            return

        if thresholds is None:
            raise ValueError("thresholds are required when groups are provided")
        writer.writerow(["hostname", "calibration_group", "threshold", "score", "prediction"])
        for hostname, group, threshold, score, prediction in zip(
            hostnames,
            groups,
            thresholds,
            scores,
            predictions,
        ):
            writer.writerow(
                [
                    hostname,
                    group,
                    f"{float(threshold):.6f}",
                    f"{float(score):.6f}",
                    int(prediction),
                ]
            )
