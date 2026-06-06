#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, IO, Iterable, Mapping


POSITIVE_LABEL = "verified_executable_semantics"
NEGATIVE_LABEL = "resolved_benign"
UNRESOLVED_LABEL = "unresolved"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"release row {line_number} is not an object")
            yield row


def released_artifact(row: Mapping[str, Any]) -> str:
    value = str(row.get("released_artifact", "")).strip()
    if not value:
        raise ValueError(f"row {row.get('public_row_id', '<unknown>')} has no released_artifact")
    return value


def calibration_group(row: Mapping[str, Any]) -> str:
    outputs = row.get("ccd_outputs", {})
    if not isinstance(outputs, Mapping):
        return ""
    return str(outputs.get("public_calibration_group", "")).strip()


def malicious_family(row: Mapping[str, Any]) -> str:
    raw = str(row.get("sink_family") or POSITIVE_LABEL).strip().lower()
    family = re.sub(r"[^a-z0-9_.:-]+", "_", raw).strip("_.:-")
    return family or POSITIVE_LABEL


class LineWriter:
    def __init__(self, handle: IO[str]) -> None:
        self.handle = handle
        self.count = 0

    def write(self, value: str) -> None:
        self.handle.write(value)
        self.handle.write("\n")
        self.count += 1


class CsvWriter:
    def __init__(self, handle: IO[str], header: list[str]) -> None:
        self.writer = csv.writer(handle, lineterminator="\n")
        self.writer.writerow(header)
        self.count = 0

    def write(self, row: list[Any]) -> None:
        self.writer.writerow(row)
        self.count += 1


class GroupWriter:
    def __init__(self, path: Path, stack: ExitStack) -> None:
        self.path = path
        self.stack = stack
        self.handle: IO[str] | None = None
        self.count = 0
        self.seen_present = False
        self.seen_missing = False

    def write_for_row(self, row: Mapping[str, Any]) -> None:
        group = calibration_group(row)
        if not group:
            self.seen_missing = True
            if self.seen_present:
                raise ValueError(f"cannot write {self.path.name}: public calibration groups are partially missing")
            return
        self.seen_present = True
        if self.seen_missing:
            raise ValueError(f"cannot write {self.path.name}: public calibration groups are partially missing")
        if self.handle is None:
            self.handle = self.stack.enter_context(self.path.open("w", encoding="utf-8", newline=""))
        self.handle.write(group)
        self.handle.write("\n")
        self.count += 1


def export_pipeline_inputs(public_release: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    by_split: Counter[str] = Counter()
    by_label: Counter[str] = Counter()
    query_labels_by_label: Counter[str] = Counter()
    row_count = 0
    files: dict[str, int] = {}

    with ExitStack() as stack:
        train_benign = LineWriter(stack.enter_context((output_dir / "benign.txt").open("w", encoding="utf-8", newline="")))
        train_malicious = CsvWriter(
            stack.enter_context((output_dir / "malicious.csv").open("w", encoding="utf-8", newline="")),
            ["hostname", "family"],
        )
        calibration_benign = LineWriter(
            stack.enter_context((output_dir / "benign_calibration.txt").open("w", encoding="utf-8", newline=""))
        )
        calibration_groups = GroupWriter(output_dir / "benign_calibration_groups.txt", stack)
        queries = LineWriter(stack.enter_context((output_dir / "queries.txt").open("w", encoding="utf-8", newline="")))
        query_groups = GroupWriter(output_dir / "query_groups.txt", stack)
        query_labels = CsvWriter(
            stack.enter_context((output_dir / "query_labels.csv").open("w", encoding="utf-8", newline="")),
            ["public_row_id", "split", "label", "source_family", "hostname"],
        )
        refresh_benign = LineWriter(
            stack.enter_context((output_dir / "recent_benign_window.txt").open("w", encoding="utf-8", newline=""))
        )
        refresh_groups = GroupWriter(output_dir / "recent_benign_window_groups.txt", stack)

        for row in iter_jsonl(public_release):
            row_count += 1
            split = str(row.get("split", ""))
            label = str(row.get("label", ""))
            by_split[split] += 1
            by_label[label] += 1

            if split == "train" and label == NEGATIVE_LABEL:
                train_benign.write(released_artifact(row))
            elif split == "train" and label == POSITIVE_LABEL:
                train_malicious.write([released_artifact(row), malicious_family(row)])
            elif split == "calibration" and label == NEGATIVE_LABEL:
                calibration_benign.write(released_artifact(row))
                calibration_groups.write_for_row(row)

            if split in {"test", "validation"} and label != UNRESOLVED_LABEL:
                queries.write(released_artifact(row))
                query_groups.write_for_row(row)
                query_labels_by_label[label] += 1
                query_labels.write(
                    [
                        row.get("public_row_id", ""),
                        split,
                        label,
                        row.get("source_family", ""),
                        released_artifact(row),
                    ]
                )

            if split == "validation" and label == NEGATIVE_LABEL:
                refresh_benign.write(released_artifact(row))
                refresh_groups.write_for_row(row)

        files["benign.txt"] = train_benign.count
        files["malicious.csv"] = train_malicious.count
        files["benign_calibration.txt"] = calibration_benign.count
        if calibration_groups.count:
            files["benign_calibration_groups.txt"] = calibration_groups.count
        files["queries.txt"] = queries.count
        if query_groups.count:
            files["query_groups.txt"] = query_groups.count
        files["query_labels.csv"] = query_labels.count
        if refresh_benign.count:
            files["recent_benign_window.txt"] = refresh_benign.count
        if refresh_groups.count:
            files["recent_benign_window_groups.txt"] = refresh_groups.count

    if row_count == 0:
        raise ValueError("public release contains no rows")

    required = {
        "train_benign": files["benign.txt"],
        "train_malicious": files["malicious.csv"],
        "calibration_benign": files["benign_calibration.txt"],
        "queries": files["queries.txt"],
    }
    empty = [name for name, count in required.items() if count == 0]
    if empty:
        raise ValueError(f"public release is missing required rows for: {', '.join(empty)}")

    missing_query_labels = [
        name
        for name, label in (
            ("query_benign", NEGATIVE_LABEL),
            ("query_malicious", POSITIVE_LABEL),
        )
        if query_labels_by_label[label] == 0
    ]
    if missing_query_labels:
        raise ValueError(
            "public release is missing required labeled query rows for: "
            f"{', '.join(missing_query_labels)}"
        )

    for filename in ("benign_calibration_groups.txt", "query_groups.txt", "recent_benign_window_groups.txt"):
        if files.get(filename, 0) == 0:
            path = output_dir / filename
            if path.exists():
                path.unlink()

    if files.get("recent_benign_window.txt", 0) == 0:
        path = output_dir / "recent_benign_window.txt"
        if path.exists():
            path.unlink()

    summary = {
        "public_release": str(public_release),
        "output_dir": str(output_dir),
        "counts": {
            "rows": row_count,
            "by_split": dict(by_split),
            "by_label": dict(by_label),
            "query_labels_by_label": dict(query_labels_by_label),
        },
        "files": files,
        "policy": {
            "released_artifact_used": True,
            "row_multiplicity_preserved": True,
            "unresolved_rows_excluded_from_training_and_queries": True,
            "malicious_family_source": "sink_family",
        },
    }
    (output_dir / "pipeline_inputs_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CAHO/CCD line and CSV inputs from a de-identified HIB public-release JSONL."
    )
    parser.add_argument("--public-release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_pipeline_inputs(args.public_release, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
