from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional


def list_file_paths(folder_path: Path) -> List[Path]:
    file_paths: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            file_paths.append(Path(dirpath) / filename)
    return file_paths


def read_hostnames_from_csv(path: Path, *, column: str = "Hostname") -> List[str]:
    hostnames: List[str] = []
    with path.open("r", newline="", errors="ignore") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if column in row:
                hostnames.append(str(row[column]))
    return hostnames


def read_hostnames_from_jsonl_dir(path: Path, *, key: str = "hostname") -> List[str]:
    hostnames: List[str] = []
    for file_path in list_file_paths(path):
        if file_path.name == ".DS_Store":
            continue
        try:
            with file_path.open("r", errors="ignore") as handle:
                for line in handle:
                    try:
                        data = json.loads(line.strip())
                        if key in data:
                            hostnames.append(str(data[key]))
                    except Exception:
                        continue
        except Exception:
            continue
    return hostnames


def read_hostnames_from_txt_dir(path: Path, *, include_csv: bool = True, csv_column: str = "Hostname") -> List[str]:
    hostnames: List[str] = []
    for file_path in list_file_paths(path):
        if file_path.name == ".DS_Store":
            continue
        try:
            if file_path.suffix.lower() == ".txt":
                with file_path.open("r", errors="ignore") as handle:
                    for line in handle:
                        hostnames.append(line.strip().replace("\n", ""))
            elif include_csv and file_path.suffix.lower() == ".csv":
                hostnames.extend(read_hostnames_from_csv(file_path, column=csv_column))
        except Exception:
            continue
    return hostnames


def read_hostnames_from_benign_dir(path: Path) -> List[str]:
    hostnames: List[str] = []
    if not path.is_dir():
        return hostnames
    for file_path in list_file_paths(path):
        if not file_path.is_file() or file_path.suffix.lower() != ".txt":
            continue
        try:
            with file_path.open("r", errors="ignore") as handle:
                for line in handle:
                    hostnames.append(line.strip())
        except Exception:
            continue
    return hostnames


def filter_hostnames(hostnames: Iterable[str], *, min_length: int = 5, dedup: bool = True) -> List[str]:
    filtered = [
        h for h in hostnames
        if isinstance(h, str) and len(h) > min_length
    ]
    if dedup:
        return list(dict.fromkeys(filtered))
    return filtered
