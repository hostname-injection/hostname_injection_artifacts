from __future__ import annotations

import bisect
import csv
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - real training environments should have torch installed.
    class Dataset:  # type: ignore[no-redef]
        pass

from .preprocess import normalize_hostname


GPT_5_5_LABEL = "GPT_5_5_IS_DNS_CMD_INJECTION"
GPT_5_5_CONF = "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE"
GPT_5_5_REASON = "GPT_5_5_DNS_CMD_INJECTION_REASON"
OPUS_LABEL = "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION"
OPUS_CONF = "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE"
OPUS_REASON = "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_REASON"
RESOLVED_LABEL = "RESOLVED_LABEL_BOTH_M"

# Backward-compatible Python names for older training code. The CSV schema uses
# GPT_5_5_* columns; these aliases do not imply Sonnet labeled the data.
SONNET_LABEL = GPT_5_5_LABEL
SONNET_CONF = GPT_5_5_CONF
SONNET_REASON = GPT_5_5_REASON


class BenchmarkFamily(str, Enum):
    USER_LOGINS = "user_logins"
    DNS_HOSTNAMES = "dns_hostnames"
    BOTH = "both"


class BenchmarkLabelMethod(str, Enum):
    RESOLVED = "resolved"
    RESOLVED_OR_DISAGREE_MALICIOUS = "resolved-or-disagree-malicious"
    GPT_5_5_ONLY = "gpt-5.5-only"
    OPUS_4_8_ONLY = "opus-4.8-only"
    SONNET_ONLY = "sonnet-only"
    OPUS_ONLY = "opus-only"
    BOTH_DISAGREE_MALICIOUS = "both-disagree-malicious"
    BOTH_DISAGREE_BENIGN = "both-disagree-benign"
    BOTH_DISAGREE_UNKNOWN = "both-disagree-unknown"
    ANY_MALICIOUS_ELSE_BENIGN = "any-malicious-else-benign"


class BenchmarkTextField(str, Enum):
    CONTENT = "CONTENT"
    HOSTNAME = "HOSTNAME"
    USERNAME = "USERNAME"
    AUTO = "auto"


POSITIVE_FAMILY_COLUMNS = (
    "sink_family",
    "SINK_FAMILY",
    "payload_family",
    "PAYLOAD_FAMILY",
    "payload_class",
    "PAYLOAD_CLASS",
    "sink_harness_class",
    "SINK_HARNESS_CLASS",
)
SPLIT_COLUMNS = (
    "split",
    "SPLIT",
    "partition",
    "PARTITION",
    "DATA_SPLIT",
    "DATA_PARTITION",
)


@dataclass(frozen=True)
class BenchmarkDatasetStats:
    total_rows: int
    selected_rows: int
    family_rows: Mapping[str, int]
    chunk_count: int
    selected_label_rows: Mapping[str, int] = field(default_factory=dict)


def normalize_benchmark_label(value: Optional[str]) -> str:
    text = "" if value is None else str(value).strip().upper()
    if text.startswith("B"):
        return "B"
    if text.startswith("M"):
        return "M"
    return "U"


def label_to_int(label: str, *, unknown_label: int = -1) -> int:
    if label == "B":
        return 0
    if label == "M":
        return 1
    return unknown_label


def resolve_benchmark_label(
    row: Mapping[str, str],
    method: Union[BenchmarkLabelMethod, str],
) -> str:
    method = BenchmarkLabelMethod(method)
    resolved_raw = row.get(RESOLVED_LABEL)
    resolved_present = resolved_raw is not None and str(resolved_raw).strip() != ""
    resolved = normalize_benchmark_label(resolved_raw)
    if method == BenchmarkLabelMethod.RESOLVED:
        return resolved

    gpt_5_5 = normalize_benchmark_label(row.get(GPT_5_5_LABEL))
    opus = normalize_benchmark_label(row.get(OPUS_LABEL))

    if method == BenchmarkLabelMethod.RESOLVED_OR_DISAGREE_MALICIOUS and resolved_present:
        return resolved
    if method == BenchmarkLabelMethod.RESOLVED_OR_DISAGREE_MALICIOUS:
        method = BenchmarkLabelMethod.BOTH_DISAGREE_MALICIOUS

    if method in {BenchmarkLabelMethod.GPT_5_5_ONLY, BenchmarkLabelMethod.SONNET_ONLY}:
        return gpt_5_5
    if method in {BenchmarkLabelMethod.OPUS_4_8_ONLY, BenchmarkLabelMethod.OPUS_ONLY}:
        return opus
    if method == BenchmarkLabelMethod.ANY_MALICIOUS_ELSE_BENIGN:
        return "M" if "M" in {gpt_5_5, opus} else "B"
    if gpt_5_5 == opus:
        return gpt_5_5
    if "M" in {gpt_5_5, opus} and method == BenchmarkLabelMethod.BOTH_DISAGREE_MALICIOUS:
        return "M"
    if "B" in {gpt_5_5, opus} and method == BenchmarkLabelMethod.BOTH_DISAGREE_BENIGN:
        return "B"
    return "U"


def benchmark_positive_family(row: Mapping[str, str], *, default: str = "positive") -> str:
    for column in POSITIVE_FAMILY_COLUMNS:
        raw = str(row.get(column, "")).strip().lower()
        if not raw or raw in {"none", "unknown", "unresolved", "withheld", "present"}:
            continue
        family = re.sub(r"[^a-z0-9_.:-]+", "_", raw).strip("_.:-")
        if family:
            return family
    return default


def benchmark_row_split(row: Mapping[str, str]) -> str:
    for column in SPLIT_COLUMNS:
        raw = str(row.get(column, "")).strip().lower()
        if raw:
            return raw
    return ""


def _normalize_splits(splits: Optional[Union[str, Sequence[str]]]) -> Optional[set[str]]:
    if splits is None:
        return None
    values = [splits] if isinstance(splits, str) else list(splits)
    normalized = {str(value).strip().lower() for value in values if str(value).strip()}
    if not normalized:
        raise ValueError("splits must contain at least one non-empty split")
    return normalized


def _as_families(family: Union[BenchmarkFamily, str, Sequence[Union[BenchmarkFamily, str]]]) -> List[str]:
    if isinstance(family, (str, BenchmarkFamily)):
        value = BenchmarkFamily(family)
        if value == BenchmarkFamily.BOTH:
            return [BenchmarkFamily.USER_LOGINS.value, BenchmarkFamily.DNS_HOSTNAMES.value]
        return [value.value]
    families: List[str] = []
    for item in family:
        families.extend(_as_families(item))
    return list(dict.fromkeys(families))


def _read_chunk_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


class HostnameCommandInjectionBenchmarkDataset(Dataset):
    """Map-style PyTorch dataset for HostnameCommandInjectionBenchmark chunks.

    The dataset indexes chunk files from `manifest.json` and loads at most a
    small LRU cache of CSV chunks into memory. When `splits` is supplied, rows
    are selected only from those benchmark partitions.
    """

    def __init__(
        self,
        root: Union[str, Path] = "HostnameCommandInjectionBenchmark",
        *,
        family: Union[BenchmarkFamily, str, Sequence[Union[BenchmarkFamily, str]]] = BenchmarkFamily.BOTH,
        label_method: Union[BenchmarkLabelMethod, str] = BenchmarkLabelMethod.BOTH_DISAGREE_UNKNOWN,
        drop_unknown: bool = True,
        include_explanations: bool = False,
        include_metadata: bool = False,
        return_dict: bool = True,
        text_field: Union[BenchmarkTextField, str] = BenchmarkTextField.AUTO,
        normalize_text: bool = False,
        unknown_label: int = -1,
        splits: Optional[Union[str, Sequence[str]]] = None,
        max_rows: Optional[int] = None,
        cache_chunks: int = 1,
        transform: Optional[Callable[[str], Any]] = None,
        target_transform: Optional[Callable[[int], Any]] = None,
    ) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.families = _as_families(family)
        self.label_method = BenchmarkLabelMethod(label_method)
        self.drop_unknown = drop_unknown
        self.include_explanations = include_explanations
        self.include_metadata = include_metadata
        self.return_dict = return_dict
        self.text_field = BenchmarkTextField(text_field)
        self.normalize_text = normalize_text
        self.unknown_label = unknown_label
        self.splits = _normalize_splits(splits)
        self.cache_chunks = max(int(cache_chunks), 0)
        self.transform = transform
        self.target_transform = target_transform

        self._chunks: List[Tuple[str, Path, int]] = []
        family_rows: Dict[str, int] = {}
        selected_label_rows: Dict[str, int] = {}
        for fam in self.families:
            dataset = self.manifest["datasets"][fam]
            family_rows[fam] = int(dataset["rows"])
            for chunk in dataset["chunks"]:
                self._chunks.append((fam, self.root / chunk["path"], int(chunk["rows"])))

        self._selected: Optional[List[Tuple[int, int]]] = None
        if self.drop_unknown or self.splits is not None or max_rows is not None:
            self._selected = []
            for chunk_index, (_, path, _) in enumerate(self._chunks):
                for row_index, row in enumerate(_read_chunk_rows(path)):
                    if self.splits is not None and benchmark_row_split(row) not in self.splits:
                        continue
                    label = resolve_benchmark_label(row, self.label_method)
                    if self.drop_unknown and label == "U":
                        continue
                    self._selected.append((chunk_index, row_index))
                    selected_label_rows[label] = selected_label_rows.get(label, 0) + 1
                    if max_rows is not None and len(self._selected) >= max_rows:
                        break
                if max_rows is not None and len(self._selected) >= max_rows:
                    break

        self._prefix: List[int] = []
        total = 0
        for _, _, rows in self._chunks:
            total += rows
            self._prefix.append(total)
        if self._selected is None and max_rows is not None:
            total = min(total, max_rows)
            self._prefix = []
            running = 0
            for _, _, rows in self._chunks:
                running += rows
                self._prefix.append(running)
                if running >= total:
                    break
        self._length = len(self._selected) if self._selected is not None else total
        self.stats = BenchmarkDatasetStats(
            total_rows=sum(family_rows.values()),
            selected_rows=self._length,
            family_rows=family_rows,
            chunk_count=len(self._chunks),
            selected_label_rows=selected_label_rows,
        )
        self._cache: OrderedDict[int, List[Dict[str, str]]] = OrderedDict()

    def __len__(self) -> int:
        return self._length

    def _locate(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._selected is not None:
            return self._selected[index]
        chunk_index = bisect.bisect_right(self._prefix, index)
        previous = 0 if chunk_index == 0 else self._prefix[chunk_index - 1]
        return chunk_index, index - previous

    def _load_chunk(self, chunk_index: int) -> List[Dict[str, str]]:
        if chunk_index in self._cache:
            rows = self._cache.pop(chunk_index)
            self._cache[chunk_index] = rows
            return rows
        _, path, _ = self._chunks[chunk_index]
        rows = _read_chunk_rows(path)
        if self.cache_chunks > 0:
            self._cache[chunk_index] = rows
            while len(self._cache) > self.cache_chunks:
                self._cache.popitem(last=False)
        return rows

    def row_at(self, index: int) -> Dict[str, str]:
        chunk_index, row_index = self._locate(index)
        return dict(self._load_chunk(chunk_index)[row_index])

    def _text_from_row(self, row: Mapping[str, str]) -> str:
        if self.text_field == BenchmarkTextField.AUTO:
            field = "USERNAME" if row.get("DATASET_FAMILY") == BenchmarkFamily.USER_LOGINS.value else "HOSTNAME"
        else:
            field = self.text_field.value
        text = str(row.get(field, ""))
        if self.normalize_text:
            text = normalize_hostname(text)
        return text

    def __getitem__(self, index: int) -> Any:
        row = self.row_at(index)
        text = self._text_from_row(row)
        label_text = resolve_benchmark_label(row, self.label_method)
        label = label_to_int(label_text, unknown_label=self.unknown_label)
        if self.transform is not None:
            text = self.transform(text)
        if self.target_transform is not None:
            label = self.target_transform(label)

        if not self.return_dict:
            if self.include_explanations:
                return text, label, row.get(SONNET_REASON, ""), row.get(OPUS_REASON, "")
            return text, label

        item: Dict[str, Any] = {
            "text": text,
            "label": label,
            "label_text": label_text,
            "row_id": row.get("ROW_ID", ""),
            "family": row.get("DATASET_FAMILY", ""),
            "split": benchmark_row_split(row),
        }
        if self.include_explanations:
            item["gpt_5_5_reason"] = row.get(GPT_5_5_REASON, "")
            item["opus_reason"] = row.get(OPUS_REASON, "")
        if self.include_metadata:
            item["row"] = row
        return item
