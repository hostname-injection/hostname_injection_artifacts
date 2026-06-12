#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


GPT_LABEL = "GPT_5_5_IS_DNS_CMD_INJECTION"
GPT_CONF = "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE"
OPUS_LABEL = "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION"
OPUS_CONF = "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE"

USECOLS = [
    "CDB",
    "OS",
    "LOGIN_PORT",
    "ERROR_CODE",
    "SUCCESSFUL_LOGIN",
    "SUCCESSFUL_QUERY",
    GPT_LABEL,
    GPT_CONF,
    OPUS_LABEL,
    OPUS_CONF,
    "RESOLVED_LABEL_BOTH_M",
    "LABEL_AGREEMENT",
    "DATASET_FAMILY",
    "CONTENT_TYPE",
    "CONTENT",
    "SOURCE_FILE",
    "PARSE_STATUS",
    "month",
    "year",
]

LABELS = ("B", "M", "U")


@dataclass
class NumericSummary:
    count: int = 0
    total: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def update(self, values: pd.Series) -> None:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return
        self.count += int(numeric.size)
        self.total += float(numeric.sum())
        chunk_min = float(numeric.min())
        chunk_max = float(numeric.max())
        self.min_value = chunk_min if self.min_value is None else min(self.min_value, chunk_min)
        self.max_value = chunk_max if self.max_value is None else max(self.max_value, chunk_max)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "mean": self.total / self.count if self.count else None,
            "min": self.min_value,
            "max": self.max_value,
        }


class Profile:
    def __init__(self) -> None:
        self.rows = Counter()
        self.unique_cdb: dict[str, set[str]] = defaultdict(set)
        self.unique_sources: dict[str, set[str]] = defaultdict(set)
        self.counters: dict[str, Counter] = defaultdict(Counter)
        self.cross: dict[str, Counter] = defaultdict(Counter)
        self.numeric: dict[str, NumericSummary] = defaultdict(NumericSummary)

    def add_chunk(self, chunk: pd.DataFrame) -> None:
        chunk = chunk.fillna("")
        families = sorted(str(x) for x in chunk["DATASET_FAMILY"].unique())
        for family in families:
            sub = chunk[chunk["DATASET_FAMILY"] == family]
            self._add_family(family, sub)
            self._add_family("overall", sub)

    def _add_family(self, family: str, df: pd.DataFrame) -> None:
        self.rows[family] += len(df)
        self.unique_cdb[family].update(v for v in df["CDB"].astype(str) if v)
        self.unique_sources[family].update(v for v in df["SOURCE_FILE"].astype(str) if v)

        for col, key in [
            ("SUCCESSFUL_LOGIN", "successful_login"),
            ("SUCCESSFUL_QUERY", "successful_query"),
            (GPT_LABEL, "gpt_5_5_label"),
            (OPUS_LABEL, "opus_4_8_label"),
            ("RESOLVED_LABEL_BOTH_M", "resolved_both_m"),
            ("LABEL_AGREEMENT", "label_agreement"),
            ("PARSE_STATUS", "parse_status"),
            ("CONTENT_TYPE", "content_type"),
            ("SOURCE_FILE", "source_file"),
            ("OS", "os"),
            ("LOGIN_PORT", "login_port"),
            ("ERROR_CODE", "error_code"),
            ("month", "month"),
            ("year", "year"),
        ]:
            self.counters[f"{family}.{key}"].update(df[col].astype(str))

        gpt = df[GPT_LABEL].astype(str)
        opus = df[OPUS_LABEL].astype(str)
        resolved = df["RESOLVED_LABEL_BOTH_M"].astype(str)
        agreement = df["LABEL_AGREEMENT"].astype(str)
        success = success_series(df)
        for key, left, right in [
            ("gpt_by_opus", gpt, opus),
            ("resolved_by_success", resolved, success),
            ("agreement_by_success", agreement, success),
            ("gpt_by_success", gpt, success),
            ("opus_by_success", opus, success),
        ]:
            self.cross[f"{family}.{key}"].update(zip(left, right))

        self.numeric[f"{family}.gpt_5_5_confidence"].update(df[GPT_CONF])
        self.numeric[f"{family}.opus_4_8_confidence"].update(df[OPUS_CONF])

        content_len = df["CONTENT"].astype(str).str.len()
        self.numeric[f"{family}.content_length"].update(content_len)
        contains_metachar = df["CONTENT"].astype(str).str.contains(r"[$`;&|<>]", regex=True)
        metachar_count = int(contains_metachar.sum())
        self.counters[f"{family}.content_contains_shell_metachar"].update({"true": metachar_count})
        self.counters[f"{family}.content_contains_shell_metachar"].update({"false": int(len(df) - metachar_count)})

    def as_dict(self, manifest: dict[str, Any], quality: dict[str, Any], deep_quality: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "benchmark_root": str(manifest_root(manifest)),
            "manifest": {
                "name": manifest.get("name"),
                "version": manifest.get("version"),
                "time_correction": manifest.get("time_correction"),
                "deduplication": manifest.get("deduplication"),
                "partitioning": manifest.get("partitioning"),
                "schema": manifest.get("schema"),
            },
            "rows": dict(self.rows),
            "quality_overall_pass": quality.get("overall_pass"),
            "deep_quality_overall_pass": deep_quality.get("overall_pass"),
            "families": {},
        }
        for family in ["overall", "user_logins", "dns_hostnames"]:
            fam: dict[str, Any] = {
                "rows": self.rows[family],
                "unique_cdb_count": len(self.unique_cdb[family]),
                "unique_source_file_count": len(self.unique_sources[family]),
                "counters": {},
                "cross_tabs": {},
                "numeric": {},
            }
            for key, counter in self.counters.items():
                prefix = f"{family}."
                if key.startswith(prefix):
                    fam["counters"][key[len(prefix) :]] = dict(counter)
            for key, counter in self.cross.items():
                prefix = f"{family}."
                if key.startswith(prefix):
                    fam["cross_tabs"][key[len(prefix) :]] = {f"{a}|{b}": n for (a, b), n in counter.items()}
            for key, summary in self.numeric.items():
                prefix = f"{family}."
                if key.startswith(prefix):
                    fam["numeric"][key[len(prefix) :]] = summary.as_dict()
            fam["cohen_kappa"] = {
                "three_class_B_M_U": cohen_kappa(self.cross[f"{family}.gpt_by_opus"], labels=LABELS),
                "binary_B_M_excluding_any_U": cohen_kappa(
                    Counter({k: v for k, v in self.cross[f"{family}.gpt_by_opus"].items() if k[0] in ("B", "M") and k[1] in ("B", "M")}),
                    labels=("B", "M"),
                ),
            }
            out["families"][family] = fam

        user_cdb = self.unique_cdb["user_logins"]
        dns_cdb = self.unique_cdb["dns_hostnames"]
        out["company_overlap"] = {
            "user_logins_unique_cdb": len(user_cdb),
            "dns_hostnames_unique_cdb": len(dns_cdb),
            "overall_unique_cdb": len(self.unique_cdb["overall"]),
            "cdb_in_both_families": len(user_cdb & dns_cdb),
            "user_logins_only_cdb": len(user_cdb - dns_cdb),
            "dns_hostnames_only_cdb": len(dns_cdb - user_cdb),
        }
        return out


def success_series(df: pd.DataFrame) -> pd.Series:
    login = df["SUCCESSFUL_LOGIN"].astype(str)
    query = df["SUCCESSFUL_QUERY"].astype(str)
    return login.where(login != "", query).replace("", "not_applicable")


def cohen_kappa(matrix: Counter, *, labels: Iterable[str]) -> dict[str, float | int | None]:
    labels = tuple(labels)
    n = sum(matrix.values())
    if n == 0:
        return {"n": 0, "observed_agreement": None, "expected_agreement": None, "kappa": None}
    observed = sum(matrix.get((label, label), 0) for label in labels) / n
    left_totals = {label: sum(count for (left, _right), count in matrix.items() if left == label) for label in labels}
    right_totals = {label: sum(count for (_left, right), count in matrix.items() if right == label) for label in labels}
    expected = sum((left_totals[label] / n) * (right_totals[label] / n) for label in labels)
    kappa = (observed - expected) / (1 - expected) if expected != 1 else None
    return {"n": n, "observed_agreement": observed, "expected_agreement": expected, "kappa": kappa}


def manifest_root(manifest: dict[str, Any]) -> Path:
    return Path(manifest.get("benchmark_root", os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark")))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_chunk_paths(root: Path, manifest: dict[str, Any]) -> Iterable[Path]:
    for family in ["user_logins", "dns_hostnames"]:
        for chunk in manifest["datasets"][family]["chunks"]:
            yield root / chunk["path"]


def pct(n: int, d: int) -> str:
    return f"{(100 * n / d):.4f}%" if d else "n/a"


def fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def top_counter(counter: dict[str, int], n: int = 10, *, drop_blank: bool = True) -> list[tuple[str, int]]:
    items = ((k, v) for k, v in counter.items() if not (drop_blank and k == ""))
    return sorted(items, key=lambda kv: (-kv[1], kv[0]))[:n]


def counts(family: dict[str, Any], key: str) -> dict[str, int]:
    return family.get("counters", {}).get(key, {})


def numeric(family: dict[str, Any], key: str) -> dict[str, float | int | None]:
    return family.get("numeric", {}).get(key, {"count": 0, "mean": None, "min": None, "max": None})


def render_markdown(profile: dict[str, Any]) -> str:
    overall = profile["families"]["overall"]
    user = profile["families"]["user_logins"]
    dns = profile["families"]["dns_hostnames"]
    total = overall["rows"]
    overlap = profile["company_overlap"]

    lines: list[str] = []
    lines.append("# Hostname Command Injection Benchmark Dataset Profile")
    lines.append("")
    lines.append("This profile summarizes the generated benchmark for publication and reproducibility. Counts are exact unless explicitly described as examples or top-k summaries.")
    lines.append("")
    lines.append("## Dataset Identity")
    lines.append("")
    lines.append(table(
        ["Field", "Value"],
        [
            ["Name", profile["manifest"]["name"]],
            ["Version", profile["manifest"]["version"]],
            ["Total rows", fmt_int(total)],
            ["User login attempts", f"{fmt_int(user['rows'])} ({pct(user['rows'], total)})"],
            ["DNS hostname attempts", f"{fmt_int(dns['rows'])} ({pct(dns['rows'], total)})"],
            ["Time range", f"{profile['manifest']['time_correction']['start_utc']} to {profile['manifest']['time_correction']['end_utc']}"],
            ["Partitioning", profile["manifest"]["partitioning"]],
            ["Quality audit pass", profile["quality_overall_pass"]],
            ["Deep quality audit pass", profile["deep_quality_overall_pass"]],
        ],
    ))
    lines.append("")
    lines.append("## Companies / Tenants")
    lines.append("")
    lines.append("`CDB` is treated as the company or tenant identifier.")
    lines.append("")
    lines.append(table(
        ["Metric", "Count"],
        [
            ["Unique CDB values overall", fmt_int(overlap["overall_unique_cdb"])],
            ["Unique CDB values in user logins", fmt_int(overlap["user_logins_unique_cdb"])],
            ["Unique CDB values in DNS hostnames", fmt_int(overlap["dns_hostnames_unique_cdb"])],
            ["CDB values present in both families", fmt_int(overlap["cdb_in_both_families"])],
            ["CDB values only in user logins", fmt_int(overlap["user_logins_only_cdb"])],
            ["CDB values only in DNS hostnames", fmt_int(overlap["dns_hostnames_only_cdb"])],
        ],
    ))
    lines.append("")
    lines.append("## Attempt Outcomes")
    lines.append("")
    lines.append(table(
        ["Family", "Outcome column", "True", "False", "Blank / N.A."],
        [
            ["User logins", "SUCCESSFUL_LOGIN", fmt_int(counts(user, "successful_login").get("True", 0)), fmt_int(counts(user, "successful_login").get("False", 0)), fmt_int(counts(user, "successful_login").get("", 0))],
            ["DNS hostnames", "SUCCESSFUL_QUERY", fmt_int(counts(dns, "successful_query").get("True", 0)), fmt_int(counts(dns, "successful_query").get("False", 0)), fmt_int(counts(dns, "successful_query").get("", 0))],
        ],
    ))
    lines.append("")
    lines.append("## Label Distribution")
    lines.append("")
    label_rows = []
    for fam_name, fam in [("Overall", overall), ("User logins", user), ("DNS hostnames", dns)]:
        rows = fam["rows"]
        for label_name, key in [("GPT-5.5", "gpt_5_5_label"), ("Claude Opus 4.8", "opus_4_8_label"), ("Resolved both-M policy", "resolved_both_m")]:
            label_counts = counts(fam, key)
            label_rows.append([
                fam_name,
                label_name,
                f"{fmt_int(label_counts.get('B', 0))} ({pct(label_counts.get('B', 0), rows)})",
                f"{fmt_int(label_counts.get('M', 0))} ({pct(label_counts.get('M', 0), rows)})",
                f"{fmt_int(label_counts.get('U', 0) + label_counts.get('', 0))} ({pct(label_counts.get('U', 0) + label_counts.get('', 0), rows)})",
            ])
    lines.append(table(["Family", "Label source", "Benign", "Malicious", "Unsure / unresolved"], label_rows))
    lines.append("")
    lines.append("## Annotator Agreement")
    lines.append("")
    kappa_rows = []
    for fam_name, fam in [("Overall", overall), ("User logins", user), ("DNS hostnames", dns)]:
        for metric_name, key in [("B/M/U", "three_class_B_M_U"), ("B/M excluding any U", "binary_B_M_excluding_any_U")]:
            metric = fam["cohen_kappa"][key]
            kappa_rows.append([
                fam_name,
                metric_name,
                fmt_int(metric["n"]),
                fmt_float(metric["observed_agreement"], 6),
                fmt_float(metric["expected_agreement"], 6),
                fmt_float(metric["kappa"], 6),
            ])
    lines.append(table(["Family", "Kappa basis", "N", "Observed agreement", "Expected agreement", "Cohen's kappa"], kappa_rows))
    lines.append("")
    lines.append("Label agreement column counts:")
    lines.append("")
    lines.append(table(
        ["Family", "Agree", "Conflict", "Unknown"],
        [
            ["Overall", fmt_int(counts(overall, "label_agreement").get("agree", 0)), fmt_int(counts(overall, "label_agreement").get("conflict", 0)), fmt_int(counts(overall, "label_agreement").get("unknown", 0))],
            ["User logins", fmt_int(counts(user, "label_agreement").get("agree", 0)), fmt_int(counts(user, "label_agreement").get("conflict", 0)), fmt_int(counts(user, "label_agreement").get("unknown", 0))],
            ["DNS hostnames", fmt_int(counts(dns, "label_agreement").get("agree", 0)), fmt_int(counts(dns, "label_agreement").get("conflict", 0)), fmt_int(counts(dns, "label_agreement").get("unknown", 0))],
        ],
    ))
    lines.append("")
    lines.append("## Quality And Repairs")
    lines.append("")
    lines.append(table(
        ["Family", "Parse status", "Rows"],
        [[fam_name, status or "blank", fmt_int(count)] for fam_name, fam in [("Overall", overall), ("User logins", user), ("DNS hostnames", dns)] for status, count in sorted(counts(fam, "parse_status").items())],
    ))
    lines.append("")
    lines.append("## Confidence And Content")
    lines.append("")
    lines.append(table(
        ["Family", "Metric", "Count", "Mean", "Min", "Max"],
        [
            [fam_name, metric, fmt_int(summary["count"]), fmt_float(summary["mean"], 6), fmt_float(summary["min"], 3), fmt_float(summary["max"], 3)]
            for fam_name, fam in [("Overall", overall), ("User logins", user), ("DNS hostnames", dns)]
            for metric, summary in [
                ("GPT-5.5 confidence", numeric(fam, "gpt_5_5_confidence")),
                ("Claude Opus 4.8 confidence", numeric(fam, "opus_4_8_confidence")),
                ("Content length", numeric(fam, "content_length")),
            ]
        ],
    ))
    lines.append("")
    lines.append("Rows whose `CONTENT` contains common shell metacharacters (`$`, backtick, `;`, `&`, `|`, `<`, `>`):")
    lines.append("")
    lines.append(table(
        ["Family", "Rows", "Share"],
        [
            [fam_name, fmt_int(counts(fam, "content_contains_shell_metachar").get("true", 0)), pct(counts(fam, "content_contains_shell_metachar").get("true", 0), fam["rows"])]
            for fam_name, fam in [("Overall", overall), ("User logins", user), ("DNS hostnames", dns)]
        ],
    ))
    lines.append("")
    lines.append("## Operational Distributions")
    lines.append("")
    for fam_name, fam in [("User logins", user), ("DNS hostnames", dns)]:
        lines.append(f"### {fam_name}")
        lines.append("")
        for title, key in [("Top operating systems", "os"), ("Top login ports", "login_port"), ("Top error codes", "error_code"), ("Top source files", "source_file")]:
            counter = counts(fam, key)
            rows = top_counter(counter, 10)
            if rows:
                lines.append(title + ":")
                lines.append("")
                lines.append(table(["Value", "Rows"], [[value, fmt_int(count)] for value, count in rows]))
                lines.append("")
    lines.append("## Temporal Distribution")
    lines.append("")
    lines.append(table(
        ["Family", "Year", "Rows"],
        [[fam_name, year or "blank", fmt_int(count)] for fam_name, fam in [("Overall", overall), ("User logins", user), ("DNS hostnames", dns)] for year, count in sorted(counts(fam, "year").items())],
    ))
    lines.append("")
    lines.append("## Notes For Publication")
    lines.append("")
    lines.append("- The benchmark is highly class-imbalanced: malicious labels are well under 1% of rows under both model-specific and resolved policies.")
    lines.append("- The generated benchmark intentionally includes no train/test/evaluation split; publications should describe their downstream split strategy and prevent temporal or tenant leakage where appropriate.")
    lines.append("- `CREATED_TIME` is uniformly reassigned over the known sampling interval; use `ORIGINAL_CREATED_TIME` if source-time provenance is needed.")
    lines.append("- `RESOLVED_LABEL_BOTH_M` labels a row malicious only when both annotators marked it malicious; unresolved blanks correspond to disagreement or unknown labels.")
    lines.append("- Report both model-specific labels and resolved labels. Kappa should be interpreted with class imbalance in mind because very high observed agreement can coexist with modest kappa.")
    lines.append("- Source CSV repair counts should be disclosed because a small number of DNS rows required multiline or overwide-row reconstruction.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the generated benchmark dataset for publication.")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark")))
    parser.add_argument("--out-md", type=Path, default=Path("benchmark_dataset_profile.md"))
    parser.add_argument("--out-json", type=Path, default=Path("benchmark_dataset_profile.json"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-chunks", type=int, default=None, help="Debug only: stop after this many manifest chunks.")
    args = parser.parse_args()

    manifest = load_json(args.root / "manifest.json")
    quality = load_json(args.root / "quality_report.json")
    deep_quality = load_json(args.root / "deep_quality_report.json")
    profile = Profile()

    paths = list(iter_chunk_paths(args.root, manifest))
    if args.max_chunks is not None:
        paths = paths[: args.max_chunks]
    for index, path in enumerate(paths, start=1):
        for chunk in pd.read_csv(path, usecols=USECOLS, dtype=str, chunksize=args.chunksize, keep_default_na=False):
            profile.add_chunk(chunk)
        if index % 50 == 0:
            print(f"profiled_chunks={index}/{len(paths)} rows={sum(profile.rows[f] for f in ['user_logins', 'dns_hostnames'])}", flush=True)

    data = profile.as_dict(manifest, quality, deep_quality)
    args.out_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
