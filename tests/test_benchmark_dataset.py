import csv
import json

from ccd.benchmark_dataset import (
    BenchmarkFamily,
    BenchmarkLabelMethod,
    HostnameCommandInjectionBenchmarkDataset,
    resolve_benchmark_label,
)
from ccd.benchmark_training import BenchmarkCAHOViewDataset


FIELDNAMES = [
    "ROW_ID",
    "CDB",
    "USERNAME",
    "HOSTNAME",
    "OS",
    "MID",
    "IP_ADDR",
    "CREATED_TIME",
    "ORIGINAL_CREATED_TIME",
    "SUCCESSFUL_LOGIN",
    "LOGIN_PORT",
    "ERROR_CODE",
    "SUCCESSFUL_QUERY",
    "GPT_5_5_IS_DNS_CMD_INJECTION",
    "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE",
    "GPT_5_5_DNS_CMD_INJECTION_REASON",
    "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION",
    "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE",
    "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_REASON",
    "RESOLVED_LABEL_BOTH_M",
    "SINK_FAMILY",
    "LABEL_AGREEMENT",
    "DATASET_FAMILY",
    "CONTENT_TYPE",
    "CONTENT",
    "SOURCE_FILE",
    "SOURCE_ROW_NUMBER",
    "PARSE_STATUS",
    "month",
    "year",
]


def _write_chunk(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _row(row_id, family, text, gpt_5_5, opus, *, hostname=None, username=None, sink_family=""):
    hostname = text if hostname is None and family == "dns_hostnames" else hostname or "10.0.0.1"
    username = text if username is None and family == "user_logins" else username or ""
    content_type = "HOSTNAME" if family == "dns_hostnames" else "USERNAME"
    return {
        "ROW_ID": row_id,
        "CDB": "PRODN_CDB_TEST_ABC",
        "USERNAME": username,
        "HOSTNAME": hostname,
        "OS": "Linux",
        "MID": "1",
        "IP_ADDR": "10.0.0.1",
        "CREATED_TIME": "2025-02-01T00:00:00.000000Z",
        "ORIGINAL_CREATED_TIME": "2025-01-01 00:00:00+00:00",
        "SUCCESSFUL_LOGIN": "True" if family == "user_logins" else "",
        "LOGIN_PORT": "22" if family == "user_logins" else "",
        "ERROR_CODE": "" if family == "user_logins" else "Non-Existent Domain",
        "SUCCESSFUL_QUERY": "" if family == "user_logins" else "False",
        "GPT_5_5_IS_DNS_CMD_INJECTION": gpt_5_5,
        "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE": "0.9",
        "GPT_5_5_DNS_CMD_INJECTION_REASON": f"gpt 5.5 {text}",
        "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION": opus,
        "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE": "0.8",
        "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_REASON": f"opus {text}",
        "RESOLVED_LABEL_BOTH_M": "",
        "SINK_FAMILY": sink_family,
        "LABEL_AGREEMENT": "conflict" if gpt_5_5 != opus else "agree",
        "DATASET_FAMILY": family,
        "CONTENT_TYPE": content_type,
        "CONTENT": text,
        "SOURCE_FILE": "source.csv",
        "SOURCE_ROW_NUMBER": "1",
        "PARSE_STATUS": "ok",
        "month": "02",
        "year": "2025",
    }


def _write_benchmark(root):
    user_rows = [
        _row("user_logins-000000000001", "user_logins", "good-user", "B", "B", hostname="10.0.0.10"),
        _row("user_logins-000000000002", "user_logins", "bad-user", "M", "B", hostname="10.0.0.11", sink_family="command"),
        _row("user_logins-000000000003", "user_logins", "unknown-user", "U", "U", hostname="10.0.0.12"),
    ]
    dns_rows = [
        _row("dns_hostnames-000000000001", "dns_hostnames", "safe.example.com", "B", "B"),
        _row("dns_hostnames-000000000002", "dns_hostnames", "evil.$(id).example", "B", "M", sink_family="query"),
        _row("dns_hostnames-000000000003", "dns_hostnames", "maybe.example", "U", "B"),
    ]
    user_path = root / "data/user_logins/chunks/user_logins_00000.csv"
    dns_path = root / "data/dns_hostnames/chunks/dns_hostnames_00000.csv"
    _write_chunk(user_path, user_rows)
    _write_chunk(dns_path, dns_rows)
    manifest = {
        "schema": FIELDNAMES,
        "datasets": {
            "user_logins": {
                "rows": len(user_rows),
                "chunk_count": 1,
                "chunk_rows_target": 100000,
                "chunks": [{"path": str(user_path.relative_to(root)), "rows": len(user_rows), "bytes": user_path.stat().st_size}],
            },
            "dns_hostnames": {
                "rows": len(dns_rows),
                "chunk_count": 1,
                "chunk_rows_target": 100000,
                "chunks": [{"path": str(dns_path.relative_to(root)), "rows": len(dns_rows), "bytes": dns_path.stat().st_size}],
            },
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_resolve_benchmark_label_policies():
    row = {
        "GPT_5_5_IS_DNS_CMD_INJECTION": "B",
        "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION": "M",
    }
    assert resolve_benchmark_label(row, BenchmarkLabelMethod.GPT_5_5_ONLY) == "B"
    assert resolve_benchmark_label(row, BenchmarkLabelMethod.OPUS_4_8_ONLY) == "M"
    assert resolve_benchmark_label(row, BenchmarkLabelMethod.BOTH_DISAGREE_MALICIOUS) == "M"
    assert resolve_benchmark_label(row, BenchmarkLabelMethod.BOTH_DISAGREE_BENIGN) == "B"
    assert resolve_benchmark_label(row, BenchmarkLabelMethod.BOTH_DISAGREE_UNKNOWN) == "U"
    assert resolve_benchmark_label(row, BenchmarkLabelMethod.ANY_MALICIOUS_ELSE_BENIGN) == "M"
    assert resolve_benchmark_label(
        {**row, "RESOLVED_LABEL_BOTH_M": "B"},
        BenchmarkLabelMethod.RESOLVED_OR_DISAGREE_MALICIOUS,
    ) == "B"
    assert resolve_benchmark_label(
        {**row, "RESOLVED_LABEL_BOTH_M": "U"},
        BenchmarkLabelMethod.RESOLVED_OR_DISAGREE_MALICIOUS,
    ) == "U"

    unknown_row = {
        "GPT_5_5_IS_DNS_CMD_INJECTION": "U",
        "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION": "U",
    }
    assert resolve_benchmark_label(unknown_row, BenchmarkLabelMethod.ANY_MALICIOUS_ELSE_BENIGN) == "B"


def test_benchmark_dataset_loads_selected_families_and_preserves_metadata(tmp_path):
    _write_benchmark(tmp_path)
    ds = HostnameCommandInjectionBenchmarkDataset(
        tmp_path,
        family=BenchmarkFamily.DNS_HOSTNAMES,
        label_method=BenchmarkLabelMethod.BOTH_DISAGREE_MALICIOUS,
        drop_unknown=False,
        include_explanations=True,
        include_metadata=True,
    )
    assert len(ds) == 3
    item = ds[1]
    assert item["text"] == "evil.$(id).example"
    assert item["label"] == 1
    assert item["label_text"] == "M"
    assert item["gpt_5_5_reason"] == "gpt 5.5 evil.$(id).example"
    assert item["opus_reason"] == "opus evil.$(id).example"
    assert item["row"]["HOSTNAME"] == "evil.$(id).example"
    assert item["row"]["CONTENT"] == "evil.$(id).example"
    assert ds.stats.total_rows == 3
    assert ds.stats.selected_rows == 3


def test_benchmark_dataset_drop_unknown_and_tuple_output(tmp_path):
    _write_benchmark(tmp_path)
    ds = HostnameCommandInjectionBenchmarkDataset(
        tmp_path,
        family=BenchmarkFamily.BOTH,
        label_method=BenchmarkLabelMethod.BOTH_DISAGREE_UNKNOWN,
        drop_unknown=True,
        return_dict=False,
    )
    assert len(ds) == 2
    assert ds[0] == ("good-user", 0)
    assert ds[1] == ("safe.example.com", 0)


def test_benchmark_dataset_auto_text_uses_username_for_user_logins(tmp_path):
    _write_benchmark(tmp_path)
    ds = HostnameCommandInjectionBenchmarkDataset(
        tmp_path,
        family=BenchmarkFamily.USER_LOGINS,
        label_method=BenchmarkLabelMethod.BOTH_DISAGREE_MALICIOUS,
        drop_unknown=False,
        include_metadata=True,
    )
    assert ds[0]["text"] == "good-user"
    assert ds[0]["row"]["HOSTNAME"] == "10.0.0.10"
    assert ds[1]["label"] == 1
    assert ds[2]["label"] == -1


def test_benchmark_dataset_any_malicious_else_benign_has_no_unknowns(tmp_path):
    _write_benchmark(tmp_path)
    ds = HostnameCommandInjectionBenchmarkDataset(
        tmp_path,
        family=BenchmarkFamily.BOTH,
        label_method=BenchmarkLabelMethod.ANY_MALICIOUS_ELSE_BENIGN,
        drop_unknown=False,
    )
    assert [ds[i]["label_text"] for i in range(len(ds))] == ["B", "M", "B", "B", "M", "B"]
    assert [ds[i]["label"] for i in range(len(ds))] == [0, 1, 0, 0, 1, 0]


def test_benchmark_caho_training_view_excludes_unresolved_rows_by_default(tmp_path):
    _write_benchmark(tmp_path)
    ds = BenchmarkCAHOViewDataset(tmp_path)

    rows = [ds[i] for i in range(len(ds))]

    assert ds.base.label_method == BenchmarkLabelMethod.RESOLVED_OR_DISAGREE_MALICIOUS
    assert ds.base.drop_unknown is True
    assert ds.base.stats.selected_rows == 4
    assert [row[0] for row in rows] == [
        "good-user",
        "bad-user",
        "safe.example.com",
        "evil.$(id).example",
    ]
    assert [row[2] for row in rows] == [0, 1, 0, 1]
    assert [row[3] for row in rows] == ["", "command", "", "query"]
