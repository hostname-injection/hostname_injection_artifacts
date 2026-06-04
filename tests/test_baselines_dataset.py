import csv
import json

from baselines.dataset import load_benchmark_dataset


FIELDNAMES = [
    "ROW_ID",
    "DATASET_FAMILY",
    "CONTENT",
    "USERNAME",
    "HOSTNAME",
    "GPT_5_5_IS_DNS_CMD_INJECTION",
    "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION",
]


def _row(row_id, family, text, label):
    return {
        "ROW_ID": row_id,
        "DATASET_FAMILY": family,
        "CONTENT": text,
        "USERNAME": text if family == "user_logins" else "",
        "HOSTNAME": text if family == "dns_hostnames" else "",
        "GPT_5_5_IS_DNS_CMD_INJECTION": label,
        "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION": label,
    }


def _write_chunk(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_benchmark(root, user_rows, dns_rows):
    user_path = root / "data/user_logins/chunks/user_logins_00000.csv"
    dns_path = root / "data/dns_hostnames/chunks/dns_hostnames_00000.csv"
    _write_chunk(user_path, user_rows)
    _write_chunk(dns_path, dns_rows)
    manifest = {
        "datasets": {
            "user_logins": {
                "rows": len(user_rows),
                "chunks": [{"path": str(user_path.relative_to(root)), "rows": len(user_rows)}],
            },
            "dns_hostnames": {
                "rows": len(dns_rows),
                "chunks": [{"path": str(dns_path.relative_to(root)), "rows": len(dns_rows)}],
            },
        }
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_load_benchmark_dataset_uses_all_families(tmp_path):
    _write_benchmark(
        tmp_path,
        [
            _row("u1", "user_logins", "good-user", "B"),
            _row("u2", "user_logins", "bad-user", "M"),
        ],
        [
            _row("d1", "dns_hostnames", "good.example", "B"),
            _row("d2", "dns_hostnames", "bad.example", "M"),
        ],
    )

    texts, labels, stats = load_benchmark_dataset(tmp_path, sample_per_class=None, deduplicate=True)

    assert stats.total_rows == 4
    assert stats.used_rows == 4
    assert stats.family_rows == {"user_logins": 2, "dns_hostnames": 2}
    assert stats.benign == 2
    assert stats.malicious == 2
    assert set(texts) == {"good-user", "bad-user", "good.example", "bad.example"}
    assert set(labels) == {0, 1}


def test_sample_per_class(tmp_path):
    user_rows = []
    dns_rows = []
    for i in range(5):
        user_rows.append(_row(f"u-good-{i}", "user_logins", f"good-user-{i}", "B"))
        user_rows.append(_row(f"u-bad-{i}", "user_logins", f"bad-user-{i}", "M"))
        dns_rows.append(_row(f"d-good-{i}", "dns_hostnames", f"good-{i}.example", "B"))
        dns_rows.append(_row(f"d-bad-{i}", "dns_hostnames", f"bad-{i}.example", "M"))
    _write_benchmark(tmp_path, user_rows, dns_rows)

    texts, labels, stats = load_benchmark_dataset(tmp_path, sample_per_class=2, seed=7)

    assert len(texts) == 4
    assert stats.benign == 2
    assert stats.malicious == 2
    assert set(labels) == {0, 1}
