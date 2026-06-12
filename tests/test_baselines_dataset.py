import csv

from baselines.dataset import load_user_logins_dataset
from ccd.user_logins import LabelPolicy


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "USERNAME",
                "GPT_5_5_IS_DNS_CMD_INJECTION",
                "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION",
                "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE",
                "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_user_logins_dataset(tmp_path):
    rows = [
        {
            "USERNAME": "good-user",
            "GPT_5_5_IS_DNS_CMD_INJECTION": "B",
            "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION": "B",
            "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE": "0.9",
            "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE": "0.9",
        },
        {
            "USERNAME": "bad-user",
            "GPT_5_5_IS_DNS_CMD_INJECTION": "M",
            "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION": "M",
            "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE": "0.95",
            "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE": "0.95",
        },
        {
            "USERNAME": "maybe-user",
            "GPT_5_5_IS_DNS_CMD_INJECTION": "U",
            "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION": "U",
            "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE": "0.5",
            "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE": "0.5",
        },
    ]
    _write_csv(tmp_path / "sample.csv", rows)

    texts, labels, stats = load_user_logins_dataset(
        tmp_path,
        label_policy=LabelPolicy.BOTH_M,
        sample_per_class=None,
        deduplicate=True,
    )

    assert stats.total_rows == 2  # only B/M rows are labeled
    assert stats.benign == 1
    assert stats.malicious == 1
    assert set(texts) == {"good-user", "bad-user"}
    assert set(labels) == {0, 1}


def test_sample_per_class(tmp_path):
    rows = []
    for i in range(5):
        rows.append(
            {
                "USERNAME": f"good{i}",
                "GPT_5_5_IS_DNS_CMD_INJECTION": "B",
                "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION": "B",
                "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE": "0.9",
                "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE": "0.9",
            }
        )
        rows.append(
            {
                "USERNAME": f"bad{i}",
                "GPT_5_5_IS_DNS_CMD_INJECTION": "M",
                "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION": "M",
                "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE": "0.9",
                "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE": "0.9",
            }
        )
    _write_csv(tmp_path / "sample.csv", rows)

    texts, labels, stats = load_user_logins_dataset(
        tmp_path,
        label_policy=LabelPolicy.BOTH_M,
        sample_per_class=2,
        seed=7,
    )

    assert len(texts) == 4
    assert stats.benign == 2
    assert stats.malicious == 2
    assert set(labels) == {0, 1}
