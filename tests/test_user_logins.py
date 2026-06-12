import csv

import ccd.user_logins as user_logins
from ccd.user_logins import (
    LabelPolicy,
    apply_confidence_filter,
    collect_label_stats_from_user_logins,
    normalize_label,
    parse_confidence,
    resolve_label,
)


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


def test_label_helpers():
    assert normalize_label("B") == "B"
    assert normalize_label("malicious") == "M"
    assert normalize_label("") == "U"
    assert parse_confidence("0.9") == 0.9
    assert parse_confidence("bad") is None
    assert apply_confidence_filter("M", 0.5, 0.9) == "U"
    assert apply_confidence_filter("B", 0.95, 0.9) == "B"


def test_resolve_label_policies():
    assert resolve_label("M", "M", LabelPolicy.BOTH_M) == "M"
    assert resolve_label("B", "B", LabelPolicy.BOTH_M) == "B"
    assert resolve_label("M", "B", LabelPolicy.BOTH_M) is None
    assert resolve_label("M", "B", LabelPolicy.EITHER_M) == "M"
    assert resolve_label("B", "U", LabelPolicy.NON_U) == "B"
    assert resolve_label("U", "M", LabelPolicy.NON_U) == "M"
    assert resolve_label("M", "B", LabelPolicy.PREFER_M) == "M"
    assert resolve_label("M", "B", LabelPolicy.PREFER_B) == "B"


def test_collect_label_stats_from_user_logins(tmp_path):
    rows = [
        {
            "USERNAME": "good.com",
            "GPT_5_5_IS_DNS_CMD_INJECTION": "B",
            "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION": "B",
            "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE": "0.9",
            "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE": "0.9",
        },
        {
            "USERNAME": "bad.com",
            "GPT_5_5_IS_DNS_CMD_INJECTION": "M",
            "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION": "M",
            "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE": "0.95",
            "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE": "0.95",
        },
        {
            "USERNAME": "mixed.com",
            "GPT_5_5_IS_DNS_CMD_INJECTION": "B",
            "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION": "M",
            "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE": "0.95",
            "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE": "0.95",
        },
    ]
    _write_csv(tmp_path / "sample.csv", rows)
    stats = collect_label_stats_from_user_logins(tmp_path, label_policy=LabelPolicy.BOTH_M)
    assert stats.total_rows == 3
    assert stats.used_benign == 1
    assert stats.used_malicious == 1
    assert stats.dropped_rows == 1


def test_user_login_only_training_helpers_are_not_exposed():
    assert not hasattr(user_logins, "collect_caho_samples_from_user_logins")
    assert not hasattr(user_logins, "build_priors_from_user_logins")
