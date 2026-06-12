import numpy as np

from ccd.calibration import (
    calibrate_threshold,
    calibrate_thresholds_by_group,
    split_conformal_threshold_metadata,
    threshold_for_group,
)


def test_calibrate_threshold():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    t = calibrate_threshold(scores, alpha=0.2)
    # ceil((1-0.2)(5+1)) = ceil(4.8) = 5 => 5th order statistic
    assert t == 0.5


def test_calibrate_threshold_requires_scores():
    try:
        calibrate_threshold(np.array([]), alpha=0.2)
    except ValueError as exc:
        assert "benign_scores" in str(exc)
        return

    assert False, "Expected ValueError for empty calibration set"


def test_calibrate_threshold_requires_valid_alpha():
    for alpha in (0.0, 1.0, -0.1):
        try:
            calibrate_threshold(np.array([0.1, 0.2]), alpha=alpha)
        except ValueError as exc:
            assert "alpha" in str(exc)
            continue

        assert False, f"Expected ValueError for alpha={alpha}"


def test_split_conformal_threshold_metadata_records_rank_and_rule():
    metadata = split_conformal_threshold_metadata(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), alpha=0.2)

    assert metadata["threshold"] == 0.5
    assert metadata["num_samples"] == 5
    assert metadata["order_statistic_rank"] == 5
    assert metadata["order_statistic_formula"] == "ceil((1-alpha)*(n+1)) clipped to [1,n]"
    assert metadata["score_order"] == "ascending"
    assert metadata["decision_rule"] == "score > threshold"
    assert metadata["calibration_scores"] == "benign_only"


def test_calibrate_thresholds_by_group():
    scores = np.array([0.1, 0.4, 0.2, 0.8], dtype=np.float32)
    grouped = calibrate_thresholds_by_group(scores, ["tenant-a", "tenant-a", "tenant-b", "tenant-b"], alpha=0.5)

    assert abs(grouped["tenant-a"]["threshold"] - 0.4) < 1e-6
    assert grouped["tenant-a"]["num_samples"] == 2
    assert grouped["tenant-a"]["order_statistic_rank"] == 2
    assert grouped["tenant-a"]["decision_rule"] == "score > threshold"
    assert abs(grouped["tenant-b"]["threshold"] - 0.8) < 1e-6


def test_calibrate_thresholds_by_group_rejects_mismatched_lengths():
    try:
        calibrate_thresholds_by_group(np.array([0.1, 0.2]), ["tenant-a"], alpha=0.5)
    except ValueError as exc:
        assert "same length" in str(exc)
        return

    assert False, "Expected ValueError for mismatched score/group lengths"


def test_threshold_for_group_uses_grouped_threshold_then_default():
    grouped = {
        "tenant-a": {"threshold": 0.7},
        "tenant-b": 0.4,
    }

    assert threshold_for_group("tenant-a", 0.5, grouped) == 0.7
    assert threshold_for_group("tenant-b", 0.5, grouped) == 0.4
    assert threshold_for_group("tenant-c", 0.5, grouped) == 0.5

    try:
        threshold_for_group("tenant-c", 0.5, grouped, missing="error")
    except KeyError as exc:
        assert "tenant-c" in str(exc)
        return

    assert False, "Expected KeyError for missing grouped threshold"
