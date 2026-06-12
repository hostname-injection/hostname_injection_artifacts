import numpy as np

from baselines.metrics import classification_metrics


def test_classification_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    metrics = classification_metrics(y_true, y_pred)
    assert metrics.tp == 2
    assert metrics.fp == 1
    assert metrics.tn == 1
    assert metrics.fn == 0
    assert np.isclose(metrics.accuracy, 0.75)
    assert np.isclose(metrics.precision, 2 / 3)
    assert np.isclose(metrics.recall, 1.0)
