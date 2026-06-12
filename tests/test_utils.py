import numpy as np

from ccd.utils import l2_normalize, softmax, stable_log, topk_indices


def test_l2_normalize_unit_norm():
    x = np.array([[3.0, 4.0]], dtype=np.float32)
    normed = l2_normalize(x, axis=1)
    assert np.allclose(np.linalg.norm(normed, axis=1), 1.0)


def test_l2_normalize_zero_vector():
    x = np.array([[0.0, 0.0]], dtype=np.float32)
    normed = l2_normalize(x, axis=1)
    assert np.allclose(normed, x)


def test_softmax_sums_to_one():
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    sm = softmax(x, axis=1)
    assert np.allclose(sm.sum(axis=1), 1.0)


def test_topk_indices_order():
    values = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    idx = topk_indices(values, k=2)
    assert idx.tolist() == [1, 2]


def test_stable_log_eps():
    x = np.array([0.0, 1.0], dtype=np.float32)
    logs = stable_log(x)
    assert logs[0] < logs[1]
