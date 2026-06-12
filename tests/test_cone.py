import numpy as np

from ccd.config import ConeConfig
from ccd.cone import ConePartition


def test_cone_sketch_is_sparse():
    config = ConeConfig(dim=4, num_cones=16, active_cones=4, use_lsh=False)
    cones = ConePartition.build(config)
    u = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    idx, weights = cones.cone_sketch(u)
    assert len(idx) == config.active_cones
    assert abs(weights.sum() - 1.0) < 1e-6


def test_cone_partition_rejects_invalid_config_and_axes():
    invalid_calls = [
        lambda: ConePartition.build(ConeConfig(dim=0, num_cones=2, active_cones=1, use_lsh=False)),
        lambda: ConePartition.build(ConeConfig(dim=2, num_cones=2, active_cones=3, use_lsh=False)),
        lambda: ConePartition.build(ConeConfig(dim=2, num_cones=2, active_cones=1, temperature=float("nan"), use_lsh=False)),
        lambda: ConePartition.build(
            ConeConfig(dim=2, num_cones=2, active_cones=1, use_lsh=False),
            axes=np.array([[1.0, 0.0], [float("nan"), 1.0]], dtype=np.float32),
        ),
        lambda: ConePartition.build(
            ConeConfig(dim=2, num_cones=2, active_cones=1, use_lsh=False),
            axes=np.zeros((2, 2), dtype=np.float32),
        ),
        lambda: ConePartition.build(
            ConeConfig(dim=2, num_cones=2, active_cones=1, use_lsh=False),
            axes=np.eye(3, dtype=np.float32),
        ),
    ]

    for call in invalid_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("Expected invalid cone configuration or axes to fail")
