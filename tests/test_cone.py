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
