import numpy as np

from ccd.cone import ConePartition
from ccd.config import ConeConfig
from ccd.scoring import ccd_scores, ccd_scores_torch
from ccd.utils import stable_log


def test_ccd_scores_torch_matches_numpy():
    try:
        import torch
    except Exception:
        return

    config = ConeConfig(dim=4, num_cones=16, active_cones=4, use_lsh=False, seed=7)
    cones = ConePartition.build(config)

    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((5, 4)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    benign_prior = rng.random(16).astype(np.float32) + 1e-6
    benign_prior /= benign_prior.sum()
    malicious_prior = rng.random(16).astype(np.float32) + 1e-6
    malicious_prior /= malicious_prior.sum()

    scores_np = ccd_scores(embeddings, cones, benign_prior, {"m": malicious_prior})

    scores_t = ccd_scores_torch(
        torch.tensor(embeddings),
        torch.tensor(cones.axes),
        torch.tensor(stable_log(benign_prior)),
        torch.tensor(np.stack([stable_log(malicious_prior)], axis=0)),
        config,
    ).detach().cpu().numpy()

    np.testing.assert_allclose(scores_np, scores_t, rtol=1e-5, atol=1e-5)


def test_ccd_scores_torch_normalizes_inputs():
    try:
        import torch
    except Exception:
        return

    config = ConeConfig(dim=4, num_cones=16, active_cones=4, use_lsh=False, seed=19)
    cones = ConePartition.build(config)

    rng = np.random.default_rng(19)
    embeddings = rng.standard_normal((5, 4)).astype(np.float32)
    unit_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scaled_embeddings = unit_embeddings * np.array([[0.5], [2.0], [3.0], [0.75], [1.5]], dtype=np.float32)

    benign_prior = rng.random(16).astype(np.float32) + 1e-6
    benign_prior /= benign_prior.sum()
    malicious_prior = rng.random(16).astype(np.float32) + 1e-6
    malicious_prior /= malicious_prior.sum()

    scores_np = ccd_scores(unit_embeddings, cones, benign_prior, {"m": malicious_prior})
    scores_t = ccd_scores_torch(
        torch.tensor(scaled_embeddings),
        torch.tensor(cones.axes),
        torch.tensor(stable_log(benign_prior)),
        torch.tensor(np.stack([stable_log(malicious_prior)], axis=0)),
        config,
    ).detach().cpu().numpy()

    np.testing.assert_allclose(scores_np, scores_t, rtol=1e-5, atol=1e-5)
