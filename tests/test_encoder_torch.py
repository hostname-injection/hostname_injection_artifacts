import numpy as np
from pathlib import Path

from ccd.config import EncoderConfig
from ccd.encoder import CahoEncoder


def test_encode_torch_matches_numpy():
    try:
        import torch
    except Exception:
        return

    checkpoint = Path("caho_model_checkpoint")
    if not checkpoint.exists():
        return

    config = EncoderConfig(model_name=str(checkpoint), device="cpu", fp16=False)
    encoder = CahoEncoder(config)

    hostnames = ["example.com", "login.microsoftonline.com"]
    emb_np = encoder.encode(hostnames, batch_size=2, normalize=True)
    emb_t = encoder.encode_torch(hostnames, batch_size=2, normalize=True)

    emb_t_np = emb_t.detach().cpu().numpy()
    assert emb_np.shape == emb_t_np.shape
    norms = np.linalg.norm(emb_t_np, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(emb_np, emb_t_np, rtol=1e-4, atol=1e-4)
