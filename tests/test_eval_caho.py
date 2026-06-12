from pathlib import Path

import numpy as np
import pytest

from ccd.cli import cmd_eval_caho


def test_eval_caho_checkpoint_npz(tmp_path: Path):
    pytest.importorskip("sentence_transformers")
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "caho_model_checkpoint"
    if not model_path.exists():
        pytest.skip("caho_model_checkpoint not found")

    input_path = tmp_path / "hosts.txt"
    input_path.write_text("example.com\nmalicious.example.com\n")
    output_path = tmp_path / "embeddings.npz"

    args = type(
        "Args",
        (),
        {
            "model": str(model_path),
            "input": input_path,
            "output": output_path,
            "format": "npz",
            "batch_size": 2,
            "device": "cpu",
            "normalize": False,
            "embed_normalize": True,
        },
    )()

    cmd_eval_caho(args)
    data = np.load(output_path, allow_pickle=True)
    embeddings = data["embeddings"]
    hostnames = data["hostnames"].tolist()
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384
    assert hostnames == ["example.com", "malicious.example.com"]
