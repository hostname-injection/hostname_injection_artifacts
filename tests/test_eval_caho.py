from pathlib import Path

import numpy as np

import ccd.cli as cli_module
from ccd.encoder import LOCAL_HASH_ENCODER


def test_eval_caho_checkpoint_npz(tmp_path: Path):
    input_path = tmp_path / "hosts.txt"
    input_path.write_text("example.com\nmalicious.example.com\n")
    output_path = tmp_path / "embeddings.npz"

    args = type(
        "Args",
        (),
        {
            "model": LOCAL_HASH_ENCODER,
            "input": input_path,
            "output": output_path,
            "format": "npz",
            "batch_size": 2,
            "device": "cpu",
            "normalize": False,
            "embed_normalize": True,
            "_allow_test_encoder": True,
        },
    )()

    cli_module.cmd_eval_caho(args)
    data = np.load(output_path, allow_pickle=True)
    embeddings = data["embeddings"]
    hostnames = data["hostnames"].tolist()
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384
    assert hostnames == ["example.com", "malicious.example.com"]
