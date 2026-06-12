import random

import pytest

from ccd.train import ContrastiveTrainer


class _TinyDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class DummyModelFactory:
    @staticmethod
    def build(dim=8):
        torch = pytest.importorskip("torch")

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(256, dim)
                self.proj = torch.nn.Linear(dim, dim, bias=False)

            def tokenize(self, texts, **kwargs):
                ids = [min(255, len(t)) for t in texts]
                input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(1)
                return {"input_ids": input_ids}

            def forward(self, tokenized):
                ids = tokenized["input_ids"]
                emb = self.embed(ids).mean(dim=1)
                return {"sentence_embedding": self.proj(emb)}

        return DummyModel()


def test_grad_cache_training_path():
    pytest.importorskip("grad_cache")
    torch = pytest.importorskip("torch")
    rng = random.Random(0)

    dataset = _TinyDataset(
        [
            ("alpha.com", "beta.com", None),
            ("gamma.com", "delta.com", None),
            ("epsilon.com", "zeta.com", None),
            ("eta.com", "theta.com", None),
        ]
    )
    model = DummyModelFactory.build(dim=8)
    model.train()
    before = [p.detach().clone() for p in model.parameters()]

    trainer = ContrastiveTrainer(
        model,
        batch_size=2,
        temperature=0.1,
        lr=1e-2,
        max_grad_norm=1.0,
        scheduler="none",
        min_lr=1e-5,
        use_grad_cache=True,
        grad_cache_chunk_size=2,
    )
    trainer.fit(dataset, epochs=1)

    after = list(model.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert changed
