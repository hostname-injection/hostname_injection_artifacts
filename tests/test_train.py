import random

import pytest

from ccd.augment import AugmentConfig, CAHOAugmenter, WeightedAugmentConfig
from ccd.train import CAHODataset, Sample, pairwise_contrastive_loss, split_input_fn, supervised_orbit_contrastive_loss


def _make_weighted_augmenter() -> CAHOAugmenter:
    weighted = WeightedAugmentConfig(
        num_augs=1,
        benign_weights={"toggle_protocol": 1.0},
        malicious_weights={"toggle_protocol": 1.0},
        retry_on_no_change=True,
        max_attempts=3,
    )
    config = AugmentConfig(
        normalize_input=False,
        use_edit_model=False,
        use_weighted_augs=True,
        weighted=weighted,
    )
    return CAHOAugmenter(config=config)


def test_split_input_fn_chunks():
    chunks = split_input_fn(["a", "b", "c", "d", "e"], chunk_size=2)
    assert chunks == [{"view": ["a", "b"]}, {"view": ["c", "d"]}, {"view": ["e"]}]


def test_weighted_augmenter_changes_hostname():
    augmenter = _make_weighted_augmenter()
    hostname = "example.com"
    out = augmenter.augment(hostname, is_malicious=False, rng=random.Random(0))
    assert out != hostname


def test_caho_dataset_include_original():
    augmenter = _make_weighted_augmenter()
    dataset = CAHODataset(
        [Sample("example.com", is_malicious=False)],
        augmenter=augmenter,
        include_original=True,
    )
    view1, view2, label = dataset[0]
    assert view1 == "example.com"
    assert view2 != "example.com"
    assert label is None


def test_pairwise_contrastive_loss_aligned():
    torch = pytest.importorskip("torch")
    embeddings = torch.eye(2)
    loss = pairwise_contrastive_loss(embeddings, embeddings, temperature=0.1)
    assert loss.item() < 0.01


def test_pairwise_contrastive_loss_misaligned():
    torch = pytest.importorskip("torch")
    embeddings1 = torch.eye(2)
    embeddings2 = torch.flip(embeddings1, dims=[0])
    loss = pairwise_contrastive_loss(embeddings1, embeddings2, temperature=0.1)
    assert loss.item() > 1.0


def test_supervised_orbit_contrastive_loss_backpropagates():
    torch = pytest.importorskip("torch")
    embeddings1 = torch.eye(3, requires_grad=True)
    embeddings2 = torch.eye(3).detach().clone().requires_grad_(True)
    labels = torch.tensor([1, 1, -1], dtype=torch.long)

    loss = supervised_orbit_contrastive_loss(embeddings1, embeddings2, labels, temperature=0.1)
    loss.backward()

    assert loss.item() >= 0.0
    assert embeddings1.grad is not None
    assert embeddings2.grad is not None
