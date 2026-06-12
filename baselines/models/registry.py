from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import importlib

from .base import BaselineModel


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    description: str
    factory: Callable[..., BaselineModel]
    supervised: bool = True
    needs_download: bool = False


def _lazy_factory(module: str, cls_name: str, **preset):
    def _factory(**kwargs):
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        return cls(**{**preset, **kwargs})

    return _factory


BASELINE_SPECS: Dict[str, BaselineSpec] = {
    "tfidf-logreg-char4": BaselineSpec(
        name="tfidf-logreg-char4",
        description="Logistic regression on char-4 TF-IDF.",
        factory=_lazy_factory("baselines.models.sklearn_tfidf", "TfidfClassifier", model_type="logreg", ngram_range=(4, 4)),
    ),
    "tfidf-logreg-char3": BaselineSpec(
        name="tfidf-logreg-char3",
        description="Logistic regression on char-3 TF-IDF.",
        factory=_lazy_factory("baselines.models.sklearn_tfidf", "TfidfClassifier", model_type="logreg", ngram_range=(3, 3)),
    ),
    "tfidf-svm-char3": BaselineSpec(
        name="tfidf-svm-char3",
        description="Linear SVM on char-3 TF-IDF.",
        factory=_lazy_factory("baselines.models.sklearn_tfidf", "TfidfClassifier", model_type="linear_svm", ngram_range=(3, 3)),
    ),
    "tfidf-ocsvm-char3": BaselineSpec(
        name="tfidf-ocsvm-char3",
        description="One-class SVM (RBF) on char-3 TF-IDF.",
        factory=_lazy_factory("baselines.models.sklearn_tfidf", "TfidfClassifier", model_type="ocsvm", ngram_range=(3, 3)),
        supervised=False,
    ),
    "tfidf-rf-char4": BaselineSpec(
        name="tfidf-rf-char4",
        description="Random Forest on char-4 TF-IDF.",
        factory=_lazy_factory("baselines.models.sklearn_tfidf", "TfidfClassifier", model_type="random_forest", ngram_range=(4, 4)),
    ),
    "tfidf-et-char3": BaselineSpec(
        name="tfidf-et-char3",
        description="ExtraTrees on char-3 TF-IDF.",
        factory=_lazy_factory("baselines.models.sklearn_tfidf", "TfidfClassifier", model_type="extra_trees", ngram_range=(3, 3)),
    ),
    "tfidf-iforest-char4": BaselineSpec(
        name="tfidf-iforest-char4",
        description="Isolation Forest on char-4 TF-IDF.",
        factory=_lazy_factory("baselines.models.sklearn_tfidf", "TfidfClassifier", model_type="isolation_forest", ngram_range=(4, 4)),
        supervised=False,
    ),
    "tfidf-xgb-char4": BaselineSpec(
        name="tfidf-xgb-char4",
        description="XGBoost on char-4 TF-IDF.",
        factory=_lazy_factory("baselines.models.sklearn_tfidf", "TfidfClassifier", model_type="xgboost", ngram_range=(4, 4)),
    ),
    "markov-char3": BaselineSpec(
        name="markov-char3",
        description="Character 3-gram Markov likelihood ratio.",
        factory=_lazy_factory("baselines.models.markov", "MarkovBaseline", n=3),
    ),
    "char-cnn": BaselineSpec(
        name="char-cnn",
        description="Character-level CNN classifier.",
        factory=_lazy_factory("baselines.models.char_cnn", "CharCNNBaseline"),
    ),
    "urlnet": BaselineSpec(
        name="urlnet",
        description="URLNet-style char+token CNN.",
        factory=_lazy_factory("baselines.models.urlnet", "URLNetBaseline"),
    ),
    "urlbert": BaselineSpec(
        name="urlbert",
        description="URLBERT transformer classifier.",
        factory=_lazy_factory("baselines.models.urlbert", "URLBERTBaseline"),
        needs_download=True,
    ),
    "csi": BaselineSpec(
        name="csi",
        description="Contrastive self-supervised encoder + linear head.",
        factory=_lazy_factory("baselines.models.csi", "CSIBaseline"),
        needs_download=True,
    ),
    "knn-density": BaselineSpec(
        name="knn-density",
        description="kNN density on CAHO embeddings.",
        factory=_lazy_factory("baselines.models.embedding", "KNNAnomalyBaseline"),
        supervised=False,
    ),
    "mahalanobis": BaselineSpec(
        name="mahalanobis",
        description="Mahalanobis distance on CAHO embeddings.",
        factory=_lazy_factory("baselines.models.embedding", "MahalanobisBaseline"),
        supervised=False,
    ),
    "deep-sad": BaselineSpec(
        name="deep-sad",
        description="Deep SAD on CAHO embeddings.",
        factory=_lazy_factory("baselines.models.deepsad", "DeepSADBaseline"),
        supervised=False,
    ),
    "deep-svdd": BaselineSpec(
        name="deep-svdd",
        description="Deep SVDD (Deep One-Class) on CAHO embeddings.",
        factory=_lazy_factory("baselines.models.deepsvdd", "DeepSVDD"),
        supervised=False,
    ),
    "deep-one-class": BaselineSpec(
        name="deep-one-class",
        description="Deep One-Class (alias of Deep SVDD) on CAHO embeddings.",
        factory=_lazy_factory("baselines.models.deepsvdd", "DeepOneClass"),
        supervised=False,
    ),
    "drocc": BaselineSpec(
        name="drocc",
        description="DROCC-style adversarial one-class classifier.",
        factory=_lazy_factory("baselines.models.drocc", "DROCCBaseline"),
        supervised=False,
    ),
    "t-mahalanobis": BaselineSpec(
        name="t-mahalanobis",
        description="Transformer embeddings + Mahalanobis distance (paper: T+Mahalanobis).",
        factory=_lazy_factory("baselines.models.embedding", "MahalanobisBaseline"),
        supervised=False,
    ),
}


def list_baselines() -> List[BaselineSpec]:
    return [BASELINE_SPECS[name] for name in sorted(BASELINE_SPECS.keys())]


def get_baseline(name: str, **kwargs) -> BaselineModel:
    if name not in BASELINE_SPECS:
        raise KeyError(f"Unknown baseline: {name}")
    return BASELINE_SPECS[name].factory(**kwargs)
