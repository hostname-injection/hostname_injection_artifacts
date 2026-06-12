from baselines.models.registry import list_baselines


def test_baseline_registry_contains_core():
    names = {spec.name for spec in list_baselines()}
    expected = {
        "tfidf-logreg-char4",
        "tfidf-svm-char3",
        "markov-char3",
        "char-cnn",
        "urlnet",
        "urlbert",
        "knn-density",
        "mahalanobis",
        "deep-sad",
        "deep-svdd",
        "deep-one-class",
        "t-mahalanobis",
        "drocc",
    }
    assert expected.issubset(names)
