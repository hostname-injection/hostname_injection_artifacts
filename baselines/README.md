# Baselines

This folder provides baseline implementations referenced in *IEEE_S_P_Hostnames.pdf*.
The goal is reproducible training + evaluation for a broad set of classical, neural, and
embedding-based baselines on the `user_logins` dataset.

## Baselines Included

- `tfidf-logreg-char4`: Logistic regression on char-4 TF-IDF.
- `tfidf-logreg-char3`: Logistic regression on char-3 TF-IDF.
- `tfidf-svm-char3`: Linear SVM on char-3 TF-IDF.
- `tfidf-ocsvm-char3`: One-class SVM (RBF) on char-3 TF-IDF.
- `tfidf-rf-char4`: Random Forest on char-4 TF-IDF.
- `tfidf-et-char3`: ExtraTrees on char-3 TF-IDF.
- `tfidf-iforest-char4`: Isolation Forest on char-4 TF-IDF.
- `tfidf-xgb-char4`: XGBoost on char-4 TF-IDF.
- `markov-char3`: Character 3-gram Markov likelihood ratio.
- `char-cnn`: Character-level CNN classifier.
- `urlnet`: URLNet-style char+token CNN.
- `urlbert`: URLBERT transformer classifier (configurable HF model).
- `csi`: Contrastive self-supervised encoder + linear head.
- `knn-density`: kNN density on CAHO embeddings.
- `mahalanobis`: Mahalanobis distance on CAHO embeddings.
- `t-mahalanobis`: Transformer embeddings + Mahalanobis distance (paper: "T+Mahalanobis").
- `deep-sad`: Deep SAD on CAHO embeddings.
- `deep-svdd`: Deep SVDD (Deep One-Class) on CAHO embeddings.
- `deep-one-class`: Alias of Deep SVDD, included because the paper lists "Deep One-Class".
- `drocc`: DROCC-style adversarial one-class classifier.

## Paper-to-Code Mapping

| Paper Baseline Name | Baseline ID in This Repo |
| --- | --- |
| URLNet | `urlnet` |
| urlBERT | `urlbert` |
| CSI | `csi` |
| Char-CNN | `char-cnn` |
| Markov (char-3) | `markov-char3` |
| LogReg L2 (char-4 TF-IDF) | `tfidf-logreg-char4` |
| Linear SVM (char-3 TF-IDF) | `tfidf-svm-char3` |
| OC-SVM RBF (char-3 TF-IDF) | `tfidf-ocsvm-char3` |
| Random Forest (char-4 TF-IDF) | `tfidf-rf-char4` |
| ExtraTrees (char-3 TF-IDF) | `tfidf-et-char3` |
| Isolation Forest (char-4 TF-IDF) | `tfidf-iforest-char4` |
| XGBoost (char-4 TF-IDF) | `tfidf-xgb-char4` |
| kNN-density | `knn-density` |
| Mahalanobis | `mahalanobis` |
| T+Mahalanobis | `t-mahalanobis` (uses `mahalanobis` code path) |
| Deep SAD | `deep-sad` |
| Deep SVDD | `deep-svdd` |
| Deep One-Class | `deep-one-class` (alias of Deep SVDD) |
| DROCC | `drocc` |

## Dependencies

Use conda to install baseline dependencies:

```bash
conda install -c conda-forge scikit-learn pandas xgboost transformers
```

With pip, use either:

```bash
python -m pip install -e '.[baselines]'
python -m pip install -r baselines/requirements.txt
```

The project base environment already includes PyTorch + sentence-transformers.

## Official Repos (Auto-download)

To download official repositories referenced in the paper:

```bash
python -m baselines.downloads --all
```

The downloader verifies that each repo contains expected files (e.g., `README.md`)
and will fail fast if something is missing.

You can also download only the repos needed by the selected baselines by passing
`--download-repos` to `run_baselines.py`:

```bash
python -m baselines.run_baselines --download-repos --sample-per-class 2000
```

If you want to load models directly from the downloaded repos where possible,
pass `--use-official-repos` as well.

## Running

List available baselines:

```bash
python -m baselines.run_baselines --list
```

Run a subset (recommended for quick checks):

```bash
python -m baselines.run_baselines \
  --baselines tfidf-logreg-char4,markov-char3,char-cnn \
  --sample-per-class 5000 \
  --output baselines/outputs/results.csv
```

Run all baselines (will be slower):

```bash
python -m baselines.run_baselines --sample-per-class 10000
```

### Model downloads

Some baselines require external model downloads (e.g., `urlbert`, `csi`).
By default these are skipped; pass `--allow-downloads` to enable.

### Dataset column

The default evaluation column for `user_logins` is `USERNAME` (not `HOSTNAME`).
Override with `--hostname-col` if needed.

## Outputs

The script writes a CSV with accuracy + latency metrics (ms/sample + samples/sec):

```
baselines/outputs/results.csv
```

Each row includes precision/recall/F1 and confusion counts for traceability.

## License

Baseline code in this folder is covered by the repository's noncommercial code license:
PolyForm Noncommercial 1.0.0. See `/LICENSE` and `/LICENSE-CODE` at the repository root.
