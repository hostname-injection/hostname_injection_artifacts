# Contrastive Cone Divergence

Reference implementation for the CAHO and CCD pipeline used in *When
Hostnames Become Code: Detecting Persisted Hostname Injection at Production
Scale*.

This repository is intentionally kept as a lean ML research-code artifact. It
contains the CAHO/CCD source package, training and evaluation scripts, small
smoke-test inputs, tests, and the de-identified public-release tooling. It
does not include pretrained models.

A reviewer-viewable de-identified data slice is available here:
https://drive.google.com/drive/folders/1KeKZyIXIqZvEJ4tZAWxE9h4gPoinZCWt?usp=drive_link

The data slice and checked-in HIB sample are de-identified. They preserve the
public replay schema and safety checks, but they are not the original private
evaluation rows. CAHO and CCD runs on de-identified data should therefore be
expected to have data shift relative to the original evaluation set used for
the reported paper results.

## What Is Included

- `ccd/`: CAHO augmentation/training, CCD priors, cone scoring, calibration,
  benign-prior refresh, explanation, and certification logic.
- `scripts/`: training, scoring, benchmark, validation, and smoke-test entry
  points.
- `deidentification_release/`: public HIB sample, schema, bundle validation,
  non-linkability checks, and release metric replay.
- `examples/`: tiny inputs used by the smoke test.
- `tests/`: unit and integration tests for the runnable code paths.

## Installation

The conda setup script creates a `ccd` environment and installs the local
package in editable mode:

```bash
bash scripts/install_conda.sh
conda activate ccd
```

You can also create the environment directly:

```bash
conda env create -f environment.yml
conda activate ccd
python -m pip install -e .
```

GradCache is required for replay-scale pairwise CAHO training. This README does
not duplicate GradCache installation instructions; use the upstream repository:
https://github.com/luyug/GradCache

## Runtime Requirements

Use Python 3.11 with the dependencies in `environment.yml`. The unit tests,
de-identification validators, and the small `examples/` smoke path are designed
to run on a normal CPU machine after the Python dependencies are installed.

Paper-scale CAHO training is a GPU workload. The shipped defaults target a
single CUDA GPU with 94 GB of VRAM; lower-memory machines should reduce batch
sizes only for debugging, and the training scripts will warn that changed
settings should be expected to produce different results. Full replay-scale
training and evaluation require the external de-identified data slice or full
release rather than the tiny checked-in smoke inputs.

## Smoke Test

Run the end-to-end smoke path after installation:

```bash
python scripts/run_artifact_smoke.py
```

For a faster check that skips the full pytest suite:

```bash
python scripts/run_artifact_smoke.py --skip-tests
```

The smoke test trains a temporary CAHO checkpoint from `examples/`, trains CCD
priors with that checkpoint, calibrates thresholds, refreshes `P_B`, scores and
certifies sample hostnames, emits explanations, evaluates CAHO embeddings, and
validates the de-identified HIB sample bundle.

## Reviewer Evidence

Use these commands to exercise the implementation surfaces that correspond to
the main method claims:

```bash
make test
python scripts/run_artifact_smoke.py --skip-tests
```

The test suite covers CAHO augmentation and training defaults, GradCache hooks,
CCD prior construction, cone scoring, split-conformal calibration, grouped
thresholds, benign-only `P_B` refresh, explanation output, finite-edit
certification, CLI CAHO-first gates, and de-identification release gates. The
smoke command is the scaled-down executable path through CAHO training, CCD
training, calibration, `P_B` refresh, scoring, explanation, certification,
CAHO embedding export, and public release validation. For paper-scale
reproduction, run the same pipeline on the reviewer data slice or full
de-identified release rather than on the checked-in `examples/` inputs.

## Required Pipeline Order

Reviewer-facing execution should follow the same order as the method:

1. Train CAHO on the available benign and malicious training data.
2. Train CCD priors with the trained CAHO checkpoint.
3. Calibrate the global or grouped fixed-FPR threshold.
4. Refresh `P_B` only from clean benign windows when modeling drift.
5. Score, explain, or certify from the calibrated CCD model bundle.

There is no supported path for training, scoring, calibrating, refreshing, or
certifying CCD without a trained CAHO checkpoint. The repository also does not
provide a user-logins-only CAHO or CCD training path; training is expected to
use all available benchmark families.
Scoring, explanation, and certification require the calibrated threshold to be
embedded in the model bundle by `ccd calibrate --save-model`; they do not accept
ad hoc threshold or calibration-file overrides.

## CAHO Training

All CAHO training entry points default to the shipped paper recipe:
`--epochs 20 --lr 1e-4 --weight-decay 1e-2 --seed 13`. The training scripts
warn when result-affecting settings are changed; results should be expected to
differ from the reported CAHO/CCD results when defaults are changed.

Small file-based training:

```bash
ccd train-caho \
  --benign data/benign.txt \
  --malicious data/malicious.csv \
  --out caho_encoder \
  --loss contrastive \
  --augmenter weighted
```

Full corpus training requires both malicious corpus sources and fails closed if
either is missing or empty:

```bash
ccd train-caho-corpus \
  --benign-dir txt_corpus/benign \
  --malicious-jsonl-dir filtered_corpus \
  --malicious-txt-dir txt_corpus/varied \
  --out caho_encoder \
  --loss contrastive \
  --augmenter weighted \
  --grad-cache \
  --batch-size 49152 \
  --grad-cache-chunk-size 8192 \
  --epochs 20
```

For the benchmark trainer with the supervised binary auxiliary head:

```bash
python scripts/train_benchmark_caho_binary.py \
  --root HostnameCommandInjectionBenchmark/train \
  --validation-root HostnameCommandInjectionBenchmark/validation \
  --validation-target-fpr 1e-4 \
  --restore-best-validation \
  --out caho_encoder
```

The 94 GB CUDA defaults are `batch_size=16384` for the actual non-GradCache
objective and `batch_size=49152` with `grad_cache_chunk_size=8192` for the
GradCache path.

## CCD Training

Train CCD priors only after CAHO training:

```bash
ccd train-priors \
  --benign data/benign.txt \
  --malicious data/malicious.csv \
  --encoder caho_encoder \
  --output ccd_model.npz
```

`train-priors` requires `--encoder` to point to a trained CAHO checkpoint. It
does not fall back to an implicit base encoder.

## Calibration

Calibrate the fixed-FPR threshold on benign calibration data:

```bash
ccd calibrate \
  --model ccd_model.npz \
  --benign data/benign_calibration.txt \
  --output calibration.json \
  --save-model ccd_model.calibrated.npz \
  --alpha 1e-4
```

For tenant/window grouped thresholds, pass one non-empty group id per benign
calibration row:

```bash
ccd calibrate \
  --model ccd_model.npz \
  --benign data/benign_calibration.txt \
  --groups data/benign_calibration_groups.txt \
  --output calibration.json \
  --save-model ccd_model.calibrated.npz \
  --alpha 1e-4
```

The calibration output records the split-conformal order-statistic rank, score
path, threshold source, and the strict `score > threshold` decision rule.
`--save-model` is required so later commands consume the same embedded
threshold and grouped-threshold state.

## Benign-Prior Refresh

Refresh only the benign reference distribution and thresholds from a clean
benign window:

```bash
ccd refresh-benign \
  --model ccd_model.calibrated.npz \
  --benign data/recent_benign_window.txt \
  --groups data/recent_benign_window_groups.txt \
  --output ccd_model.refreshed.npz \
  --report refresh.json \
  --alpha 1e-4
```

This updates `P_B` and `tau_alpha`; `P_M`, CAHO, cone axes, scoring
configuration, and the score path remain fixed. The refresh is transactional:
if recalibration or grouped-threshold validation fails, the previous model
state remains unchanged.

## Scoring And Explanation

Score hostnames with a calibrated or refreshed model bundle:

```bash
ccd score \
  --model ccd_model.refreshed.npz \
  --input data/queries.txt \
  --groups data/query_groups.txt \
  --output scores.csv \
  --require-group-thresholds
```

Explain CCD decisions:

```bash
ccd explain \
  --model ccd_model.refreshed.npz \
  --input data/queries.txt \
  --groups data/query_groups.txt \
  --output explanations.json \
  --top-k 3
```

The outputs include the resolved threshold, threshold source, score, strict
prediction, and top contributing cones and priors.

## Certification

Certification uses the calibrated threshold and the frozen edit manifest:

```bash
ccd certify \
  --model ccd_model.refreshed.npz \
  --input data/queries.txt \
  --groups data/query_groups.txt \
  --output certificates.json \
  --radius 1 \
  --require-group-thresholds
```

Certificates are scoped to the frozen normalizer, cone sketch, score path,
threshold, and edit manifest. Inputs fail closed on invalid thresholds, invalid
edit-ball limits, non-finite bounds, or missing required group thresholds.

## De-Identification Release Checks

Validate the checked-in public HIB sample bundle:

```bash
python deidentification_release/scripts/validate_public_bundle.py \
  --bundle deidentification_release/data/release/hib_release_public_bundle.tar.gz
python deidentification_release/scripts/validate_release_gate.py \
  --public-release deidentification_release/data/release/hib_release.jsonl \
  --audit-dir deidentification_release/data/audits \
  --bundle deidentification_release/data/release/hib_release_public_bundle.tar.gz \
  --count-rows
python deidentification_release/scripts/recompute_metrics.py \
  --public-release deidentification_release/data/release/hib_release.jsonl
```

Public reports intentionally do not disclose whether duplicate raw hostnames
or raw-hostname grouping conditions existed in the private input.

## Useful Commands

```bash
make test
make artifact-smoke
make diagnose CHECKPOINT=caho_encoder BATCH=256
make score MODEL=ccd_model.refreshed.npz INPUT=data/queries.txt OUTPUT=out/scores.csv
make artifact-latency CHECKPOINT=caho_encoder
```

Installed console entry points:

```bash
ccd-diagnose --checkpoint caho_encoder --batch-size 256
ccd-score --model ccd_model.refreshed.npz --input data/queries.txt --output out/scores.csv
ccd-explain --model ccd_model.refreshed.npz --input data/queries.txt --output out/explanations.json
```

## Data Formats

- `benign.txt`: one benign hostname per line.
- `malicious.csv`: CSV with `hostname,family` columns.
- `queries.txt`: one hostname per line.
- Group files: one non-empty group id per corresponding hostname row.

## Notes

- CCD scoring follows the Eq. (1) likelihood-ratio score path implemented in
  `ccd/scoring.py`.
- Model loading, scoring, explanation, calibration, refresh, and certification
  reject malformed priors, bad cone axes, non-finite embeddings, invalid
  thresholds, and incompatible score-path configuration.
- `ccd refresh-benign` is deliberately narrow: only `P_B` and calibrated
  thresholds move.
- The edit model E1-E12 is implemented in `ccd/edit_model.py`.
