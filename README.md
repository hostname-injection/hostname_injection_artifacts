# Contrastive Cone Divergence (CCD)

Official reference implementation of **Contrastive Cone Divergence (CCD)** from the paper
*When Hostnames Become Code: Detecting Persisted Hostname Injection at Production Scale*.

This repository provides:
- CAHO (Class‑Aware Hostname Obfuscation) augmentation and training utilities.
- Cone partitioning + multi‑probe LSH retrieval.
- Cone sketches and cross‑entropy scoring (GLRT / Eq. 1).
- Benign and malicious priors.
- Split‑conformal calibration for fixed‑FPR global and tenant/window grouped
  thresholds.
- Optional finite-edit decision-stability helpers and search utilities.
- Method-contract accounting for Table 1 and Appendix C CCD/CAHO assumptions.
- Paper-claim coverage accounting for contributions, figures, tables,
  formal claims, and appendices.
- Headline paper-claim accounting for abstract/conclusion numeric anchors.
- HIB de-identification, public-bundle validation, and replay metric utilities.
- HIB dataset-profile aggregate accounting for source, label, split, and
  verified-positive profile claims.
- Evaluation-unit and reproducibility-boundary accounting for Table 2 and
  Appendix E/Table 11.
- Source-code reachability accounting for the paper's static-analysis scope
  check.
- Public-scope taxonomy accounting that keeps public anchors separate from HIB
  training and production positives.
- Live-overlap aggregate accounting for CCD-vs-Regex/WAF reviewed items.
- Table 8 sink-evidence accounting for controlled metadata-to-code replay
  traces.
- Paper metric-table aggregate accounting for baseline, ablation, and mimicry
  claims.
- Stability, drift, family-holdout, and public-real scope aggregate accounting.
- Production latency and throughput aggregate accounting for Figure 5.
- Local latency smoke benchmarking for the encoder and CCD scoring kernel.
- A one-command release-safe paper claim check runner.

For evaluator-facing setup, badge scope, and paper-claim mapping, start with
`BADGE_READINESS.md` and `ARTIFACT_EVALUATION.md`.

`ARTIFACT_MANIFEST.json` is the machine-readable map from paper claims to files,
commands, expected outcomes, release gates, and remaining external publication
items. Check it with:

```bash
python scripts/audit_artifact_readiness.py
```

That audit checks the manifest, required files, public release gates,
privacy/portability wording, and absence of common web-tracking scripts. The
stricter final-publication gate is expected to fail until DOI URLs and external
full-replay artifacts are staged:

```bash
python scripts/audit_artifact_readiness.py --strict-final
```

For IEEE S&P packaging-script submission, start from `metadata.template.toml`.
It includes the claim text, commands, resource notes, provenance/ethics pointers,
and explicit URL placeholders that must be replaced with the final anonymous
submission or DOI-backed artifact URLs.

To build a clean archive for DOI deposition after local checks pass:

```bash
python scripts/build_artifact_archive.py
```

## Artifact Smoke Test

After installation, run the end-to-end smoke path:

```bash
python scripts/run_artifact_smoke.py
```

For a faster detector and bundle gate check without running `pytest`:

```bash
python scripts/run_artifact_smoke.py --skip-tests
```

The smoke path trains a temporary CCD prior bundle from `examples/`, calibrates
global and grouped thresholds into a self-contained model bundle, refreshes
`P_B` plus global/grouped thresholds from a clean benign window, scores and
certifies sample hostnames from that refreshed bundle with grouped thresholds,
emits explanations, encodes hostnames with the bundled CAHO checkpoint, and validates the
checked-in de-identified HIB sample bundle from both the repository and an
extracted archive copy. It also recomputes public replay metrics for the sample
release. The sample bundle is deliberately small; the paper-scale 200.3M-row
replay requires the separate full HIB-Real de-identified release.

## Release-Safe Paper Claim Check Runner

To run the release-safe paper claim surface in one pass:

```bash
python scripts/run_artifact_claim_checks.py
```

This runs the readiness audit, aggregate recomputation scripts, headline-claim
audit, and checked-in public sample fixed-FPR replay metrics without private
data. It emits one JSON report with per-check commands, timing, summaries, and
failures.

## Installation (Conda)

Use the provided install script to set up a fresh environment:

```bash
bash scripts/install_conda.sh
```

This creates a `ccd` conda environment, installs dependencies with conda, then
installs the local package in editable mode. GradCache is optional and must be
installed separately from GitHub (see below). To let the script install
GradCache during setup, run `INSTALL_GRADCACHE=1 bash scripts/install_conda.sh`.

You can also use the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate ccd
```

### Optional: GradCache

GradCache is not on conda or PyPI, so install it directly from GitHub:

```bash
git clone https://github.com/luyug/GradCache /tmp/GradCache
python -m pip install /tmp/GradCache
rm -rf /tmp/GradCache
```

Verify the install:

```bash
python -c "import grad_cache; print(grad_cache.__version__)"
```

If you prefer manual steps:

```bash
conda create -y -n ccd python=3.11
conda activate ccd
conda install -y -c conda-forge -c pytorch \
  numpy scipy pytorch sentence-transformers idna pytest sentencepiece scikit-learn
python -m pip install -e .
```

## Quick Start

### 1) Train / fine‑tune CAHO encoder (optional)

```bash
ccd train-caho \
  --benign data/benign.txt \
  --malicious data/malicious.csv \
  --out caho_encoder
```

For weighted hostname augmentations and two-view contrastive training, use:

```bash
ccd train-caho \
  --benign data/benign.txt \
  --malicious data/malicious.csv \
  --out caho_encoder \
  --loss contrastive \
  --augmenter weighted
```

For very large batch sizes, you can enable GradCache:

```bash
ccd train-caho \
  --benign data/benign.txt \
  --malicious data/malicious.csv \
  --out caho_encoder \
  --loss contrastive \
  --augmenter weighted \
  --grad-cache \
  --grad-cache-chunk-size 128
```

The deployed-style benchmark trainer with the binary auxiliary head is exposed
by `scripts/train_benchmark_caho_binary.py`; it trains the same two-view CAHO
path with supervised orbit contrastive loss, explicit AdamW weight decay, and
an L2-normalized binary classifier head over both CAHO views. Its defaults
match the Appendix C deployed recipe (`lr=1e-4`, weight decay `1e-2`, batch
size `256`, up to `50` epochs). For a local parser/training-loop smoke on
commodity hardware, use `--device cpu --max-rows ... --max-steps ...`; use
`--require-cuda` when replaying a GPU training run and you want CPU fallback to
fail closed. This binary-head trainer intentionally rejects `--grad-cache`
because the Appendix C objective requires supervised orbit labels inside the
contrastive loss; GradCache remains available on the regular pairwise CAHO
trainers.
For paper-style model selection, pass a validation-only benchmark root and
select the checkpoint by validation TPR at the target false-positive rate:

```bash
python scripts/train_benchmark_caho_binary.py \
  --root HostnameCommandInjectionBenchmark/train \
  --validation-root HostnameCommandInjectionBenchmark/validation \
  --validation-target-fpr 1e-4 \
  --restore-best-validation \
  --out caho_model_checkpoint
```

The validation history records the benign-only split-conformal threshold,
clipped order-statistic rank, strict `score > threshold` decision rule, score
source (`binary_auxiliary_head_sigmoid`), and score view
(`canonical_view1_only`) used for the fixed-FPR checkpoint comparison. When
`--restore-best-validation` is set, the trainer restores the encoder and
binary head from the best validation epoch before saving.

All CAHO training entry points accept `--seed` (or `--caho-seed` for
`train-user-logins`) and default to `13`; benchmark training reports record the
seed so augmentation/order replay is explicit.

### Replicating The Full Corpus Training Script

If you have the JSONL/TXT corpora used in the updated script, you can replicate it via:

```bash
ccd train-caho-corpus \
  --benign-dir ../txt_corpus/benign \
  --malicious-jsonl-dir ../filtered_corpus \
  --malicious-txt-dir ../txt_corpus/varied \
  --out ../models/contrastive_sentence_transformer \
  --loss contrastive \
  --augmenter weighted \
  --contrastive-loss learnable \
  --grad-cache \
  --grad-cache-chunk-size 128 \
  --batch-size 8192 \
  --epochs 20 \
  --num-workers 1 \
  --save-best \
  --no-save-final \
  --resume \
  --device auto \
  --empty-cache \
  --no-normalize
```

Or use the wrapper script:

```bash
python scripts/train_caho_corpus.py \
  --benign-dir ../txt_corpus/benign \
  --malicious-jsonl-dir ../filtered_corpus \
  --malicious-txt-dir ../txt_corpus/varied \
  --out ../models/contrastive_sentence_transformer \
  --loss contrastive \
  --augmenter weighted \
  --contrastive-loss learnable \
  --grad-cache \
  --grad-cache-chunk-size 128 \
  --batch-size 8192 \
  --epochs 20 \
  --num-workers 1 \
  --save-best \
  --no-save-final \
  --resume \
  --device auto \
  --empty-cache \
  --no-normalize
```

### 2) Build priors + cone partition

```bash
ccd train-priors \
  --benign data/benign.txt \
  --malicious data/malicious.csv \
  --output ccd_model.npz
```

### Train Directly From `user_logins` CSVs

If you have the benchmark CSVs under `hostname_injection_benchmark/user_logins`, you can build
the full CCD priors in one command:

```bash
ccd train-user-logins \
  --output ccd_user_logins.npz
```

By default, `train-user-logins` applies labels to the `USERNAME` column. If your CSV uses
`HOSTNAME` instead, override with `--hostname-col HOSTNAME`. For other datasets, `HOSTNAME`
remains the typical default.

Label policies control how GPT 5.5 / Claude Opus 4.8 labels are combined
(e.g., `both-m`, `either-m`, `agreement`, `gpt-5.5-only`, `opus-4.8-only`,
`non-u`, `prefer-m`, `prefer-b`). Deprecated `sonnet-only` and `opus-only`
aliases remain for compatibility. The default is `both-m`, which treats a
hostname as malicious only when both models label it `M` and drops rows where
labels are `U`.

You can also:
- Run a dry run to see label counts: `--dry-run`
- Filter low-confidence labels: `--min-confidence 0.9` (or per-model `--min-sonnet-confidence`,
  `--min-opus-confidence`)
- Fine-tune CAHO first (with optional sampling): `--train-caho --caho-sample 100000`

### 3) Score hostnames

```bash
ccd score \
  --model ccd_model.npz \
  --input data/queries.txt \
  --output scores.csv
```

`scores.csv` records the threshold used for each row, the CCD score, and the
strict-threshold prediction. With `--groups`, the row threshold is the resolved
tenant/window threshold; otherwise it is the global model, calibration-file, or
CLI threshold.

For maximum throughput, you can use fast approximate scoring (hard-cone). This may reduce
accuracy; use only when you need the extra throughput:

```bash
ccd score \
  --model ccd_model.npz \
  --input data/queries.txt \
  --output scores.csv \
  --approximate
```

## Baselines

The `baselines/` folder implements classical + neural baselines referenced in
`IEEE_S_P_Hostnames.pdf` and provides accuracy + latency analysis.

Quick start:

```bash
python -m baselines.run_baselines --list
python -m baselines.run_baselines \
  --baselines tfidf-logreg-char4,markov-char3,char-cnn \
  --sample-per-class 5000 \
  --output baselines/outputs/results.csv
```

Some baselines (e.g., `urlbert`, `csi`) download external models. By default they are
skipped; pass `--allow-downloads` to enable. See `baselines/README.md` for full
dependency and usage notes. For the complete optional baseline dependency set,
run `python -m pip install -e '.[baselines]'` or
`python -m pip install -r baselines/requirements.txt`.

## Source-Reachability Scope Check

The paper's 50-repo CodeQL/Semgrep comparison is a scope check for
second-order telemetry paths, separate from HIB replay accuracy. The
release-safe aggregate accounting is checked with:

```bash
python scripts/recompute_source_reachability_metrics.py
```

See `source_reachability/README.md` for the JSONL format used when full
public-corpus finding labels are staged.

## Paper Headline Claim Check

The headline-claim audit recomputes the paper's most visible numeric anchors
from release-safe aggregate artifacts: replay scale, CCD fixed-FPR recall,
latency and throughput, live-overlap added value, label totals,
decision-stability coverage, source/public scope, and hostile-mimicry ranges.

```bash
python scripts/recompute_paper_headline_claims.py
```

See `paper_headline_claims/README.md`.

## Paper Claim Coverage Check

The paper-coverage matrix maps named paper surfaces to artifact evidence:
five contributions, Eq. 1, Proposition 5.1, Lemma C.1, Figures 1-7,
Tables 1-12, and Appendices A-F. Recompute it with:

```bash
python scripts/recompute_paper_claim_coverage.py
```

This check verifies that each item references existing files, aligns with a
manifest claim, and keeps full-data or production-private dependencies marked
as external completion items. See `paper_claim_coverage/README.md`.

## Method Contract Check

Table 1 and Appendix C define the CCD decision contract, frozen configuration
surface, edit-manifest scope, and CAHO training assumptions. Recompute the
release-safe method-contract checks with:

```bash
python scripts/recompute_method_contracts.py
```

This validates Eq. 1 score-path availability, exact full-axis scanning for the
deployed top-R cone sketch that bypasses LSH by default for
calibration/certification, fixed-FPR global and tenant/window
grouped calibration defaults, 4096-cone/top-8 CCD configuration, E1-E12
finite-edit coverage, deterministic certificate closure, calibrated-margin
certificates with deterministic enumeration fallback, CAHO two-view supervised
orbit contrastive training plus an L2-normalized binary head with explicit
AdamW weight decay, benign-only `(P_B, tau_alpha)` refresh, and model bundle
persistence plus validation for config, cone axes, priors, and global/grouped
calibrated thresholds. See `method_contracts/README.md`.

## Public-Scope Taxonomy Check

Public reports and public anchors test taxonomy scope; they are not counted as
HIB training or production positives. Recompute the release-safe aggregate
accounting with:

```bash
python scripts/recompute_public_scope_metrics.py
```

See `public_scope/README.md` for the JSONL format used when individual
public-report labels are staged.

## HIB Dataset-Profile Check

Appendix B and Tables 3, 4, and 9 report source, label, split, and
verified-positive profile denominators for the 200,339,886-row HIB production
replay. Recompute the release-safe aggregate accounting with:

```bash
python scripts/recompute_hib_profile_metrics.py
```

This validates the published aggregate rows and derives consistency checks such
as resolved replay denominator, positive prevalence, source percentages, quality
repair rate, and largest verified-positive family share. Row-level reproduction
still requires the full HIB-Real release described in `ARTIFACT_EVALUATION.md`.

## Evaluation Accounting Check

Table 2 defines the denominator vocabulary used throughout the evaluation, and
Appendix E/Table 11 defines what is released or recomputable versus withheld.
Recompute the release-safe accounting with:

```bash
python scripts/recompute_evaluation_accounting.py
```

This confirms unresolved marker-like strings are excluded from resolved replay
denominators, validates the released-versus-withheld boundary, and checks that
the manifest still lists the external completion items needed for Table 11.
See `evaluation_accounting/README.md`.

## Live-Overlap Aggregate Check

The live/shadow comparison is released as aggregate accounting and masked
summaries, not raw live streams. Recompute the Table 7 overlap accounting with:

```bash
python scripts/recompute_live_overlap_metrics.py
```

See `live_overlap/README.md` for the JSONL format used when masked live labels
are staged.

## Sink-Evidence Aggregate Check

Table 8 reports controlled replay evidence for audited metadata-to-code paths,
not production-compromise claims. Recompute the release-safe accounting with:

```bash
python scripts/recompute_sink_evidence_metrics.py
```

This validates the three controlled matching-sink replay cases, requires
parser/persistence/consumer/effect/detector-boundary fields, checks that CCD
fired before downstream consumption, and confirms that side effects are blocked
or restricted to researcher-controlled endpoints. See `sink_evidence/README.md`.

## Paper Metric-Table Checks

Tables 5, 6, 10, and 12 plus Appendix F synthetic-real summaries are
represented as release-safe aggregate accounting:

```bash
python scripts/recompute_paper_metric_tables.py
```

This validates the published aggregate rows and derives consistency checks such
as CCD's Table 5 TPR lead, LLM checkpoint coverage, the largest ablation drop,
and the weakest hostile-mimicry cell. See `paper_metric_tables/README.md`.

## Stability And Scope Checks

Figure 6, Figure 7, and Section 8.3 report finite-edit stability,
family-holdout/depth, drift-refresh, and public-real scope checks. Recompute the
release-safe aggregate accounting with:

```bash
python scripts/recompute_stability_scope_metrics.py
```

This validates the published aggregate values and explicitly records that
decision-stability certificates are scoped to the frozen normalizer, cone
sketch, score path, threshold, and edit manifest, not downstream sink safety.
Certificate JSON also records the strict `score > threshold` decision rule and
the score-path constants used by the certificate.
See `stability_scope/README.md`.

## Production-Latency Aggregates

Figure 5 and the Table 5 latency columns report production full-path latency,
throughput, and scoring-kernel timing. Recompute the release-safe aggregate
accounting with:

```bash
python scripts/recompute_production_latency_metrics.py
```

This validates the published production aggregate values and the boundary that
the local latency smoke exercises code paths but is not expected to reproduce
production hardware numbers. See `production_latency/README.md`.

## Convenience Commands

This repo includes a `Makefile` plus a few CLI entry points to make demos and reviews easy.

### Make targets

```bash
make sanity
make diagnose
make explain MODEL=ccd_model.npz INPUT=data/queries.txt
make score MODEL=ccd_model.npz INPUT=data/queries.txt OUTPUT=out/scores.csv
```

Variables you can override: `MODEL`, `INPUT`, `OUTPUT`, `CHECKPOINT`, `PER_CLASS`, `BATCH`.

### Entry points

These are installed with the package:

```bash
ccd-sanity --per-class 50
ccd-diagnose --checkpoint caho_model_checkpoint --batch-size 256
ccd-explain --model ccd_model.npz --input data/queries.txt --top-k 3
ccd-score --model ccd_model.npz --input data/queries.txt --output out/scores.csv
```

## Diagnostics

Use diagnostics to confirm which device is being used (GPU, MPS, CPU) and to measure
encoder throughput:

```bash
ccd-diagnose --checkpoint caho_model_checkpoint --batch-size 256 --num-samples 2048
```

For an evaluator-facing latency smoke over the local CAHO checkpoint and the
CCD cone-scoring kernel:

```bash
python scripts/benchmark_artifact_latency.py --num-samples 64 --repeats 1 --warmup 0
```

This is hardware-dependent. It exercises the artifact paths but does not assert
the paper's production p95/p99/p99.9 or 0.60 ms amortized latency numbers on an
arbitrary evaluator machine.

## Explainability

To inspect *why* a hostname was scored as malicious or benign, use `ccd-explain` to get the
top contributing cones and priors:

```bash
ccd-explain --model ccd_model.npz --input data/queries.txt --top-k 3 --output out/explanations.json
```

The output includes:
- Per-hostname score and prediction.
- The top-k cone indices, similarities, and weights.
- Benign/malicious log-prior contributions for each cone.

If you want a smaller accuracy hit than hard-cone, try a small top-k:

```bash
ccd score \
  --model ccd_model.npz \
  --input data/queries.txt \
  --output scores.csv \
  --approximate-k 4
```

### Evaluate / Encode With CAHO Checkpoint

To encode hostnames using the bundled checkpoint at `caho_model_checkpoint`:

```bash
ccd eval-caho \
  --input data/queries.txt \
  --output embeddings.npz
```

You can also specify a custom checkpoint:

```bash
ccd eval-caho \
  --model /path/to/caho_model_checkpoint \
  --input data/queries.txt \
  --output embeddings.csv \
  --format csv
```

### 4) Calibrate a fixed‑FPR threshold

```bash
ccd calibrate \
  --model ccd_model.npz \
  --benign data/benign_calibration.txt \
  --output calibration.json \
  --save-model ccd_model.calibrated.npz \
  --alpha 1e-4
```

`calibration.json` records the calibrated threshold, the clipped split-conformal
order-statistic rank, the strict `score > threshold` decision rule, and the score
path. The optional `--save-model` output embeds global or grouped thresholds in a
model bundle, so later `score` and `certify` commands can use the frozen
threshold without a separate calibration file. Calibration fails closed if any
benign calibration score or loaded threshold is non-finite.

To calibrate per tenant/window while preserving a global fallback threshold,
pass one group id per benign calibration row:

```bash
ccd calibrate \
  --model ccd_model.npz \
  --benign data/benign_calibration.txt \
  --groups data/benign_calibration_groups.txt \
  --output calibration.json \
  --alpha 1e-4
```

At scoring or certification time, pass the matching query group file. Add
`--require-group-thresholds` if every query must have an explicit grouped
threshold in `calibration.json` or the saved model bundle. The same `--groups`
file can be supplied to `ccd-explain` so explanation rows use and record the
same tenant/window threshold as scoring. The Python `CCDModel.predict(...)`
API accepts the same `calibration_groups` and `missing_group_threshold`
arguments for model-level tenant/window decisions. Group files must contain one
non-empty group id per non-empty hostname row; empty group ids are rejected
instead of falling back to the global threshold.

If you intend to use `--approximate` at inference time, calibrate with it as well:

```bash
ccd calibrate \
  --model ccd_model.npz \
  --benign data/benign_calibration.txt \
  --output calibration.json \
  --alpha 1e-4 \
  --approximate
```

Likewise, if you use `--approximate-k`, calibrate with the same value:

```bash
ccd calibrate \
  --model ccd_model.npz \
  --benign data/benign_calibration.txt \
  --output calibration.json \
  --alpha 1e-4 \
  --approximate-k 4
```

### 5) Refresh the benign reference without retraining

For benign-only drift refresh, update `P_B` and recalibrate `tau_alpha` while
leaving the encoder, cone axes, scoring configuration, and malicious priors
fixed:

```bash
ccd refresh-benign \
  --model ccd_model.calibrated.npz \
  --benign data/recent_benign_window.txt \
  --groups data/recent_benign_window_groups.txt \
  --output ccd_model.refreshed.npz \
  --report refresh.json \
  --alpha 1e-4
```

`refresh.json` records the old threshold, new global/grouped thresholds, score
path, and the refresh scope. The intended contract is narrow: `P_B` and
`tau_alpha` move; `P_M`, the CAHO encoder, cone partition, and score path stay
frozen. Omit `--groups` for a global-only refresh only when the input model is
not already carrying grouped thresholds. If the input model has tenant/window
grouped thresholds, `ccd refresh-benign` requires replacement `--groups` and
fails closed unless `--drop-grouped-thresholds` is passed to explicitly discard
those grouped thresholds. Refresh is transactional: if recalibration or grouped
threshold validation fails, the in-memory model keeps the previous `P_B`,
global threshold, and grouped thresholds.

## Data Formats

- `benign.txt`: one hostname per line.
- `malicious.csv`: standard CSV with `hostname,family` columns (header optional);
  quote fields that contain commas or other CSV metacharacters.
- `queries.txt`: one hostname per line.

## Configuration

The CCD configuration is a JSON file that maps to `CCDConfig`:

```json
{
  "encoder": {"model_name": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"},
  "cone": {"dim": 384, "num_cones": 4096, "active_cones": 8, "temperature": 10.0},
  "prior": {"smoothing": 1e-6},
  "calibration": {"alpha": 1e-4},
  "scoring": {"effective_count": 1.0, "mixture_weights": {}}
}
```

Pass this file via `--config` to `train-priors`.
If you fine‑tuned a CAHO encoder, you can override it with `--encoder`.

## Library Usage

```python
from ccd.io import load_model

model = load_model("ccd_model.npz")
scores = model.score(["example.com"], normalize=True)
```

## Notes on Fidelity

- Cone sketching and likelihood-ratio scoring follow Eq. (1) in the paper:
  `S(q) = log sum_k pi_k exp(n0 * (H(q; P_B) - H(q; P_M,k)))`.
- The deployed normalizer decodes valid UTF-8 percent runs before Unicode/IDNA
  normalization, while retaining a byte-residue fallback for malformed runs.
  Certificate output includes a per-row normalization trace for URL-like
  scheme/userinfo/port/path/query/fragment segmentation and decode changes.
- Benign drift refresh is implemented as a narrow `(P_B, tau_alpha)` update;
  `P_M`, cone axes, encoder config, and scoring config remain fixed.
- CCD model bundles fail closed on non-finite or shape-incompatible cone axes
  and malformed prior arrays before scoring or certification.
- The edit model E1–E12 is implemented in `ccd/edit_model.py`; emitted
  stability certificates can use calibrated-margin bounds and otherwise use
  deterministic finite-edit closure. Certification inputs fail closed on
  non-finite thresholds, invalid bounds, and invalid edit-ball limits before
  scoring.
- The CAHO augmentation sets match the paper’s benign/malicious design.
- Public de-identification reports intentionally do not disclose whether
  private/raw hostnames were duplicated; private-origin grouping checks are used
  only as fail-closed validation gates.

## Project Layout

- `ccd/` core library
- `scripts/` convenience scripts (mirrors CLI)
- `baselines/` replay baselines
- `deidentification_release/` HIB public-release pipeline and checked sample
- `examples/` tiny smoke-test inputs
- `ARTIFACT_EVALUATION.md` paper-claim and badge guide

## License

This repository is released under a research-only noncommercial split license:

- Code: PolyForm Noncommercial License 1.0.0 (`/LICENSE-CODE`)
- Data, docs, and model artifacts: CC BY-NC 4.0 (`/LICENSE-ARTIFACTS`)

See `/LICENSE` for scope details and precedence rules.
