# Artifact Evaluation Guide

This guide is written for IEEE S&P artifact evaluators. It maps the paper
claims in `IEEE_S_P_Hostnames.pdf` to code, data, and commands in this
repository, and it states which claims require the separate full HIB-Real
de-identified release rather than the small checked-in sample bundle.

## Target Badges

The current IEEE S&P 2027 artifact page names three badges:

- **Available**: the final artifact should be deposited permanently and
  publicly, for example on Zenodo/FigShare/Dryad with a DOI. This repository is
  structured so that the code, sample release bundle, model files, and external
  full HIB release can be archived together. The DOI is a publication step, not
  something this local worktree can create.
- **Functional**: the code must be documented, complete enough to exercise the
  paper artifact, and runnable on machines other than the authors'. The smoke
  path below is the first evaluator command to run. The public artifact path
  runs on commodity CPU Linux/macOS hosts or public research infrastructure
  without SSH service, GUI, paid API, special hardware, GPU, or
  production-network access.
- **Reproduced**: evaluators should be able to obtain results supporting the
  paper's main claims. Fast smoke commands reproduce mechanics; headline replay
  metrics require the full de-identified HIB-Real bundle described in Appendix
  B/E of the paper.

## Quick Evaluator Path

From a fresh checkout on Python 3.11:

```bash
conda env create -f environment.yml
conda activate ccd
python scripts/audit_artifact_readiness.py
python scripts/run_artifact_smoke.py
```

If `pytest` is already known to pass and you only want the detector plus
release-gate smoke:

```bash
python scripts/run_artifact_smoke.py --skip-tests
```

The smoke command compiles the code, optionally runs tests, validates the
checked-in HIB public sample bundle from the repository and from an extracted
archive copy, trains a temporary CCD prior bundle from `examples/`, calibrates
a split-conformal global and grouped threshold set into a self-contained model
bundle, refreshes P_B/global/grouped thresholds from a clean benign window,
scores, explains, and certifies sample hostnames from that refreshed bundle
with grouped thresholds,
recomputes public replay metrics for the sample release, and encodes hostnames
with `caho_model_checkpoint`.

Expected runtime on a laptop CPU is a few minutes after dependencies are
installed. The smoke path does not require private data, network egress, or a
GPU.

`ARTIFACT_MANIFEST.json` records the paper-claim map, expected commands, target
badges, public release bundle, and external completion items in a
machine-readable form. `scripts/audit_artifact_readiness.py` checks that
manifest, required files, public release gates, public privacy wording,
author-local path portability, and absence of common web-tracking snippets.
`python scripts/audit_artifact_readiness.py --strict-final` additionally fails
while DOI placeholders or external full-replay completion items remain.
`metadata.template.toml` is a draft input/output shape for the IEEE S&P
artifact packaging script; replace its `example.org` URLs before HotCRP
submission. `ARTIFACT_MANIFEST.json` also records the IEEE S&P operational
requirements we can check locally: source packaging, public-infrastructure
runability, one-day/scaled-down runtime justification, tracking absence,
claim-to-script mapping, and external full-data boundaries. It also encodes the
badge criteria from the IEEE page directly: Available requires DOI-backed
permanent public retrieval, Functional is split into documentation,
completeness, exercisability, and other-machine portability evidence, and
Reproduced records the independent replay path plus tolerance/scaled-down
experiment justification.

## Current Local Verification Snapshot

This worktree has been checked with:

```bash
uv run --python 3.11 --with pytest --with numpy --with scipy --with idna \
  --with torch --with sentence-transformers --with sentencepiece \
  --with scikit-learn \
  python -m pytest -q

PYTHONPATH=deidentification_release/scripts \
python3 deidentification_release/scripts/validate_public_bundle.py \
  --bundle deidentification_release/data/release/hib_release_public_bundle.tar.gz

PYTHONPATH=deidentification_release/scripts \
python3 deidentification_release/scripts/validate_release_gate.py \
  --public-release deidentification_release/data/release/hib_release.jsonl \
  --audit-dir deidentification_release/data/audits \
  --bundle deidentification_release/data/release/hib_release_public_bundle.tar.gz \
  --count-rows

uv run --python 3.11 --with pytest --with numpy --with scipy --with idna \
  --with torch --with sentence-transformers --with sentencepiece \
  --with scikit-learn \
  python scripts/run_artifact_smoke.py --skip-tests

python3 scripts/audit_artifact_readiness.py

python3 scripts/run_artifact_claim_checks.py

python3 scripts/audit_artifact_readiness.py --strict-final

python3 scripts/recompute_method_contracts.py

python3 scripts/recompute_paper_claim_coverage.py

python3 scripts/recompute_paper_headline_claims.py

python3 scripts/recompute_hib_profile_metrics.py

python3 scripts/recompute_evaluation_accounting.py

python3 scripts/recompute_source_reachability_metrics.py

python3 scripts/recompute_public_scope_metrics.py

python3 scripts/benchmark_artifact_latency.py --num-samples 64 --repeats 1 --warmup 0

python3 scripts/recompute_production_latency_metrics.py

python3 scripts/recompute_live_overlap_metrics.py

python3 scripts/recompute_sink_evidence_metrics.py

python3 scripts/recompute_paper_metric_tables.py

python3 scripts/recompute_stability_scope_metrics.py

python3 scripts/build_artifact_archive.py --dry-run
```

Observed status: the pytest suite passed (`217 passed, 1 skipped`), the public
bundle validator passed with 32 archived files, the release gate passed with 150
public sample rows, the same gate passed from an extracted bundle copy, and the
artifact smoke and readiness audit passed, including portability, privacy, and
web-tracking scans. The release-safe paper claim check runner passed 14 checks
from one command. The strict final-publication audit correctly reported 11
remaining DOI and external completion items. The method-contract script
validated Table 1 contract rows, Appendix C CCD defaults, edit-manifest
coverage, global and tenant/window grouped split-conformal thresholding, CAHO
supervised orbit contrastive/binary-head training support with benign diversity
preservation, binary auxiliary loss over both L2-normalized CAHO views,
fail-closed GradCache handling for the supervised binary trainer, and explicit AdamW
weight decay, Appendix C CAHO deployed-recipe optimizer defaults in the benchmark binary
trainer plus 94 GB CUDA batch defaults for replay-scale actual and regular
GradCache CAHO training, exact full-axis scanning for the deployed top-R cone sketch that
bypasses LSH by default for calibration/certification, unit-embedding normalization
across exact, torch, top-k, and fast scoring paths, calibrated-margin certificates with
deterministic enumeration fallback, benign-only P_B/threshold refresh, and
global/grouped calibrated-threshold bundle persistence. The paper-claim coverage script validated 33 coverage items
covering five contributions, Eq. 1, Proposition 5.1, Lemma C.1, Figures 1-7,
Tables 1-12, and Appendices A-F, with full-data or production-private
dependencies tied to manifest external completion items. The headline-claim script
validated the paper's headline numeric anchors from release-safe aggregates,
including replay scale, CCD fixed-FPR recall, latency, live-overlap added
value, label totals, decision-stability coverage, scope checks, and
hostile-mimicry ranges. The HIB profile script validated the Table 3
source/resolved/unresolved counts, Appendix B label and
event counts, and Table 4/Table 9 verified-positive family and obfuscation
denominators. The evaluation-accounting script validated Table 2 denominator
units and Appendix E/Table 11 released-versus-withheld boundaries against the
manifest. The source-reachability accounting script reproduced the paper's
CodeQL/Semgrep aggregate counts. The public-scope accounting script reproduced
the 118/127 public-report taxonomy count and verified that public anchors are
not HIB positives. The local latency smoke reported hardware-dependent encoder
and scoring-kernel timings. The production-latency script validated Figure 5
full-path latency and throughput aggregates, scoring-kernel latency, and Table 5
CCD tail-latency alignment. The live-overlap accounting script reconstructed
Table 7 aggregate cells and derived reviewed-item PPV/review-load metrics. The
sink-evidence script validated the three Table 8 controlled matching-sink
replay traces and their non-compromise boundary. The
paper metric-table script validated Tables 5, 6, 10, and 12 plus Appendix F
aggregate metrics. The stability/scope script validated Figure 7 certificate
coverage, Figure 6 family-holdout/depth ranges, drift-refresh invariants,
independent replay tolerance, and public-real recall ordering. The archive
builder dry-run passed with 232 files after excluding intermediate training
checkpoints; a temporary full archive build produced 232 hashed files with a
valid SHA-256 sidecar and manifest. The full paper-scale
metrics still require the external full HIB-Real release bundle described below.

## Verified Public Sample Bundle

The checked-in de-identification sample is intentionally small. It proves the
release mechanics, privacy gates, checksum sidecars, public/private boundary,
and fixed-FPR replay code path over reviewed public scores and release-safe
public calibration groups, including non-empty TP/FP/TN/FN accounting cells; it is not the
200.3M-row HIB replay used for the paper's headline accuracy numbers.

Run:

```bash
PYTHONPATH=deidentification_release/scripts \
python deidentification_release/scripts/validate_public_bundle.py \
  --bundle deidentification_release/data/release/hib_release_public_bundle.tar.gz

PYTHONPATH=deidentification_release/scripts \
python deidentification_release/scripts/validate_release_gate.py \
  --public-release deidentification_release/data/release/hib_release.jsonl \
  --audit-dir deidentification_release/data/audits \
  --bundle deidentification_release/data/release/hib_release_public_bundle.tar.gz \
  --count-rows
```

Important privacy invariant: public rows, public audit reports, and the public
bundle do not disclose whether raw/private hostnames were duplicated. Private
raw-hostname grouping checks may be used internally as fail-closed gates, but
their counts and existence results are withheld from public artifacts.

## Full HIB Replay Path

Appendix B/E of the paper describes the full HIB-Real replay: 200,339,886
de-identified rows, labels, splits, detector outputs, thresholds, and scripts
that rebuild replay denominators, resolved TPR/FPR, detector overlap, and
calibration accounting while keeping unresolved rows separate.

The full JSONL is too large to keep directly in this source repository. When
the full HIB-Real release bundle is available, place it outside git, validate
it with the same validators, and recompute public metrics:

```bash
PYTHONPATH=deidentification_release/scripts \
python deidentification_release/scripts/validate_public_bundle.py \
  --bundle /path/to/hib_release_public_bundle.tar.gz

PYTHONPATH=deidentification_release/scripts \
python deidentification_release/scripts/validate_release_gate.py \
  --public-release /path/to/hib_release.full.jsonl \
  --audit-dir /path/to/audits \
  --bundle /path/to/hib_release_public_bundle.tar.gz \
  --count-rows

python deidentification_release/scripts/recompute_metrics.py \
  --public-release /path/to/hib_release.full.jsonl \
  --alpha 1e-4 \
  --out /path/to/audits/recomputed_public_metrics.json
```

For the `Reproduced` badge, the full release is the authoritative data source
for paper-scale replay claims such as 93.5% recall at `1e-4` FPR over 363,401
verified executable-semantics positives. The recomputation script reports
label accounting, calibration rows, the split-conformal threshold when public
CCD scores are present, fixed-FPR confusion counts, unresolved-row exclusion,
and detector-overlap tables. The checked-in sample bundle is only a functional
and privacy-gate fixture.

## Paper Claim Map

| Paper claim / component | Repository evidence | Commands |
| --- | --- | --- |
| CCD log-sum-exp likelihood-ratio cone sketch, Eq. 1 | `ccd/scoring.py`, `ccd/cone.py`, `ccd/model.py` | `python -m pytest tests/test_scoring.py tests/test_cone.py tests/test_model_pipeline.py` |
| Split-conformal fixed-FPR thresholding | `ccd/calibration.py`, `ccd/cli.py`, `ccd/io.py` | `ccd calibrate --model ... --benign ... --alpha 1e-4 --save-model ...` |
| CAHO hostname augmentation and edit model | `ccd/augment.py`, `ccd/edit_model.py`, `ccd/train.py` | `python -m pytest tests/test_augment.py tests/test_train.py` |
| CAHO benchmark training with binary auxiliary head | `ccd/benchmark_training.py`, `scripts/train_benchmark_caho_binary.py` | `python -m pytest tests/test_benchmark_training.py` |
| Table 1 / Appendix C method contracts | `method_contracts/`, `scripts/recompute_method_contracts.py` | `python scripts/recompute_method_contracts.py` |
| Named paper-claim coverage matrix | `paper_claim_coverage/`, `scripts/recompute_paper_claim_coverage.py` | `python scripts/recompute_paper_claim_coverage.py` |
| Headline numeric paper claims | `paper_headline_claims/`, `scripts/recompute_paper_headline_claims.py` | `python scripts/recompute_paper_headline_claims.py` |
| Finite-edit decision-stability coverage | `ccd/certify.py`, `ccd/edit_model.py` | `python -m pytest tests/test_certify.py tests/test_augment.py` |
| HIB chunked dataset and benchmark training | `ccd/benchmark_dataset.py`, `ccd/benchmark_training.py`, `scripts/train_benchmark_caho*.py` | `python -m pytest tests/test_benchmark_dataset.py` |
| Classical/neural baselines | `baselines/` | `python -m baselines.run_baselines --list` |
| HIB dataset-profile aggregate accounting | `hib_profile/`, `scripts/recompute_hib_profile_metrics.py` | `python scripts/recompute_hib_profile_metrics.py` |
| Evaluation units and reproducibility boundary | `evaluation_accounting/`, `scripts/recompute_evaluation_accounting.py` | `python scripts/recompute_evaluation_accounting.py` |
| Source-code reachability scope check | `source_reachability/`, `scripts/recompute_source_reachability_metrics.py` | `python scripts/recompute_source_reachability_metrics.py` |
| Public-scope taxonomy check | `public_scope/`, `scripts/recompute_public_scope_metrics.py` | `python scripts/recompute_public_scope_metrics.py` |
| Local latency smoke | `scripts/benchmark_artifact_latency.py`, `ccd/diagnostics.py`, `baselines/latency.py` | `python scripts/benchmark_artifact_latency.py --num-samples 64 --repeats 1 --warmup 0` |
| Production-latency aggregate accounting | `production_latency/`, `scripts/recompute_production_latency_metrics.py` | `python scripts/recompute_production_latency_metrics.py` |
| Live-overlap aggregate accounting | `live_overlap/`, `scripts/recompute_live_overlap_metrics.py` | `python scripts/recompute_live_overlap_metrics.py` |
| Table 8 sink-evidence aggregate accounting | `sink_evidence/`, `scripts/recompute_sink_evidence_metrics.py` | `python scripts/recompute_sink_evidence_metrics.py` |
| Paper metric-table aggregate accounting | `paper_metric_tables/`, `scripts/recompute_paper_metric_tables.py` | `python scripts/recompute_paper_metric_tables.py` |
| Stability, drift, and scope aggregate accounting | `stability_scope/`, `scripts/recompute_stability_scope_metrics.py` | `python scripts/recompute_stability_scope_metrics.py` |
| De-identification and public release gates | `deidentification_release/scripts/hib_deid.py`, validators, sample bundle | commands in "Verified Public Sample Bundle" |
| Artifact kick-the-tires path | `scripts/run_artifact_smoke.py`, `examples/` | `python scripts/run_artifact_smoke.py --skip-tests` |
| One-command release-safe paper claim checks | `scripts/run_artifact_claim_checks.py` | `python scripts/run_artifact_claim_checks.py` |
| DOI-ready archive assembly | `scripts/build_artifact_archive.py`, `ARTIFACT_MANIFEST.json` | `python scripts/build_artifact_archive.py --dry-run` |

## Security Notes

Hostnames in the benchmark can contain executable syntax for shells, SQL,
templates, URL fetchers, and lookup callbacks. Treat all benchmark strings as
untrusted input. Do not paste examples into shells, database consoles, template
renderers, or network-enabled callback harnesses. The provided smoke and replay
commands inspect and score strings offline.

## Expected Limitations

- Live/shadow deployment evidence in the paper is released only as audited
  aggregate counts and masked summaries, not raw live streams.
- Tenant identities, raw operational logs, reversible mappings, raw callback
  domains, private sink details, production control internals, and exact
  private strings are intentionally outside the public artifact boundary.
- Hardware affects latency measurements. Classification/count metrics should be
  stable across machines when using the released de-identified rows and fixed
  detector outputs.

## Not Yet Proven By This Worktree Alone

- Permanent public archival with a DOI is an external publication step required
  for the `Available` badge.
- `metadata.template.toml` still contains placeholder URLs and must be rewritten
  with the final anonymous submission URL and permanent DOI-backed artifact URL
  before HotCRP/final-public submission.
- The checked-in 150-row sample proves mechanics, privacy gates, and fixed-FPR
  replay over reviewed public scores, but it cannot reproduce the paper-scale
  200,339,886-row HIB denominators or headline fixed-FPR metrics.
- The `Reproduced` badge remains dependent on validating the full HIB-Real
  release bundle and rerunning the replay commands above on that full release.
