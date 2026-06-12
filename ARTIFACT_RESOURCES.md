# Artifact Resources and Runtime Notes

This file is intended for IEEE S&P artifact metadata and evaluator planning.

## Kick-The-Tires Path

Command:

```bash
python scripts/run_artifact_smoke.py --skip-tests
```

Expected runtime: a few minutes after dependencies are installed.

Minimum practical resources:

- CPU: 4 commodity cores.
- RAM: 8 GB.
- Disk: the repository plus at least 5 GB free for temporary model outputs and
  Python package caches.
- GPU: not required.
- Network: not required after dependencies and the repository are available.
- GUI: not required.
- Paid services/API keys: not required.

This path is suitable for commodity evaluator machines and public research
infrastructure. It does not require SSH access to author-controlled hosts,
production-network access, special hardware, or private services.

The smoke path compiles code, validates the public sample HIB bundle in-place
and after extraction, recomputes public sample metrics, trains a tiny CCD model
from `examples/`, calibrates a fixed-FPR threshold into a self-contained model
bundle, refreshes `P_B` plus global/grouped thresholds from a clean benign
window, scores, explains, and certifies from that refreshed bundle with grouped
thresholds, and evaluates the bundled CAHO checkpoint.

## Release-Safe Paper Claim Check Runner

Command:

```bash
python scripts/run_artifact_claim_checks.py
```

Expected runtime: under a few minutes on commodity CPU hardware. This command
runs the readiness audit, all release-safe aggregate recomputation scripts, the
headline-claim audit, and the checked-in public sample fixed-FPR replay metric
check. It emits one JSON report with per-check commands, timing, summaries, and
failures.

## Full Test Path

Command:

```bash
python -m pytest -q
```

Expected runtime: under one hour on commodity CPU hardware after dependencies
are installed; current local runs are well below that.

## Source-Reachability Accounting

Command:

```bash
python scripts/recompute_source_reachability_metrics.py
```

Expected runtime: under a second. This recomputes the release-safe aggregate
CodeQL/Semgrep accounting from the paper's 50-repo static-analysis scope check.
End-to-end reruns require staging the public corpus manifest and labeled tool
outputs described in `source_reachability/README.md`.

## Method-Contract Accounting

Command:

```bash
python scripts/recompute_method_contracts.py
```

Expected runtime: under a second. This validates release-safe Table 1 and
Appendix C method contracts against the public code: CCD defaults, Eq. 1 score
path, fixed-FPR calibration, finite-edit closure, CAHO contrastive/binary-head
training support with explicit AdamW weight decay, and model-bundle
persistence.

## Paper-Claim Coverage Accounting

Command:

```bash
python scripts/recompute_paper_claim_coverage.py
```

Expected runtime: under a second. This validates the release-safe coverage
matrix for five contributions, Eq. 1, Proposition 5.1, Lemma C.1, Figures 1-7,
Tables 1-12, and Appendices A-F, and confirms that full-data or
production-private dependencies remain tied to manifest external completion
items.

## Paper Headline-Claim Accounting

Command:

```bash
python scripts/recompute_paper_headline_claims.py
```

Expected runtime: under a second. This centrally validates the paper's
high-visibility numeric anchors against release-safe aggregate artifacts:
replay scale, CCD fixed-FPR recall, latency and throughput, live-overlap added
value, label totals, decision-stability coverage, source/public scope, and
hostile-mimicry ranges.

## Public-Scope Taxonomy Accounting

Command:

```bash
python scripts/recompute_public_scope_metrics.py
```

Expected runtime: under a second. This recomputes the release-safe aggregate
public-report taxonomy accounting and verifies that public anchors are not
counted as HIB training or production positives. End-to-end reruns from
individual report labels require staging the JSONL export described in
`public_scope/README.md`.

## HIB Dataset-Profile Accounting

Command:

```bash
python scripts/recompute_hib_profile_metrics.py
```

Expected runtime: under a second. This validates release-safe aggregate source,
label, resolved/unresolved, event, repair, attack-family, and obfuscation
counts for the HIB production replay profile. Row-level reproduction requires
the full HIB-Real release bundle.

## Evaluation Accounting

Command:

```bash
python scripts/recompute_evaluation_accounting.py
```

Expected runtime: under a second. This validates Table 2 evidence units and
Appendix E/Table 11 reproducibility boundaries, including manifest alignment
for external full-replay, masked live-overlap, masked sink-evidence, and stress
output completion items.

## Local Latency Smoke

Command:

```bash
python scripts/benchmark_artifact_latency.py --num-samples 64 --repeats 1 --warmup 0
```

Expected runtime: a few seconds on commodity CPU hardware after dependencies are
installed. The command reports local encoder and CCD scoring-kernel timings.
These numbers are hardware-dependent and are not expected to reproduce the
paper's production latency table.

## Production-Latency Accounting

Command:

```bash
python scripts/recompute_production_latency_metrics.py
```

Expected runtime: under a second. This validates release-safe aggregate values
for Figure 5 full-path latency, single-host throughput, scoring-kernel timing,
and Table 5 CCD tail-latency alignment. Raw production serving traces are not
included in the public source artifact.

## Live-Overlap Accounting

Command:

```bash
python scripts/recompute_live_overlap_metrics.py
```

Expected runtime: under a second. This recomputes release-safe aggregate
CCD-vs-Regex/WAF live-overlap cells and derived reviewed-item PPV/review-load
metrics. End-to-end reruns from masked row-level labels require staging the
JSONL export described in `live_overlap/README.md`.

## Sink-Evidence Accounting

Command:

```bash
python scripts/recompute_sink_evidence_metrics.py
```

Expected runtime: under a second. This validates release-safe Table 8
controlled replay evidence for audited metadata-to-code paths, including
parser, persistence, consumer, effect, and detector-boundary fields. Raw
retained values, callback domains, private sink details, and production
side-effect traces are not included in the public source artifact.

## Paper Metric-Table Accounting

Command:

```bash
python scripts/recompute_paper_metric_tables.py
```

Expected runtime: under a second. This validates release-safe aggregate rows
for Tables 5, 6, 10, and 12 plus Appendix F synthetic-real summary metrics.
End-to-end reruns from row-level predictions or stress-test cases require the
full HIB-Real release and staged evaluation outputs described in
`paper_metric_tables/README.md`.

## Stability And Scope Accounting

Command:

```bash
python scripts/recompute_stability_scope_metrics.py
```

Expected runtime: under a second. This validates release-safe aggregate values
for Figure 7 certificate coverage, Figure 6 family-holdout/depth behavior,
benign-only drift refresh, independent replay tolerance, and public-real recall
ordering. End-to-end reruns require the full HIB-Real release and staged stress
outputs described in `stability_scope/README.md`.

## Archive Path

Command:

```bash
python scripts/build_artifact_archive.py
```

The archive builder verifies required manifest paths, excludes local virtualenvs,
caches, generated egg metadata, intermediate training checkpoints, and git
metadata, then writes a tarball, SHA-256 sidecar, and file-hash manifest for DOI
deposition.

## Full HIB-Real Replay

The checked-in sample bundle is not the paper-scale HIB replay. Reproducing
headline denominators and fixed-FPR metrics requires the separate full
de-identified HIB-Real release bundle described in `ARTIFACT_EVALUATION.md`.

Plan resources according to the final bundle size. For a 200,339,886-row JSONL
release, evaluators should use a machine with ample local SSD space for the
bundle, extracted JSONL, audit outputs, and temporary files. The replay scripts
are streaming CPU/IO workloads and do not require a GPU.

## Destructive Actions

The evaluator commands write only temporary outputs, explicit `--out` paths, and
the release audit files named in the commands. They do not modify private source
data or contact production services. Hostname strings may contain executable
syntax; do not paste benchmark strings into shells, SQL consoles, template
renderers, or network-enabled callback harnesses.
