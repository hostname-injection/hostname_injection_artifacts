# IEEE S&P Artifact Badge Readiness

This checklist maps the IEEE S&P 2027 artifact instructions to the current
artifact state. It is meant to be read before `ARTIFACT_EVALUATION.md` when
deciding what is already proven by the repository and what remains an external
publication step.

## Available

IEEE S&P criterion: the artifact must be permanently and publicly retrievable,
with final permanent storage backed by a DOI.

Current evidence:

- `scripts/build_artifact_archive.py` builds a DOI-ready tarball, SHA-256
  sidecar, and file-hash manifest.
- `metadata.template.toml` contains the packaging metadata shape and explicit
  URL placeholders.
- `ARTIFACT_MANIFEST.json` lists required files and external completion items.
- `ARTIFACT_MANIFEST.json` records the IEEE Available criterion explicitly:
  permanent public retrieval and DOI-backed storage are required, while the
  current repository state is only DOI-prepared until the external deposit
  happens.
- `scripts/audit_artifact_readiness.py --strict-final` intentionally fails
  while DOI placeholders or external completion items remain, so final badge
  readiness is not confused with local functional readiness.

Current status: locally prepared, not complete. The final DOI deposit and URL
replacement must happen outside this worktree.

## Functional

IEEE S&P criterion: the artifact should conform to the paper's functionality,
usability, and relevance expectations, and should work on machines other than
the authors' machine.

Current evidence:

- Documentation: `README.md`, `ARTIFACT_EVALUATION.md`,
  `ARTIFACT_RESOURCES.md`, and this checklist define installation, smoke tests,
  claim mapping, expected outputs, resources, and limitations.
- The manifest has machine-readable Functional evidence for the three IEEE
  subcriteria: documentation, completeness, and exercisability. The readiness
  audit fails if any subcriterion or evidence path is missing.
- Public infrastructure: the evaluator path is a source-code artifact that runs
  on commodity CPU Linux/macOS hosts or public research infrastructure without
  SSH service, GUI, paid API, special hardware, GPU, or production-network
  access.
- Completeness: code covers CCD scoring/calibration, including grouped
  tenant/window thresholds, CAHO augmentation and training, finite-edit
  decision-stability checks, HIB de-identification gates,
  Table 1/Appendix C method-contract accounting,
  paper-claim coverage accounting,
  headline numeric claim accounting,
  replay metric recomputation, HIB dataset-profile aggregate accounting,
  evaluation-unit/reproducibility-boundary accounting,
  benchmark Dataset wrappers, baseline runners, source-reachability scope-check
  accounting, and public-scope taxonomy accounting, live-overlap aggregate
  accounting, Table 8 sink-evidence accounting, paper metric-table aggregate
  accounting, stability/drift/scope aggregate accounting, production-latency
  aggregate accounting, plus a local latency smoke benchmark for the encoder
  and CCD scoring kernel.
- Exercisability: `scripts/run_artifact_smoke.py --skip-tests` compiles code,
  validates the public bundle in-place and after extraction, recomputes sample
  metrics, trains a tiny CCD model, calibrates a fixed-FPR threshold into a
  self-contained model bundle, scores from that bundle, writes explanations and
  certificate records, and evaluates the CAHO checkpoint.
- Release-safe claim checks: `scripts/run_artifact_claim_checks.py` runs the
  readiness audit, aggregate recomputation scripts, headline-claim audit, and
  checked-in public sample fixed-FPR replay metrics from one evaluator command.
- Portability: benchmark scripts use relative paths or `HIB_BENCHMARK_ROOT` /
  `HIB_SOURCE_ROOT` environment variables rather than author-local paths.
- Runtime: the kick-the-tires path runs in minutes on a CPU-only laptop; the
  full test suite is well below the one-day evaluation limit on commodity CPU
  hardware.
- Packaging: `metadata.template.toml` follows the IEEE S&P artifact packaging
  script shape for HotCRP submission, and `scripts/build_artifact_archive.py`
  builds the source bundle intended for DOI deposition.
- Tracking: the artifact is source/data/scripts only and does not embed web
  analytics or tracking code for evaluator access; the readiness audit scans
  repository text for common web-tracking endpoints and snippets.

Current status: locally supported. The latest local evidence is recorded in
`ARTIFACT_EVALUATION.md`.

## Reproduced

IEEE S&P criterion: evaluators should be able to independently obtain results
that support the paper's main claims, with scaled-down experiments acceptable
for lengthy work when their significance is clearly explained.

Current evidence:

- The checked-in 150-row public bundle proves de-identification mechanics,
  release gates, fixed-FPR replay-metric recomputation over reviewed public
  scores and release-safe public calibration groups, and privacy boundaries.
- The manifest records the IEEE Reproduced criterion explicitly: main-result
  support, independent replay path, tolerance/scaled-down-experiment
  justification, and the external full-data boundary are all checked by the
  readiness audit.
- The full replay path in `ARTIFACT_EVALUATION.md` describes how to validate
  the full HIB-Real release and recompute the paper-scale fixed-FPR counts.
- The method-contract script validates Table 1 and Appendix C CCD/CAHO
  assumptions directly against the public code, grouped calibration,
  CAHO contrastive/binary-head training support, explicit AdamW weight decay,
  and bundle format.
- The paper-claim coverage script maps five contributions, Eq. 1,
  Proposition 5.1, Lemma C.1, Figures 1-7, Tables 1-12, and Appendices A-F
  to artifact evidence or explicit external completion items.
- The headline-claim script centrally validates the paper's most visible
  numeric anchors from the release-safe aggregate files: replay scale,
  fixed-FPR recall, latency, live-overlap added value, label totals,
  decision-stability coverage, scope checks, and hostile-mimicry ranges.
- The HIB profile script validates release-safe aggregate source, label,
  resolved/unresolved, repair, attack-family, and obfuscation counts for the
  paper's production-replay profile.
- The evaluation-accounting script validates Table 2 denominator units and
  Appendix E/Table 11 released-versus-withheld boundaries against the manifest.
- The source-reachability script recomputes the paper's aggregate CodeQL and
  Semgrep accounting, with a JSONL path for staged public-corpus finding labels.
- The public-scope script recomputes the 118/127 public-report taxonomy
  accounting and confirms public anchors are not counted as HIB positives.
- The live-overlap script recomputes release-safe Table 7 aggregate overlap
  accounting and derived reviewed-item PPV/review-load metrics.
- The sink-evidence script validates the three Table 8 controlled replay traces
  and explicitly records them as evidence rather than production-compromise
  claims.
- The paper metric-table script validates release-safe aggregate metrics for
  Table 5 baseline audit, Table 6 ablations, Table 10 LLM checkpoint
  identifiers, Table 12 hostile mimicry, and Appendix F synthetic-real
  validity.
- The stability/scope script validates release-safe aggregate metrics for
  Figure 6 family-holdout/depth behavior, Figure 7 finite-edit certificate
  coverage, drift-refresh invariants, and public-real recall ordering.
- The production-latency script validates release-safe Figure 5 and Table 5
  latency/throughput aggregates while keeping raw serving traces private.
- The paper-scale replay remains intentionally external because the full
  200,339,886-row HIB-Real release is too large for the source repository.

Current status: not complete from this worktree alone. Reproduced-badge proof
requires staging the full HIB-Real de-identified release, rerunning the replay
commands, and archiving the resulting metrics.
