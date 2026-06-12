# Method Contract Checks

This directory contains release-safe checks for Table 1 and Appendix C method
contracts. It is not a raw replay artifact; it validates the public code and
configuration surface that implements the paper's CCD, CAHO, calibration, and
certification assumptions.

Run:

```bash
python scripts/recompute_method_contracts.py
```

The checker verifies the Table 1 contract rows, default CCD scoring
configuration, fixed-FPR calibration defaults plus tenant/window grouped
threshold lookup, edit-manifest coverage,
deterministic finite-edit certificate closure, CAHO two-view contrastive
training plus an L2-normalized binary head with explicit AdamW weight decay,
Appendix C CAHO deployed-recipe optimizer defaults in the benchmark binary trainer,
94 GB CUDA batch defaults for replay-scale actual and regular GradCache CAHO
training,
validation-only fixed-FPR score provenance and best-epoch restoration,
exact full-axis scanning for the deployed top-R cone sketch that bypasses LSH
by default for certification/calibration,
training-time validation of cone axes, prior smoothing floors, and benign or
malicious prior embeddings,
score-path rejection of non-finite, zero-norm, or dimension-incompatible
embeddings,
calibrated-margin certificates with deterministic enumeration fallback, CAHO
supervised orbit contrastive training with benign diversity preserved,
grouped-threshold explanations for decision inspection, benign-only
`(P_B, tau_alpha)` refresh that rejects non-finite benign embeddings before
mutating detector state, and model-bundle persistence of axes, priors, config,
and optional global/grouped calibrated thresholds. Bundle loading, saving, and
in-memory detector construction also reject mismatched cone configuration,
non-unit or shape-incompatible cone axes, malformed prior arrays, invalid
effective counts, bad thresholds, and mixture weights that do not match the
serialized malicious priors before a detector can emit scores.

The 94 GB training defaults are `batch_size=16384` for the actual non-GradCache
objective and `batch_size=49152` with `grad_cache_chunk_size=8192` for regular
GradCache CAHO training. The Appendix C batch-size value remains recorded as
the paper deployed recipe and can be passed explicitly when reproducing that
exact setting.

The paper's full deployed training/replay provenance still requires the full
HIB-Real release and associated run outputs. This checker keeps the public
source artifact honest about which method contracts are implemented locally.
