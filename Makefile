SHELL := /bin/sh

MODEL ?= ccd_model.npz
INPUT ?= data/input.txt
OUTPUT ?= out/scores.csv
CHECKPOINT ?= sentence-transformers/all-MiniLM-L6-v2
PER_CLASS ?= 50
BATCH ?= 128
BASELINES ?=
BASELINE_SAMPLE ?= 5000

.PHONY: help sanity diagnose explain score test artifact-smoke artifact-claim-checks artifact-audit artifact-archive artifact-latency artifact-method-contracts artifact-paper-coverage artifact-paper-headlines artifact-hib-profile artifact-evaluation-accounting artifact-live-overlap artifact-sink-evidence artifact-metric-tables artifact-stability-scope artifact-production-latency train-caho-corpus train-user-logins baselines

help:
	@echo "Targets:"
	@echo "  sanity            Run user_login sanity check (PER_CLASS=$(PER_CLASS))"
	@echo "  diagnose          Show device + throughput diagnostics"
	@echo "  explain           Explain CCD predictions for INPUT"
	@echo "  score             Score INPUT with MODEL"
	@echo "  test              Run pytest"
	@echo "  artifact-smoke    Run evaluator smoke test"
	@echo "  artifact-claim-checks Run all release-safe paper claim checks"
	@echo "  artifact-audit    Check manifest, public gates, and public privacy wording"
	@echo "  artifact-archive  Build DOI-ready tarball in dist/"
	@echo "  artifact-latency  Run local hardware-dependent latency smoke benchmark"
	@echo "  artifact-method-contracts Recompute release-safe Table 1/Appendix C method contracts"
	@echo "  artifact-paper-coverage Recompute named paper-claim coverage matrix"
	@echo "  artifact-paper-headlines Recompute headline paper-claim numeric anchors"
	@echo "  artifact-production-latency Recompute release-safe production latency accounting"
	@echo "  artifact-hib-profile Recompute release-safe HIB profile aggregate accounting"
	@echo "  artifact-evaluation-accounting Recompute release-safe Table 2/Table 11 accounting"
	@echo "  artifact-live-overlap Recompute release-safe live-overlap aggregate accounting"
	@echo "  artifact-sink-evidence Recompute release-safe Table 8 sink-evidence accounting"
	@echo "  artifact-metric-tables Recompute release-safe paper metric-table accounting"
	@echo "  artifact-stability-scope Recompute release-safe stability/drift/scope accounting"
	@echo "  train-caho-corpus Train CAHO encoder from corpus"
	@echo "  train-user-logins Train CCD model from user_logins"
	@echo "  baselines         Run baseline sweep (BASELINES=$(BASELINES), BASELINE_SAMPLE=$(BASELINE_SAMPLE))"
	@echo ""
	@echo "Variables: MODEL, INPUT, OUTPUT, CHECKPOINT, PER_CLASS, BATCH, BASELINES, BASELINE_SAMPLE"

sanity:
	python -m ccd.sanity --per-class $(PER_CLASS)

diagnose:
	python -m ccd.diagnostics --checkpoint $(CHECKPOINT) --batch-size $(BATCH)

explain:
	python -m ccd.explain --model $(MODEL) --input $(INPUT)

score:
	python -m ccd.score_cli --model $(MODEL) --input $(INPUT) --output $(OUTPUT) --batch-size $(BATCH)

test:
	pytest

artifact-smoke:
	python scripts/run_artifact_smoke.py

artifact-claim-checks:
	python scripts/run_artifact_claim_checks.py

artifact-audit:
	python scripts/audit_artifact_readiness.py

artifact-archive:
	python scripts/build_artifact_archive.py

artifact-latency:
	python scripts/benchmark_artifact_latency.py

artifact-method-contracts:
	python scripts/recompute_method_contracts.py

artifact-paper-coverage:
	python scripts/recompute_paper_claim_coverage.py

artifact-paper-headlines:
	python scripts/recompute_paper_headline_claims.py

artifact-production-latency:
	python scripts/recompute_production_latency_metrics.py

artifact-hib-profile:
	python scripts/recompute_hib_profile_metrics.py

artifact-evaluation-accounting:
	python scripts/recompute_evaluation_accounting.py

artifact-live-overlap:
	python scripts/recompute_live_overlap_metrics.py

artifact-sink-evidence:
	python scripts/recompute_sink_evidence_metrics.py

artifact-metric-tables:
	python scripts/recompute_paper_metric_tables.py

artifact-stability-scope:
	python scripts/recompute_stability_scope_metrics.py

train-caho-corpus:
	python -m ccd.cli train-caho-corpus --benign-dir txt_corpus/benign --malicious-jsonl-dir filtered_corpus --malicious-txt-dir txt_corpus/varied --out out/caho_model_checkpoint

train-user-logins:
	python -m ccd.cli train-user-logins --user-logins-dir hostname_injection_benchmark/user_logins --out $(MODEL)

baselines:
	python -m baselines.run_baselines --baselines $(BASELINES) --sample-per-class $(BASELINE_SAMPLE)
