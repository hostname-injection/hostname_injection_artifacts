SHELL := /bin/sh

MODEL ?= ccd_model.npz
INPUT ?= data/input.txt
OUTPUT ?= out/scores.csv
CHECKPOINT ?= caho_encoder
BATCH ?= 128
BASELINES ?=
BASELINE_SAMPLE ?= 5000

.PHONY: help diagnose explain score test artifact-smoke artifact-latency train-caho-corpus baselines

help:
	@echo "Targets:"
	@echo "  diagnose          Show device + throughput diagnostics"
	@echo "  explain           Explain CCD predictions for INPUT"
	@echo "  score             Score INPUT with MODEL"
	@echo "  test              Run pytest"
	@echo "  artifact-smoke    Run evaluator smoke test"
	@echo "  artifact-latency  Run local hardware-dependent latency smoke benchmark"
	@echo "  train-caho-corpus Train CAHO encoder from corpus"
	@echo "  baselines         Run baseline sweep (BASELINES=$(BASELINES), BASELINE_SAMPLE=$(BASELINE_SAMPLE))"
	@echo ""
	@echo "Variables: MODEL, INPUT, OUTPUT, CHECKPOINT, BATCH, BASELINES, BASELINE_SAMPLE"

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

artifact-latency:
	python scripts/benchmark_artifact_latency.py --checkpoint $(CHECKPOINT)

train-caho-corpus:
	python -m ccd.cli train-caho-corpus --benign-dir txt_corpus/benign --malicious-jsonl-dir filtered_corpus --malicious-txt-dir txt_corpus/varied --out out/caho_model_checkpoint

baselines:
	python -m baselines.run_baselines --baselines $(BASELINES) --sample-per-class $(BASELINE_SAMPLE)
