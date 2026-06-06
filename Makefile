SHELL := /bin/sh

PYTHON ?= python3
MODEL ?= ccd_model.npz
INPUT ?= data/input.txt
OUTPUT ?= out/scores.csv
CHECKPOINT ?= caho_encoder
BATCH ?= 128

.PHONY: help diagnose explain score test artifact-readiness artifact-smoke artifact-latency train-caho-corpus

help:
	@echo "Targets:"
	@echo "  diagnose          Show device + throughput diagnostics"
	@echo "  explain           Explain CCD predictions for INPUT"
	@echo "  score             Score INPUT with MODEL"
	@echo "  test              Run pytest"
	@echo "  artifact-readiness Run fast reviewer artifact readiness checks"
	@echo "  artifact-smoke    Run evaluator smoke test"
	@echo "  artifact-latency  Run local hardware-dependent latency smoke benchmark"
	@echo "  train-caho-corpus Train CAHO encoder from corpus"
	@echo ""
	@echo "Variables: PYTHON, MODEL, INPUT, OUTPUT, CHECKPOINT, BATCH"

diagnose:
	$(PYTHON) -m ccd.diagnostics --checkpoint $(CHECKPOINT) --batch-size $(BATCH)

explain:
	$(PYTHON) -m ccd.cli explain --model $(MODEL) --input $(INPUT)

score:
	$(PYTHON) -m ccd.cli score --model $(MODEL) --input $(INPUT) --output $(OUTPUT) --batch-size $(BATCH)

test:
	$(PYTHON) -m pytest

artifact-readiness:
	$(PYTHON) scripts/check_artifact_readiness.py

artifact-smoke:
	$(PYTHON) scripts/run_artifact_smoke.py

artifact-latency:
	$(PYTHON) scripts/benchmark_artifact_latency.py --checkpoint $(CHECKPOINT)

train-caho-corpus:
	$(PYTHON) -m ccd.cli train-caho-corpus --benign-dir txt_corpus/benign --malicious-jsonl-dir filtered_corpus --malicious-txt-dir txt_corpus/varied --out out/caho_model_checkpoint
