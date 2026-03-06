.PHONY: all dev extract steer baselines evaluate visualize clean setup

CONFIG ?= configs/default.yaml
MODEL ?=
PERSONA ?=

export PYTHONPATH := $(CURDIR)

# Optional filters
MODEL_FLAG = $(if $(MODEL),--model $(MODEL),)
PERSONA_FLAG = $(if $(PERSONA),--persona $(PERSONA),)

# ---- Full Pipeline ----

all: extract baselines steer evaluate

# ---- Development (single model, single persona) ----

dev:
	$(MAKE) extract CONFIG=configs/dev.yaml
	$(MAKE) baselines CONFIG=configs/dev.yaml
	$(MAKE) steer CONFIG=configs/dev.yaml
	$(MAKE) evaluate CONFIG=configs/dev.yaml

# ---- Individual Phases ----

extract:
	python scripts/run_extraction.py --config $(CONFIG) $(MODEL_FLAG) $(PERSONA_FLAG)

steer:
	python scripts/run_steering.py --config $(CONFIG) $(MODEL_FLAG) $(PERSONA_FLAG)

baselines:
	python scripts/run_baselines.py --config $(CONFIG) $(MODEL_FLAG) $(PERSONA_FLAG)

evaluate:
	python scripts/run_evaluation.py --config $(CONFIG)

evaluate-classifier:
	python scripts/run_evaluation.py --config $(CONFIG) --tier classifier

evaluate-judge:
	python scripts/run_evaluation.py --config $(CONFIG) --tier judge

# ---- Setup ----

setup:
	pip install -r requirements.txt

# ---- SLURM ----

slurm-all:
	bash slurm/run_all.sh

slurm-model:
	MODEL=$(MODEL) bash slurm/run_all.sh

# ---- Cleanup ----

clean:
	rm -rf results/vectors results/generations results/evaluation results/logs
