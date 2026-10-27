# APGI Implementation — Makefile
# Reproduces all figures referenced in Paper 1 of the APGI series.
#
# Usage:
#   make reproduce-paper   — generate all paper figures into outputs/
#   make figure-s1         — Figure S1 only (onset timeline schematic)
#   make figure-07         — Hierarchical system plots
#   make figure-08         — Spectral validation plots
#   make test              — run full test suite
#   make clean             — remove pycache and temporary files
#   make clean-outputs     — remove generated outputs

PYTHON      ?= python3
OUTPUT_DIR   = outputs
EXAMPLES_DIR = examples

.PHONY: install install-dev reproduce-paper manifest figure-s1 figure-07 figure-08 test lint clean clean-outputs

## One-step install into ./.venv (see install.sh)
install:
	./install.sh

install-dev:
	./install.sh --dev

## Primary target: reproduce all paper figures
reproduce-paper: $(OUTPUT_DIR)
	@echo "================================================================"
	@echo "Reproducing APGI Paper 1 figures"
	@echo "Output directory: $(OUTPUT_DIR)/"
	@echo "================================================================"
	$(PYTHON) $(EXAMPLES_DIR)/16_figures.py
	$(PYTHON) $(EXAMPLES_DIR)/07_hierarchical_system.py
	$(PYTHON) $(EXAMPLES_DIR)/08_spectral_validation.py
	$(MAKE) manifest
	@echo "================================================================"
	@echo "All figures written to $(OUTPUT_DIR)/"
	@echo "Provenance: $(OUTPUT_DIR)/run.manifest.json"
	@echo "================================================================"

## Provenance record for the figure run: seed, resolved config + hash, git
## commit, interpreter and dependency versions. Written alongside the figures so
## any published output can be traced back to the code that produced it.
manifest: $(OUTPUT_DIR)
	$(PYTHON) -c "from core.config_io import load_config; from core.manifest import write_manifest; \
	  p = write_manifest('$(OUTPUT_DIR)/run.manifest.json', load_config('configs/prod.toml', use_env=False), \
	  extra={'target': 'reproduce-paper'}); print(f'wrote {p}')"

## Individual figure targets
## NOTE: the generator is examples/16_figures.py; it WRITES outputs/figure_s1.png.
## The Makefile previously invoked examples/figure_s1.py, a file that has never
## existed, so `make reproduce-paper` aborted on its first command.
figure-s1: $(OUTPUT_DIR)
	$(PYTHON) $(EXAMPLES_DIR)/16_figures.py

figure-07: $(OUTPUT_DIR)
	$(PYTHON) $(EXAMPLES_DIR)/07_hierarchical_system.py

figure-08: $(OUTPUT_DIR)
	$(PYTHON) $(EXAMPLES_DIR)/08_spectral_validation.py

## Test suite (branch coverage, matching CI)
test:
	$(PYTHON) -m pytest tests/ --cov --cov-branch --cov-report=term-missing

## All quality gates, matching CI
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m mypy .
	$(PYTHON) -m bandit -r . -c bandit.yaml -ll --exclude ./tests,./build,./dist

## Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

## Clean generated outputs
clean-outputs:
	rm -rf $(OUTPUT_DIR)

## Clean pycache and temporary files
clean:
	$(PYTHON) delete_pycache.py --yes
