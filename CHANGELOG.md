# Changelog

All notable changes to `apgi` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

Computation-engine work for S1 (register items `S1-IMP-01`…`S1-IMP-20`, `S1-K7-*`, `XC-CI-02`).
See `docs/ENGINE.md` for the item-by-item map.

### Added
- `models/`: seven model classes behind one interface. `apgi_reduced` is the single qualified
  numerical reference (decision D14); `apgi_reference`, `apgi_extended`, `apgi_hierarchical`,
  `apgi_liquid`, `apgi_agent` wrap the pipeline; `apgi_null` is the benchmark.
- `model_registry/`: variant registry (refuses a second authoritative variant) and parameter
  mapping across spec symbols, reduced-model names and pipeline keys.
- `engine/`: precision-weighted prediction errors, somatic gain and threshold adaptation,
  ignition probability and hard-threshold simulation, bootstrap-filter latent-state estimation,
  counterfactual conditions on common random numbers, provenance records with deterministic replay.
- `estimation/` (decision D2): numpy likelihood and priors, MAP / Laplace / differential-evolution
  MCMC, PyMC transcription (optional, checked against the numpy likelihood), ODE-based synthetic
  generator, K7a structural identifiability, profile likelihood, K7c recovery simulation,
  sensitivity analysis, the `precision_weighted_learning_composite` estimator and the gate table.
- `adapters/`: canonical inputs and outputs, and a version adapter that retires legacy names with
  `DeprecationWarning`s (`beta`, `ignite_tau`, `strict_mode`, `ai_F_values`).
- `runtime/`: session engine, atomic checksummed checkpoints with deterministic-replay fallback,
  scheduling-independent batch runner, lattice-stamped run manifest.
- `qualification/`: executable qualification of `apgi_reduced` and its committed record.
- `tools/check_dependencies.py` (acyclic imports, pins, no copied models, deprecation drift) and
  `tools/run_k7.py`; `DEPENDENCY-MATRIX.md`; `gates/` (K7 results and `gate-table.yaml`).
- CI: gates for numerical parity, stability, performance, seeded repeatability and K7c recovery
  smoke, plus dependency-policy and PyMC jobs (`docs/CI-GATES.md`).

### Changed
- Installation: dependency caps relaxed (`numpy>=1.24,<3.0`, `pandas<4`, `structlog<27`, `ruff`/`mypy`
  as ranges). The old `numpy<2.0` cap forced a source build of numpy 1.26 on current Python releases and
  uninstalled the user's numpy 2.x, breaking scipy and matplotlib. The suite passes on numpy 2.5 / scipy
  1.18 / pandas 3.1 / pytest 9.
- `install.sh` (and `make install`): one-step venv install for macOS/Linux with a smoke test.
- CI: coverage floor 85%; ruff-format, mypy, pip-audit and the performance job are advisory;
  third-party deprecation warnings no longer fail tests; macOS install job.
- `pyproject.toml`: new packages in the wheel; upper bounds on the optional `scipy`/`matplotlib`
  requirements (they were the only unbounded dependencies); `estimation` extra.
- `.coveragerc`: `estimation/pymc_models.py` is covered by its own CI job, not the main one.

## [1.1.0] - 2026-10-17

Release aligned with `apgi-framework` 1.1.0.

- `pyproject.toml`, `CITATION.cff`, `README.md` (BibTeX) and `main.py`: version 1.1.0. The CLI reads
  its version from a single `APGI_VERSION` constant instead of two hard-coded strings.
- Project name unified as Allostatic Precision-Gated Ignition. `main.py` previously said
  "Adaptive Precision Gated Ignition" in its docstring and `--help` text.
- `pyproject.toml`: removed the `orcid` key from `project.authors`. PEP 621 allows only `name` and
  `email`, so setuptools rejected the file and `pip install -e .` failed. The ORCID stays in
  `CITATION.cff`.
- Authorship: `pyproject.toml` and `CITATION.cff` name Dimitry Pesochin (with ORCID in
  `CITATION.cff`) in place of the generic "APGI Team" / "APGI Framework contributors".
- `requirements.txt` added with the same bounds as `pyproject.toml`; `numpy` capped at `<2.0`
  for reproducibility of the published figures.
- README: citation and archival sections revised.
- `tests/test_release_metadata.py` (new): fails if the version in `pyproject.toml`,
  `CITATION.cff`, the README BibTeX, this changelog and `main.py` disagree.

## [1.0.0] - 2026-07-29

First DOI-archived release ([10.5281/zenodo.21632264](https://doi.org/10.5281/zenodo.21632264)).
