# Changelog

All notable changes to `apgi` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Audit remediation (APGI 1.2 Software Validation Specification)

Closes the code defects recorded in the 2026-11-05 audit register and advances
the requirement items they block. Item-by-item status: `docs/AUDIT-STATUS.md`.
Regression tests: `tests/test_audit_defects.py`.

#### Fixed
- AUD-DEF-01: NaN/inf inputs are rejected before any state, RNG or history
  change in `APGIPipeline.step`, every model `step`/`run`, `SessionEngine.advance`
  and the fitting boundary (`core/finite.py`); `ReducedParams` rejects non-finite
  `sigma_r`/`sigma_y`.
- AUD-DEF-02: run manifests are reproducible only with verified identities
  (digest/commit formats, registered model version, gate-table content, dataset
  provenance, verified protocol lock); `verify_manifest` re-verifies them.
- AUD-DEF-03: `APGIPipeline.validate()` reports `partial` (with
  `failed_analyses`, `certifiable: false`) when a sub-analysis fails.
- AUD-DEF-04: composite αΠ standard error used SSE curvature without the 1/(2σ²)
  factor (intervals ~0.71× too narrow); now √2·σ_r/√SSE″, boundary optima flagged.
- AUD-DEF-05: checkpoints/records from another model or software version, or
  with inconsistent history, are refused on resume/replay.
- AUD-DEF-06: batch cache is bound to code/model/environment identity, carries a
  document digest and is re-derived from stored outputs; rejected files are
  reported in `BatchResult.invalidated`. Cache format 2 (v1 files are re-run).
- AUD-DEF-07: qualification vectors moved into the `qualification` package and
  shipped as package data; CI qualifies the installed wheel outside the tree.
- AUD-DEF-08: three mypy errors (no numerical change).
- AUD-DEF-09: intermittent `test_pipeline_bold_calibration` failure — the test
  was unseeded and set a non-existent key; now seeded and exercises the real
  Landauer/BOLD floor.
- AUD-DEF-10: security CI audits the built wheel's resolved runtime dependency
  set, emits a CycloneDX SBOM, runs a content secret scan, retains artefacts.
- AUD-DEF-11: `expose_factors` has no caller bypass; production exposure needs an
  approved gate table with K7a, K7b and K7c CONFIRMED. Desk verdicts move to
  `desk_exposable` / `provisional_allow_list`.
- AUD-DEF-12: EEG loader selects its parser by format, refuses pickles, separates
  missing MNE from parse failures and records file provenance.
- AUD-K7-CONVERGENCE: composite K7c verdict requires converged α and Π chains.

#### Added
- `runtime/protocol_lock.py`: fail-closed protocol definition verification (REP-001).
- `adapters/quarantine.py`: record screening, named outcomes, reconciliation (DAT-002).
- `hierarchy/levels.py`: canonical L0–L4 labels, bands and index convention (SCI-003).
- `model_registry/default_provenance.py`: provenance basis for every default (SCI-005).
- Canonical output schema v2 with `protocol_version` and `producer`;
  `exchange_payload()` refuses payloads missing them (INT-001). Energy units declared.
- Draft governance documents: `docs/APPLICATION-VALIDATION-PROFILE.md`,
  `docs/TRACEABILITY.md`, `docs/CLAIM-REGISTER.md`, `docs/COMPATIBILITY-MATRIX.md`,
  `docs/RELEASE.md`, `docs/AUDIT-STATUS.md` — none approved yet.

#### Changed
- `mode="confirmatory"` run manifests are refused without a verified protocol lock;
  `core.manifest` labels runs `exploratory` with `registered_plan_verified: false`.
- Classifier lowered from Production/Stable to Beta (no release certificate yet).
- `docs/ENGINE.md` marks K7 numbers as provisional and unretained.

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
