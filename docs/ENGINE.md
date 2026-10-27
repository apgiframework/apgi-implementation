# Computation engine

How the register's `apgi-implementation – computation engine` items map onto code, what the
reference model is, and what the K7 desk-science chain found.

## Package map

| Register item | Where | Tests |
| --- | --- | --- |
| S1-IMP-01 Sequential latent-state estimation | `engine/latent_state.py` (bootstrap particle filter for the signal `S_t`) | `tests/test_engine.py::TestLatentState` |
| S1-IMP-02 Precision-weighted prediction errors | `engine/prediction_error.py` (pipeline steps 1–5 from `core` primitives) | `TestPredictionError` (parity with `APGIPipeline`) |
| S1-IMP-03 Somatic-marker gain, threshold adaptation | `engine/somatic.py` | `TestSomatic` |
| S1-IMP-04 Ignition probability, hard-threshold simulation | `engine/ignition_sim.py` | `TestIgnitionSim` |
| S1-IMP-05 Parameter recovery, sensitivity (hosts K7c) | `estimation/recovery_simulation.py`, `estimation/sensitivity.py` | `tests/test_estimation.py` |
| S1-IMP-06 Counterfactual / simulated conditions | `engine/counterfactual.py` (common random numbers across conditions) | `TestCounterfactual` |
| S1-IMP-07 Liquid-network and hierarchical variants | `models/pipeline_models.py` (`apgi_liquid`, `apgi_hierarchical`) | `tests/test_models.py` |
| S1-IMP-08 Provenance and deterministic replay | `engine/provenance.py` | `tests/test_reproducibility.py` |
| S1-IMP-09 PyMC estimation model in `estimation/` | `estimation/pymc_models.py`, `recovery_simulation.py`, `structural_identifiability.py`, `profile_likelihood.py` | `tests/test_estimation*.py` |
| S1-IMP-10 `adapters/` | `adapters/canonical_inputs.py`, `canonical_outputs.py`, `version_adapter.py` | `tests/test_protocol_compatibility.py` |
| S1-IMP-11 `runtime/` | `runtime/session_engine.py`, `checkpoint.py`, `batch_runner.py` | `tests/test_runtime.py` |
| S1-IMP-12 `model_registry/` | `model_registry/variants.py`, `parameter_mapping.py` | `tests/test_models.py` |
| S1-IMP-13 Test suite | `tests/test_reference_vectors.py`, `test_protocol_compatibility.py`, `test_reproducibility.py` | — |
| S1-IMP-14.1–14.7 Seven model classes | `models/` — `apgi_reference`, `apgi_reduced`, `apgi_extended`, `apgi_hierarchical`, `apgi_liquid`, `apgi_agent`, `apgi_null` | `tests/test_models.py` |
| S1-IMP-15 Qualify `apgi_reduced` (D14) | `qualification/reference.py`, `qualification/apgi_reduced.qualification.json` | `tests/test_reference_vectors.py` |
| S1-IMP-16 Run manifest with lattice stamps | `runtime/run_manifest.py` | `tests/test_protocol_compatibility.py::TestRunManifest` |
| S1-IMP-17 Shared ODE-based synthetic generator | `estimation/synthetic_generator.py` | `TestSyntheticGenerator` |
| S1-IMP-18 `α·Π` as `precision_weighted_learning_composite` | `estimation/composite.py` | `TestComposite` |
| S1-IMP-19 Acyclic graph, pinned packages, no copied models | `tools/check_dependencies.py` | `tests/test_dependency_policy.py` |
| S1-IMP-20 Retire duplicates through adapters | `adapters/version_adapter.py`, `DEPENDENCY-MATRIX.md` | `tests/test_dependency_policy.py` |
| S1-K7-01…05, 11, 15 | `estimation/structural_identifiability.py`, `recovery_simulation.py`, `gate_table.py`; driver `tools/run_k7.py`; output `gates/` | `tests/test_estimation.py`, `tests/test_tools.py` |
| XC-CI-02.1–.5 | `tests/test_ci_gates.py`, `.github/workflows/ci.yml`, `docs/CI-GATES.md` | — |

Dependency direction inside the repository: `core` ← `models` ← `model_registry` ← `engine`,
`estimation` ← `qualification`; `adapters` and `runtime` sit beside them. `tools/check_dependencies.py`
verifies there is no import-time cycle.

## The reference model: `apgi_reduced`

Per trial, with stimulus `x_t`:

```
ε_t      = x_t − x̂_t
drive_t  = tanh(γ_φ·Π·|ε_t|) + β                 γ_φ = 2
S_t      = (1−λ)·ρ^{B_{t−1}}·S_{t−1} + λ·drive_t  λ = 1−e^{−0.1}, ρ = 0.1
p_t      = σ((S_t − θ_t)/τ_σ)                    τ_σ = 0.5
B_t      = Bernoulli(p_t)   or   1[S_t > θ_t]
θ_{t+1}  = θ₀ + (θ_t − θ₀)·e^{−κ} + δ·B_t         κ = 0.15, δ = 0.5
x̂_{t+1}  = x̂_t + k·ε_t,        k = αΠ/(1+αΠ)
```

Estimable parameters: `α`, `Π`, `θ₀`, `β_DA`. Three observation channels: a behavioural report
`r_t ~ N(x̂_t, σ_r²)`, the ignition `B_t`, and an amplitude `y_t ~ N(S_t, σ_y²)`. Constants follow
`config.CONFIG` (`lam`, `RHO_RETAIN`, `kappa`, `delta`, `ignite_tau`).

**Status of this definition.** The register places the qualified reference model in
`mathematical-contracts.yaml` (S1-REG-04), which lives in the specification repository and is not
available here. The equations above are therefore this repository's explicit reading of
`apgi_reduced`, written to be checked line by line, and the qualification record certifies the
code against *these* equations. Treat every K7 result below as provisional until REG-04 is
qualified; if the contract differs, update `models/reduced.py` and re-run
`python -m qualification.reference --update` and `python tools/run_k7.py`.

## Qualification (D14)

`python -m qualification.reference` runs eight executable checks (registry authority, an
independent plain-float re-implementation, frozen reference vectors, the `αΠ` symmetry, the
constant-stimulus fixed point, signal bounds, determinism, ODE-generator parity) and compares the
result with the committed `qualification/apgi_reduced.qualification.json`. CI fails if the code and
the record drift apart. Exactly one registered variant has `authority = True`; the registry refuses
a second.

## K7: identifiability → recovery → gate table

```bash
python tools/run_k7.py --out gates --subjects 60            # DE-MC estimator (default)
python tools/run_k7.py --out /tmp/try --subjects 10 --estimator laplace   # quick look
```

* **K7a** (`structural_identifiability.py`): the exact `(α, Π) → (sα, Π/s)` symmetry of the
  behavioural output; sensitivity rank and condition number over a Latin-hypercube sweep of the
  plausible box; multi-start search for equivalent solutions; explicit tests of the suspect pairs
  `α–Π` and `θ₀–β` through partial collinearity of the sensitivity columns.
* **K7c** (`recovery_simulation.py`): 60 synthetic subjects drawn from the plausible box, generated
  by the ODE generator, fitted by differential-evolution MCMC. Reports recovery `r`, bias, nRMSE,
  95% coverage, pairwise posterior correlation, per tercile of the true value, and R̂/ESS
  separately. Criteria: `r ≥ 0.70`, pairwise `|r| < 0.60`, coverage in `[0.90, 0.98]`.
* **Gate table** (`gates/gate-table.yaml`): `k7a`, `k7b`, `k7c`, `exposable` per parameter and
  for the composite. `k7b` is empirical and stays `PENDING_DATA`; `exposable` is the desk-science
  verdict (K7a and K7c both `CONFIRMED`) and is provisional until K7b reports.

### What the analysis shows

* Behaviour constrains only `αΠ`: the symmetry gap is exactly 0, the sensitivity matrix has rank 1,
  and multi-start search finds equivalent solutions. Reporting `Π` alone from such a task would be
  reporting the prior. The composite `precision_weighted_learning_composite` is identifiable and
  recovered (r ≈ 0.997).
* Adding the ignition and amplitude channels makes the Jacobian full rank (median condition ≈ 28), but
  `α` and `Π` remain nearly collinear (median partial collinearity ≈ 0.98; they exceed the 0.99
  limit at about a fifth of the plausible range), and the posterior ridge couples them with `β`
  (pairwise `|r|` ≈ 0.9). Recovery correlations are high (r ≈ 0.9) yet the pairwise criterion fails,
  which is exactly the three-parameter trade-off that pairwise screening can miss.
* `θ₀` is the one individual parameter that clears every desk gate under the full design.
* Convergence and identifiability are reported separately. See `gates/gate-table.yaml` for the
  committed verdicts and `gates/k7c-results.json` for R̂/ESS.

### Limits

* The Laplace estimator (`--estimator laplace`) is a screening tool: it is local and under-covers
  along the `α–Π–β` ridge. Gate tables come from `mcmc`. PyMC NUTS is available
  (`estimation/pymc_models.py`, `pip install apgi[estimation]`) and is checked against the numpy
  likelihood in CI, but it is too slow to run per subject at K7c scale here.
* `estimation` needs only `numpy`/`scipy`; PyMC's current releases require numpy>=2 while this
  package caps numpy below 2 for the published figures, so install the extra into a separate
  environment (see `DEPENDENCY-MATRIX.md`).
* Noise levels `σ_r`, `σ_y` are treated as known (0.15, 0.10); the design is 240 trials with a
  reversal every 30. Other designs need their own K7 run.

## Run manifests

`runtime/run_manifest.py` builds the lattice-stamped manifest: `run_id`, `study`, protocol
id/version/hash, model id/version, commits (implementation, framework, spec), `dataset_hash`,
`seed`, `data_origin`, `mode`, `env_lock`, `gate_table` path + hash, `feature_version`,
`platform_release`. A run is marked `reproducible` only when nothing blocks it: unset seed,
unresolved or dirty commit, missing dataset or protocol hash, or an inaccessible gate table each
list a blocker. `verify_manifest` re-checks referenced files against their recorded hashes.
