# CI gates for apgi-implementation

Register item **XC-CI-02**. Every gate is its own job in `.github/workflows/ci.yml` so that
branch protection can require it by name. Each gate is a pytest marker (registered in
`pytest.ini`), so the same check runs locally with `pytest -m <marker>`.

| Job (required check name) | What it asserts | Local command |
| --- | --- | --- |
| `numerical parity` | `apgi_reduced` equals an independent plain-float re-implementation to 1e-12; the ODE generator equals the scorer to 1e-6; `apgi_reference` equals the pipeline it wraps; the error channel equals the pipeline's; the likelihood equals the particle-filter evidence at zero process noise; the qualification record is current. | `pytest -m parity` |
| `stability` | State stays finite and inside its analytic bounds for stimulus scales from 1e-12 to 1e8, at all 16 corners of the parameter support and over 50 000 steps; outputs are continuous in the parameters; perturbed initial states decay; estimation survives degenerate data (no ignitions, all ignitions, constant stimulus). | `pytest -m stability` |
| `performance` | Throughput floors, about 10x below measured speed: reference model > 50 000 steps/s, likelihood < 2 ms, Laplace fit < 10 s, ODE generator < 2 s, pipeline variant > 1 500 steps/s. Runs **without** coverage tracing. | `pytest -m performance` |
| `seeded runs are bit-for-bit reproducible` | Seeded runs are identical across processes and `PYTHONHASHSEED` values; checkpoint/resume equals an uninterrupted run; batch results do not depend on worker count. | `pytest tests/test_reproducibility.py tests/test_ci_gates.py -m "conformance or repeatability"` |
| `K7c recovery smoke test` | A 14-subject recovery run finishes in under a minute; the `αΠ` composite is recovered (r > 0.95) under the full design and under behaviour alone, `θ₀` and `β` only under the full design. | `pytest -m recovery_smoke` |

Supporting jobs that are also worth requiring: `dependency policy and qualification`
(`tools/check_dependencies.py`, `python -m qualification.reference`) and
`PyMC model matches the numpy reference` (`tests/test_estimation_pymc.py`, in its own
environment so the heavy PyMC install cannot disturb the other jobs).

## Making the gates required

"Required on protected branches" is a repository setting, not something a workflow file can
enforce. In GitHub: *Settings → Branches → Branch protection rule for `main` → Require status
checks to pass*, then add the job names in the table above (plus the existing `test`, `lint & types`,
`security`, `wheel installs and imports`).

## Thresholds are deliberate

* Performance floors catch an accidental quadratic or a lost vectorisation. Tighten them only
  after measuring on the CI runner class you actually use.
* The numerical tolerances (1e-12, 1e-6) are the stated agreement of the implementations
  involved, not tuned to pass; loosening one is a change to the model's contract.
