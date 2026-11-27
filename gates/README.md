# K7 gate table (live)

Produced on 27 November 2026 by

```bash
python tools/run_k7.py --out gates --subjects 60
```

(defaults: MCMC, 3000 draws and 1500 warm-up per chain, 100 sweep points, 16 starts,
seed 20261116) on commit 5802bf2, a clean tree (`k7-manifest.json`: `"dirty": false`).
The model, models/reduced.py, is unchanged since the 27 October run archived in
`archive/2026-10-27/`; this run re-derives the gate decisions under the current gate
rules (`tests/test_tools.py::TestCommittedGateTable`).

Determinism: a first run on 67a6c3f (same code path, seed and settings) produced
identical K7a and K7c results (sha256 630cff21… and 938a6026…); only that run's
manifest differed, because it was marked dirty by its own output files (fixed in
5802bf2, F21).

Reading the table: K7b is `PENDING_DATA`, so no parameter is production-exposable and
the allow list is empty. Desk-exposable (K7a and K7c both CONFIRMED, recovery and
coverage criteria met): `theta_base` and `precision_weighted_learning_composite`.
`alpha` and `pi` are individually non-identifiable (the registered α–Π degeneracy);
`beta` passes K7a but fails K7c. The mathematical-contracts file (S1-REG-04) is not yet
qualified, so the table is labelled provisional. These are properties of the reduced
model and the synthetic design, not evidence for or against APGI.
