# K7 results of 27 October 2026 (archived, superseded gate rules)

These files are the K7 identifiability and recovery results and the gate table
that `tools/run_k7.py` produced on 27 October 2026, on branch
`claude/awesome-hypatia-2q329f` (commit f596c26). They never reached `main`
before that branch was retired.

They are archived, not live, because PR #3 later changed the gate rules: the
current schema carries a `desk_exposable` flag per parameter and a stricter
allow list, and `tests/test_tools.py::TestCommittedGateTable` rejects this
table under those rules (it marks `theta_base` and others exposable). Read
these numbers as a record of the earlier run, not as the current gate decision.

To produce a live gate table, run K7 again on the current tree:

```bash
python tools/run_k7.py --out gates --subjects 60
```

That writes `gates/gate-table.{json,yaml}`, `gates/k7-manifest.json` and the
K7a/K7c results, which the committed-gate-table tests then check.
