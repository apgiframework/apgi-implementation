# Release compatibility matrix (draft)

**Status: DRAFT (AUD-REL-004).** APGI-REL-004 requires a *tested* matrix that
relates each release to the protocol bundle, canonical schemas, reference
revisions and papers/archives. The version domains below are independent; a
different number in each is expected, not an error. Only combinations marked
**tested** have evidence in CI; everything else is declared intent.

## Version domains in this repository

| Domain | Current value | Where defined | Semantic-change rule (proposed) |
|---|---|---|---|
| Package (`apgi`) | 1.1.0 | `pyproject.toml` | SemVer: major = incompatible API/numerics, minor = additive, patch = fixes with identical reference outputs |
| Config schema / version adapter | 1.1.0 | `adapters/version_adapter.py` `SPEC_VERSION` | Bumped when config keys are renamed/removed (migrations required) |
| Reference model `apgi_reduced` | 1.0.0 | `models/reduced.py` | Any change to reference outputs ⇒ major; checkpoints/records from another version are refused (AUD-DEF-05) |
| Canonical output schema | 2 | `adapters/canonical_outputs.py` | Bumped to 2 by adding `protocol_version` + `producer` (APGI-INT-001) |
| Run manifest schema | 1 | `runtime/run_manifest.py` | — |
| Checkpoint format | 1 | `runtime/checkpoint.py` | — |
| Batch cache format | 2 | `runtime/batch_runner.py` | v1 cache files are invalidated and re-run |
| Protocol lock format | 1 | `runtime/protocol_lock.py` | — |
| Gate table schema | 1 | `estimation/gate_table.py` | Rows gained `desk_exposable`; table gained `approval`, `provisional_allow_list` |
| Protocol bundle | **none supplied** | external (A01/A05) | Protocol v1.2 lock to be pinned by hash |
| Canonical apgi-api schemas | **not pinned** | external (A06) | Schema hashes to be recorded in the release manifest |
| apgi-framework reference | **not pinned** | external (A03) | Commit to be recorded in run manifests (`framework_commit`) |

## Tested combinations

| apgi | Python | numpy | Protocol bundle | apgi-api schemas | Evidence |
|---|---|---|---|---|---|
| 1.1.0 (+ this change) | 3.10, 3.11, 3.12, 3.13 | 1.26.x (<2) | none (exploratory only) | none | CI `test` matrix, `package` (installed-wheel qualification) |
| 1.1.0 `estimation` extra | 3.12 | ≥2 (separate env) | none | none | CI `estimation-pymc` |

## Untested / blocked

* Any Protocol v1.2 bundle — blocked on AUD-PRO-001 / AUD-REP-001 (approved lock).
* macOS ARM64 — no CI evidence (AUD-REP-002).
* Paper/archive revisions (Zenodo 10.5281/zenodo.21632264, OSF t5hcq) — not
  linked to a certified release set (AUD-REL-005).
