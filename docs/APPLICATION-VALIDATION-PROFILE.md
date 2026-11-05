# Application Validation Profile — apgi-implementation (A04)

| Field | Value |
|---|---|
| Status | **DRAFT — not approved.** Proposed by the implementation team; every decision below needs a named approver before it counts as a profile decision. |
| Component | A04 `apgi-implementation` — reference numerical model (Software Validation Specification v1.2, §5.1) |
| Governing document | APGI 1.2 Software Validation Specification (itself *Draft for approval*, AUD-GOV-01) |
| Package version | `apgi` 1.1.0 (`pyproject.toml`) |
| Release owner | **TBD** — to be named (AUD-GOV-04) |
| Scientific authority | **TBD** — owns S1-REG-01, S1-REG-04, K7 approvals |
| Closes | Drafts AUD-PRO-004 / AUD-GOV-02; proposes scope for AUD-API-002, AUD-API-004, AUD-INT-004, AUD-CON-001, AUD-CON-002, AUD-ACQ-001, AUD-CLI-001, AUD-DAT-003, AUD-PRO-003 |

## 1. What this component is

A Python library (with a numerical CLI, `apgi`) that implements the APGI
equations, the qualified reference model `apgi_reduced`, six exploratory/extension
variants, parameter estimation (K7a/K7c), a session/batch runtime and run
manifests. It is distributed as a wheel. It does **not** run a network service,
host a participant-facing client, perform acquisition, or store participant
data on behalf of anyone.

## 2. Supported scope (proposed)

| Item | Supported | Notes |
|---|---|---|
| Platforms | Linux x86-64, Python 3.10–3.13 | CI matrix. macOS ARM64 is **not** yet evidenced (AUD-REP-002). |
| Interfaces | Python library API (`docs/API-REFERENCE.md`); numerical CLI `apgi` | No HTTP API. |
| Exchanged payloads | `CanonicalOutputs.exchange_payload()` (schema v2, carries schema/protocol/producer versions) | Canonical apgi-api schemas not yet pinned (AUD-INT-001). |
| Protocol definitions | Loaded **only** through `runtime.protocol_lock.verify_protocol_bundle` | The approved Protocol v1.2 bundle (protocol_0 … protocol_7) and its lock are not in this repository (AUD-PRO-001). Until they are supplied, the supported protocol set is **empty** and every run is labelled exploratory. `adapters.version_adapter.SPEC_VERSION = 1.1.0` is a config-schema version, not a Protocol v1.2 support declaration. |
| Reference model | `apgi_reduced` 1.0.0 (sole numerical authority, D14 — decision closure pending S1-DEC-06) | Other variants are exploratory/extension/comparison/benchmark. |
| Data classes | (a) synthetic/simulated series; (b) de-identified research recordings supplied by the caller (EEG `.npy/.npz` or MNE formats, behavioural CSV/JSON/HDF5) | See §4. |

## 3. Requirement applicability (Section 5.2 A04 column, with proposed rationale)

| Requirement | A04 matrix | Proposed decision | Rationale / boundary |
|---|---|---|---|
| PRO-001…004 | R | Applicable | PRO-003 applies to session resume and batch cache: ordering = strict input-prefix equality on resume; idempotency = verified cache identity (AUD-DEF-06). No message correlation/retry semantics exist because there is no messaging interface. |
| SCI-001…007 | R | Applicable | SCI-001/002 blocked on S1-REG-04. |
| REP-001…005 | R | Applicable | REP-005 limited to declared energy fields (`C_landauer` J, `bits_erased` bit). |
| IMP-001…003 | R | Applicable | |
| DAT-001, 002, 004 | R | Applicable | Quarantine via `adapters/quarantine.py`. |
| DAT-003 | R | Applicable, **bounded** | The library persists only caller-directed artefacts (checkpoints, batch cache, manifests) to caller-chosen paths. Proposed: retention/deletion of those paths is the caller's responsibility; the library does not maintain stores. Needs approval. |
| API-001 | C | Applicable to the Python API | Contract = `docs/API-REFERENCE.md` + canonical output schema v2. |
| API-002 | – | **Proposed NOT_APPLICABLE** | No protected operations or authenticated service. Reassess if the library is embedded in a service processing protected data. |
| API-003 | R | Applicable | `np.load(allow_pickle=False)`; content secret scan and release dependency audit in CI. |
| API-004 | – | **Proposed NOT_APPLICABLE for hosted error semantics** | False-success behaviour is nevertheless fixed (AUD-DEF-03); library errors are typed (`NonFiniteInputError`, `IncompatibleCheckpointError`, `ProtocolLockError`, `DatasetLoadError`, …). |
| INT-001…003 | R | Applicable | INT-002 blocked on the platform end-to-end scenario. |
| INT-004 | – | **Proposed NOT_APPLICABLE** | Explorer/Appendix/website/quiz are other components (A07–A10); display parity belongs to them. |
| REL-001…005 | R | Applicable | |
| CON-001, CON-002 | – | **Proposed NOT_APPLICABLE with input boundary** | The library ingests recordings supplied by a caller but neither collects nor stores participant data. Proposed boundary: callers may only pass data whose consent scope already covers this processing (consent-cleared or synthetic); consent enforcement and withdrawal orchestration are owned by apgi-api (A06). Needs approval; reassess if the library ever stores participant data. |
| ACQ-001 | – | **Proposed NOT_APPLICABLE** | No acquisition hardware or timing. Ingested files are checksummed (`provenance.sha256`) and handed off with their provenance. |
| VAL-001 | R | Applicable | K7 gate semantics: desk vs. approved exposure (AUD-DEF-11). |
| CLI-001 | – | **Proposed NOT_APPLICABLE** | "Client" means participant instruments; `apgi` is a numerical CLI. |

## 4. Data handling (proposed)

* **Malformed input**: rejected before any state change (`core/finite.py`); record-level
  inputs are screened with `adapters.quarantine.screen_records`, whose outcomes
  (`malformed`, `duplicate`, `conflict`, `delayed`, `unmatched`, `missing`) must be
  reported with every analysis that uses screened data.
* **Untrusted files**: pickled NumPy objects are refused; unknown formats raise.
* **Provenance**: a run may be called reproducible only with a dataset record carrying
  source, version, checksum, license and processing history (`runtime/run_manifest.py`).

## 5. Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Release owner | | | |
| Scientific authority | | | |
| Data-protection reviewer (CON/DAT-003 boundary) | | | |
