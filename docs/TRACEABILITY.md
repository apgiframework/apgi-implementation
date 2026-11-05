# Requirement → code → test traceability (draft)

**Status: DRAFT working matrix (AUD-GOV-03).** It links each APGI 1.2 requirement
that applies to this component (see `docs/APPLICATION-VALIDATION-PROFILE.md`) to
the code that implements it and the tests that exercise it. It is not yet the
release-linked, approved validation traceability matrix the specification
requires: it has no run/evidence/disposition columns, because no certified
release candidate exists yet. Claims are tracked in `docs/CLAIM-REGISTER.md`;
per-item audit status is in `docs/AUDIT-STATUS.md`.

| Requirement | Implementation | Tests | Status |
|---|---|---|---|
| PRO-001 protocol identity | `runtime/protocol_lock.py`, `adapters/version_adapter.py` | `tests/test_audit_defects.py::TestRep001ProtocolLock`, `tests/test_protocol_compatibility.py` | Blocked: approved Protocol v1.2 bundle/map absent |
| PRO-002 reject malformed input atomically | `core/finite.py`; `pipeline.py` `step`; `models/base.py` `run`; `models/reduced.py`; `models/pipeline_models.py`; `runtime/session_engine.py` `advance`/`resume`; `estimation/likelihood.py` `Observed` | `TestDef01NonFiniteRejectedAtomically`, `TestDef05CheckpointCompatibility` | Implemented |
| PRO-003 ordering/idempotency | `runtime/session_engine.py` prefix check; `runtime/batch_runner.py` cache identity | `tests/test_runtime.py`, `TestDef06BatchCacheIntegrity` | Implemented; scope proposed |
| PRO-004 validation profile | `docs/APPLICATION-VALIDATION-PROFILE.md` | — | Draft |
| SCI-001 ignition inequality | `core/ignition.py`, `models/reduced.py` | `tests/test_core_ignition.py`, `tests/test_reference_vectors.py` | Blocked: formulation reconciliation (S1-REG-04) |
| SCI-002 formulation identity | `qualification/reference.py`, `models/reduced.py` | `tests/test_reference_vectors.py` | Blocked: independent approved vectors |
| SCI-003 L0–L4 hierarchy | `hierarchy/levels.py`, `docs/HIERARCHICAL-GUIDE.md` | `TestSci003Levels` | Implemented; ranges need confirmation |
| SCI-004 notation | `model_registry/parameter_mapping.py`, `hierarchy/levels.py` | `tests/test_models.py` | Partial: approved Notation Appendix absent |
| SCI-005 default provenance | `model_registry/default_provenance.py` | `TestSci005DefaultProvenance` | Partial: 80 defaults unclassified |
| SCI-006 acronym | `README.md`, `CITATION.cff`, `pyproject.toml` | — | Pass (register) |
| SCI-007 claims | `docs/CLAIM-REGISTER.md` | — | Draft |
| REP-001 protocol lock | `runtime/protocol_lock.py`; `runtime/run_manifest.py` | `TestRep001ProtocolLock`, `TestDef02ManifestIdentity` | Mechanism implemented; approved lock absent |
| REP-002 seeded reproducibility | `core/rng.py`, `engine/provenance.py`, `runtime/` | `tests/test_reproducibility.py`, `tests/test_ci_gates.py` | Partial: cross-platform evidence absent |
| REP-003 statistics | `estimation/composite.py`, `estimation/gate_table.py` | `TestDef04CompositeStandardError` | Partial: full inventory open |
| REP-004 registered plan / exploratory labels | `runtime/run_manifest.py` (confirmatory gate), `core/manifest.py` (`analysis_label`) | `TestDef02ManifestIdentity::test_confirmatory_requires_verified_lock` | Implemented for manifests |
| REP-005 energy units | `adapters/canonical_outputs.py` `OUTPUT_SCHEMA` | `tests/test_pipeline_extended_coverage.py::test_pipeline_bold_calibration` | Partial |
| IMP-001 build provenance | `pyproject.toml` package-data; CI `security` (lock, SBOM, wheel digest), `package` (installed qualification) | `test_def07_qualification_resources_are_package_data` | Partial |
| IMP-002 security configuration | CI `security` | — | Partial |
| DAT-001 processing validation | `core/finite.py`, `adapters/canonical_inputs.py` | `TestDef01…` | Partial |
| DAT-002 quarantine/reconciliation | `adapters/quarantine.py`, `runtime/batch_runner.py` | `TestDat002Quarantine`, `TestDef06…` | Implemented (mechanism) |
| DAT-004 dataset provenance | `runtime/run_manifest.py`, `validation/empirical_validation.py` | `TestDef02ManifestIdentity::test_dataset_provenance_required`, `TestDef12EmpiricalLoader` | Partial: not on every entry point |
| API-003 secrets/untrusted data | `validation/empirical_validation.py` (`allow_pickle=False`); CI secret scan | `TestDef12EmpiricalLoader` | Partial |
| INT-001 versioned payloads | `adapters/canonical_outputs.py` `exchange_payload` | `test_int001_exchange_payload_requires_all_versions` | Partial: apgi-api schemas not pinned |
| INT-003 failure propagation | `pipeline.py` `validate`; `validation/empirical_validation.py`; `runtime/batch_runner.py` | `TestDef03FailedSubAnalysis`, `TestDef12…`, `TestDef06…` | Implemented for known paths |
| VAL-001 verdicts | `estimation/gate_table.py`, `estimation/composite.py` | `TestK7CompositeConvergence`, `TestDef11GateApproval`, `tests/test_estimation.py` | Partial: locked rule bundle absent |
| REL-001…005 | `.github/workflows/ci.yml`, `docs/RELEASE.md`, `docs/COMPATIBILITY-MATRIX.md` | CI jobs | Blocked: certification process |
