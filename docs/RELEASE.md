# Release runbook (draft)

**Status: DRAFT (AUD-REL-002, AUD-REL-003).** This is the proposed release
procedure for `apgi`. No release has yet been certified under it; a tabletop
rollback has not been recorded. Approval by the release owner is required.

## 1. Preconditions (blocking)

1. Every CI job on the candidate commit is green: `test` (3.10–3.13),
   `lint & types`, `security`, `package`, `reproducibility`, the XC-CI-02 gates,
   `dependency-policy`, `estimation-pymc`.
2. No open item in `docs/AUDIT-STATUS.md` that is non-waivable for this
   component is `Fail` (Specification §9: SCI-001/002, PRO-001, REP-001,
   API-001/004, VAL-001, DAT-001/002/004, API-003, IMP-002, INT-002, REL-001).
   A mandatory test that cannot run is recorded as **BLOCKED**, never PASS.
3. The change-impact record (§4) exists for every material change since the
   previous release.

Until items 2–3 can be met the project can publish *development* builds but not
a certified release.

## 2. Build and record

1. Tag the candidate commit (`vX.Y.Z`); the working tree must be clean.
2. Retain the CI artefacts of that commit: `release-security-evidence`
   (`release-lock.txt`, `release-artifacts.sha256`, `pip-audit.json`,
   `sbom.cdx.json`, `secrets-scan.json`) and `coverage-xml`.
3. Record in the release manifest: wheel SHA-256, commit, release lock digest,
   protocol lock hash (once supplied), canonical schema hashes (once pinned),
   `apgi_reduced` version and qualification record digest.

## 3. Deploy (distribution)

`apgi` is distributed as a wheel. Deployment = publishing the exact recorded
wheel to the index / archive. Never rebuild after certification.

## 4. Change-impact assessment (REL-003)

For each material change record:

| Field | Content |
|---|---|
| Change | commit(s) / PR |
| Category | scientific model · numerical implementation · protocol · API/schema · dependency · security · docs only |
| Affected requirements and claims | ids from `docs/TRACEABILITY.md` / `docs/CLAIM-REGISTER.md` |
| Risk | low / medium / high, with reason |
| Regression selection | which test sets were run and why (default: full suite) |
| Result and disposition | link to CI run; defects opened |

A change in the scientific model, protocol bundle, numerical implementation, API
contract or security-relevant processing invalidates the affected certificate
scopes immediately (Specification §10).

## 5. Rollback

**Criteria** — any of: a reference-output regression (qualification fails on
the installed wheel), a security finding of high severity in the release set,
a defect that produces a false scientific success, or a broken install.

**Procedure**

1. Yank the affected version on the package index (do not delete) and mark the
   archive record as superseded.
2. Re-point documentation and citation metadata to the last certified version.
3. Consumers pin `apgi==<last good>`; checkpoints and batch caches written by
   the bad version are refused automatically if the model or package version
   differs (AUD-DEF-05, AUD-DEF-06).
4. Open an incident record and a change-impact assessment for the fix.
5. Record the rollback (who, when, version) in `CHANGELOG.md`.

## 6. Known limitations to state with every release

* No Protocol v1.2 bundle is verified; all runs are exploratory.
* `apgi_reduced` qualification is self-qualification against internal vectors;
  independently approved vectors (S1-REG-04) are pending.
* K7b is pending; no individual parameter is approved for production exposure.
* Linux x86-64 only has CI evidence.

## 7. Monitoring hand-off

The library has no runtime service. Post-release monitoring = watching the
issue tracker and re-running the `security` job (manually via
`workflow_dispatch`; no schedule is configured yet) against the released lock
to catch newly published vulnerabilities.
