# Scientific claim register (draft)

**Status: DRAFT — no entry is approved (AUD-SCI-007).** APGI-SCI-007 requires
every user-visible scientific statement to map to a controlled claim with an
epistemic tier and status. This register lists the statements this repository
makes in `README.md`, `docs/ENGINE.md` and code-level labels, and proposes a
tier for each. The project-wide `CLM-nnn` numbering is owned by the scientific
authority; the `A04-CLM-nnn` ids below are local placeholders to be mapped onto
it.

Tiers: **established** (accepted science, cited) · **derived** (follows
mathematically from the model equations) · **calibrated** (fitted to data) ·
**speculative** (hypothesis, not yet tested). Status: **unreviewed** until the
scientific authority approves.

| Local id | Statement (where) | Proposed tier | Evidence in this repo | Status |
|---|---|---|---|---|
| A04-CLM-001 | The learning gain depends on α and Π only through αΠ, so behaviour alone cannot separate them (ENGINE.md; `models/reduced.py`) | derived | `k = αΠ/(1+αΠ)`; `qualification` product-symmetry check | unreviewed |
| A04-CLM-002 | `precision_weighted_learning_composite` is identifiable and recovered, r ≈ 0.997 (ENGINE.md) | derived (desk simulation) | Development K7 run, **not retained** | unreviewed — provisional |
| A04-CLM-003 | With ignition + amplitude channels the Jacobian is full rank (median condition ≈ 28) but α, Π remain near-collinear (≈ 0.98) (ENGINE.md) | derived (desk simulation) | Development K7 run, not retained | unreviewed — provisional |
| A04-CLM-004 | θ₀ is the only individual parameter clearing every desk gate under the full design (ENGINE.md) | derived (desk simulation) | Development K7 run, not retained; K7b pending | unreviewed — provisional |
| A04-CLM-005 | `apgi_reduced` is the single qualified numerical reference (README, `model_registry`) | governance decision (D14) | `qualification/apgi_reduced.qualification.json` (self-qualification only) | unreviewed — S1-DEC-06 pending |
| A04-CLM-006 | Precision system is "Free Energy Principle-grounded" (README feature list) | speculative (theoretical framing) | None in this repo | unreviewed |
| A04-CLM-007 | Landauer's principle gives a lower bound on per-step energy cost (README; `core/thermodynamics.py`) | established (physics) / speculative (application to ignition) | Unit-tested arithmetic only | unreviewed |
| A04-CLM-008 | BOLD-calibrated energy conversion, 1.2e-18 J per 1 % BOLD per cm³ and a 5–10 % ignition spike (`config.py`, `energy/`) | calibrated? — source not recorded | None; default provenance `assumed` | unreviewed |
| A04-CLM-009 | Threshold dynamics show 1/f (pink-noise) structure and avalanches near exponent 1.5 (`pipeline.validate`) | speculative (prediction) | Validation routines exist; no empirical result | unreviewed |
| A04-CLM-010 | Five-level temporal hierarchy L0–L4 with the APGI-SCI-003 spans (`hierarchy/levels.py`) | established within APGI theory (specification) | Label/band checks only; demo config does not realise the spans | unreviewed |
| A04-CLM-011 | "Observable mapping to neural/behavioral data" (README feature list) | speculative | Mapping code; no empirical validation retained | unreviewed |
| A04-CLM-012 | Seeded runs are bit-for-bit reproducible (README) | derived (software property) | `tests/test_reproducibility.py`, CI `reproducibility` (Linux only) | unreviewed — cross-platform evidence absent |

## Rules for new statements

1. A new scientific statement in docs, docstrings shown to users, or outputs
   gets a row here before merge.
2. Numbers quoted in docs must cite a retained artefact (path + SHA-256) or be
   labelled *provisional, not retained*.
3. Exploratory/extension model results never carry reference authority
   (`APGIModel.authority`).
