"""Regression tests for the audit defects AUD-DEF-01 … AUD-DEF-12 and AUD-K7-CONVERGENCE.

Each test reproduces the probe recorded in the audit register (the input that
exposed the defect) and asserts the corrected behaviour. They are named after
the register ids so the requirement-to-test trace (docs/TRACEABILITY.md) can
point at them directly.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from adapters.canonical_outputs import IncompletePayloadError, to_canonical_outputs
from adapters.quarantine import ReconciliationError, ScreeningReport, screen_records
from config import CONFIG
from core.finite import NonFiniteInputError, require_finite_array, require_finite_scalar
from engine.provenance import record_run, replay_run
from estimation import gate_table as gt
from estimation.composite import COMPOSITE_NAME, fit_composite
from estimation.likelihood import Observed
from estimation.recovery_simulation import RecoveryResult
from hierarchy import levels
from model_registry import create_model
from model_registry.default_provenance import (
    BASES,
    UNCLASSIFIED,
    all_defaults,
    provenance_for,
)
from models.reduced import ReducedParams, learning_gain
from pipeline import APGIPipeline
from runtime.batch_runner import make_jobs, run_batch
from runtime.checkpoint import save_checkpoint
from runtime.protocol_lock import ProtocolLockError, build_lock, lock_digest, verify_protocol_bundle
from runtime.run_manifest import build_run_manifest, verify_manifest
from runtime.session_engine import IncompatibleCheckpointError, SessionEngine
from validation.empirical_validation import (
    DatasetConfig,
    DatasetLoadError,
    EmpiricalDataLoader,
    OptionalDependencyError,
    UnsupportedFormatError,
)

BAD = [math.nan, math.inf, -math.inf]


# --------------------------------------------------------------------------- DEF-01
class TestDef01NonFiniteRejectedAtomically:
    @pytest.mark.parametrize("bad", BAD)
    def test_pipeline_step_rejects_without_mutation(self, bad):
        p = APGIPipeline({**CONFIG, "seed": 3})
        p.step(x_e=0.4, x_i=0.1)
        before = (
            copy.deepcopy(p.state),
            p.S,
            p.theta,
            p.rng.bit_generator.state,
            len(p.history["S"]),
        )
        for kwargs in (
            {"x_e": bad, "x_i": 0.0},
            {"x_e": 0.0, "x_i": bad},
            {"x_e": 0.0, "x_i": 0.0, "x_hat_e": bad},
        ):
            with pytest.raises(NonFiniteInputError):
                p.step(**kwargs)
        after = (p.state, p.S, p.theta, p.rng.bit_generator.state, len(p.history["S"]))
        assert after[0] == before[0] and after[1:] == before[1:]

    @pytest.mark.parametrize("model_id", ["apgi_reduced", "apgi_null", "apgi_reference"])
    def test_session_advance_rejects_without_mutation(self, model_id):
        eng = SessionEngine(model_id, seed=1)
        eng.advance(0.3)
        state, digest = eng.model.get_state(), eng.output_digest()
        with pytest.raises(NonFiniteInputError):
            eng.advance(math.nan)
        assert eng.n_done == 1 and eng.output_digest() == digest
        assert eng.model.get_state() == state

    def test_model_run_validates_whole_sequence_before_stepping(self):
        m = create_model("apgi_reduced", {}, seed=0)
        x = np.zeros(10)
        x[7] = math.nan
        with pytest.raises(NonFiniteInputError, match="index 7"):
            m.run(x)
        # nothing advanced: a fresh model on the clean prefix gives the same first step
        fresh = create_model("apgi_reduced", {}, seed=0)
        assert m.step(0.0) == fresh.step(0.0)

    @pytest.mark.parametrize(
        "field,value",
        [("sigma_r", math.nan), ("sigma_y", math.inf), ("sigma_r", 0.0), ("sigma_y", -1.0)],
    )
    def test_reduced_params_reject_bad_noise(self, field, value):
        with pytest.raises(ValueError):
            ReducedParams(**{field: value})

    def test_fitting_boundary_rejects_nonfinite(self):
        with pytest.raises(ValueError):
            Observed(stimulus=[0.0, math.nan], B=[0, 1])
        with pytest.raises(ValueError):
            Observed(stimulus=[0.0, 1.0], B=[0, 1], report=[0.1, math.inf])
        with pytest.raises(ValueError, match="binary"):
            Observed(stimulus=[0.0, 1.0], B=[0, 2])
        with pytest.raises(ValueError):
            fit_composite([0.0, math.nan, 1.0], [0.0, 0.1, 0.2])

    def test_helpers(self):
        assert require_finite_scalar("x", np.float32(1.5)) == 1.5
        assert require_finite_scalar("x", np.array(2.0)) == 2.0
        with pytest.raises(NonFiniteInputError):
            require_finite_scalar("x", True)
        with pytest.raises(NonFiniteInputError):
            require_finite_scalar("x", "1.0")
        with pytest.raises(NonFiniteInputError, match="2-D"):
            require_finite_array("a", np.zeros(3), ndim=2)
        with pytest.raises(NonFiniteInputError, match="length 4"):
            require_finite_array("a", np.zeros(3), length=4)
        with pytest.raises(NonFiniteInputError, match="empty"):
            require_finite_array("a", [], allow_empty=False)
        with pytest.raises(NonFiniteInputError, match="numeric"):
            require_finite_array("a", ["x"])


# --------------------------------------------------------------------------- DEF-02 / REP-001
def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    d = tmp_path / "bundle"
    d.mkdir(parents=True)
    for i in range(8):
        (d / f"protocol_{i}.json").write_text(json.dumps({"id": f"protocol_{i}", "v": "1.2"}))
    lock = build_lock(d, protocol_version="1.2", bundle="test-bundle", timestamp_proof="ots:test")
    lp = tmp_path / "protocol.lock.json"
    lp.write_text(json.dumps(lock))
    return d, lp


def _gate_table(tmp_path: Path) -> Path:
    path = tmp_path / "gates" / "gate-table.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": "1", "parameters": [{"parameter": "alpha"}]}))
    return path


class TestRep001ProtocolLock:
    def test_valid_bundle_verifies(self, tmp_path):
        d, lp = _bundle(tmp_path)
        v = verify_protocol_bundle(lp, d, expected_protocol_version="1.2")
        assert len(v.definitions) == 8 and v.lock_hash == lock_digest(json.loads(lp.read_text()))

    def test_tampered_definition_refused(self, tmp_path):
        d, lp = _bundle(tmp_path)
        (d / "protocol_3.json").write_text(json.dumps({"id": "protocol_3", "v": "tampered"}))
        with pytest.raises(ProtocolLockError, match="protocol_3.json"):
            verify_protocol_bundle(lp, d)

    def test_missing_unlisted_and_substituted_lock_refused(self, tmp_path):
        d, lp = _bundle(tmp_path)
        pinned = lock_digest(json.loads(lp.read_text()))
        (d / "protocol_8.json").write_text("{}")
        with pytest.raises(ProtocolLockError, match="not covered"):
            verify_protocol_bundle(lp, d)
        (d / "protocol_8.json").unlink()
        (d / "protocol_0.json").unlink()
        with pytest.raises(ProtocolLockError, match="missing"):
            verify_protocol_bundle(lp, d)
        with pytest.raises(ProtocolLockError, match="pinned"):
            verify_protocol_bundle(lp, d, expected_lock_hash="0" * 64)
        assert pinned

    def test_bad_lock_documents(self, tmp_path):
        d, lp = _bundle(tmp_path)
        lp.write_text("not json")
        with pytest.raises(ProtocolLockError, match="unreadable"):
            verify_protocol_bundle(lp, d)
        lp.write_text(
            json.dumps(
                {
                    "lock_version": "1",
                    "files": {"protocol_0.json": "xyz"},
                    "protocol_version": "1.2",
                    "bundle": "b",
                }
            )
        )
        with pytest.raises(ProtocolLockError, match="SHA-256"):
            verify_protocol_bundle(lp, d)
        with pytest.raises(ProtocolLockError, match="expected"):
            lp.write_text(json.dumps(build_lock(d, protocol_version="1.1", bundle="b")))
            verify_protocol_bundle(lp, d, expected_protocol_version="1.2")


def _commit() -> str:
    from core.manifest import git_revision

    return git_revision()["commit"] or "0" * 40


def _full_manifest(tmp_path, **over):
    d, lp = _bundle(tmp_path)
    lock = verify_protocol_bundle(lp, d)
    kw = {
        "study": "S1",
        "protocol_id": "P-001",
        "protocol_version": "1.2",
        "protocol_hash": lock.lock_hash,
        "model_id": "apgi_reduced",
        "model_version": "1.0.0",
        "seed": 1,
        "data_origin": "synthetic",
        "mode": "desk",
        "dataset_hash": "cd" * 32,
        "framework_commit": "a" * 40,
        "spec_commit": "b" * 40,
        "gate_table_path": _gate_table(tmp_path),
        "protocol_lock": lock,
        "dataset": {
            "source": "synthetic generator",
            "version": "1",
            "checksum": "cd" * 32,
            "license": "CC0-1.0",
            "history": ["generated"],
        },
    }
    kw.update(over)
    return build_run_manifest(**kw), d


class TestDef02ManifestIdentity:
    def test_audit_probe_is_not_reproducible(self, tmp_path):
        """Invalid strings, unknown model, null commits and a non-gate file (the probe)."""
        junk = tmp_path / "gate.yaml"
        junk.write_text("not a gate table")
        m = build_run_manifest(
            study="S",
            protocol_id="p",
            protocol_version="1",
            protocol_hash="not-a-hash",
            model_id="no_such_model",
            model_version="9",
            seed=1,
            data_origin="synthetic",
            mode="desk",
            dataset_hash="also-bad",
            gate_table_path=junk,
        )
        assert m["reproducible"] is False
        text = " | ".join(m["reproducibility_blockers"])
        for needle in (
            "protocol hash is not",
            "dataset hash is not",
            "not registered",
            "framework commit",
            "spec commit",
            "does not parse as a gate table",
            "protocol lock not verified",
            "dataset provenance record",
        ):
            assert needle in text, needle

    def test_fully_verified_manifest_and_later_tamper(self, tmp_path):
        m, bundle = _full_manifest(tmp_path)
        blockers = [
            b
            for b in m["reproducibility_blockers"]
            if "uncommitted" not in b and "implementation commit" not in b
        ]
        assert blockers == []
        # A development checkout may be dirty; pin the implementation commit so
        # the verification path is exercised in every environment.
        m = {**m, "reproducible": True, "reproducibility_blockers": []}
        m["commits"] = {**m["commits"], "implementation": _commit()}
        assert verify_manifest(m) == []

    def test_verify_detects_tamper_after_build(self, tmp_path):
        m, bundle = _full_manifest(tmp_path)
        m = {**m, "reproducible": True, "reproducibility_blockers": []}
        m["commits"] = {**m["commits"], "implementation": _commit()}
        (bundle / "protocol_5.json").write_text("{}")
        problems = verify_manifest(m)
        assert any("re-verification failed" in p for p in problems)

    def test_version_mismatches_block(self, tmp_path):
        m, _ = _full_manifest(tmp_path, model_version="2.0.0", protocol_version="1.1")
        text = " ".join(m["reproducibility_blockers"])
        assert "differs from registered" in text and "protocol version does not match" in text
        m2, _ = _full_manifest(tmp_path / "x", protocol_hash="ef" * 32)
        assert any("lock hash" in b for b in m2["reproducibility_blockers"])

    def test_dataset_provenance_required(self, tmp_path):
        m, _ = _full_manifest(tmp_path, dataset={"source": "s", "checksum": "ef" * 32})
        text = " ".join(m["reproducibility_blockers"])
        for f in ("version", "license", "history"):
            assert f"lacks {f!r}" in text
        assert "does not equal dataset_hash" in text

    def test_confirmatory_requires_verified_lock(self, tmp_path):
        with pytest.raises(ValueError, match="confirmatory"):
            _full_manifest(tmp_path, mode="confirmatory", protocol_lock=None)


# --------------------------------------------------------------------------- DEF-03
class TestDef03FailedSubAnalysis:
    def test_injected_avalanche_failure_is_partial_not_success(self, monkeypatch):
        import stats.avalanche as av

        def boom(*a, **k):
            raise RuntimeError("injected")

        monkeypatch.setattr(av, "validate_avalanche_power_law", boom)
        p = APGIPipeline({**CONFIG, "seed": 1})
        for t in range(200):
            p.step(x_e=float(np.sin(t / 5.0)), x_i=0.1)
        out = p.validate()
        assert out["status"] == "partial"
        assert out["certifiable"] is False
        assert out["failed_analyses"] == ["avalanche_power_law"]

    def test_clean_run_is_success(self):
        p = APGIPipeline({**CONFIG, "seed": 1})
        for t in range(200):
            p.step(x_e=float(np.sin(t / 5.0)), x_i=0.1)
        out = p.validate()
        assert out["status"] == "success" and out["certifiable"] and out["failed_analyses"] == []


# --------------------------------------------------------------------------- DEF-04
class TestDef04CompositeStandardError:
    def _data(self, seed, k_true=0.25, T=200, sigma=0.15):
        rng = np.random.default_rng(seed)
        x = rng.choice([-1.0, 1.0], size=T) + 0.1 * rng.standard_normal(T)
        xhat, path = 0.0, np.empty(T)
        for t in range(T):
            path[t] = xhat
            xhat += k_true * (x[t] - xhat)
        return x, path + sigma * rng.standard_normal(T)

    def test_se_matches_independent_observed_information(self):
        """Independent check: SE(k) = 1/sqrt(d²NLL/dk²) with NLL = SSE/(2σ²)."""
        from scipy import optimize

        x, r = self._data(0)
        est = fit_composite(x, r, sigma_r=0.15)

        def nll(k):
            xhat, s = 0.0, 0.0
            for t in range(x.size):
                s += (r[t] - xhat) ** 2
                xhat += k * (x[t] - xhat)
            return s / (2 * 0.15**2)

        k_hat = optimize.minimize_scalar(
            nll, bounds=(1e-6, 1 - 1e-6), method="bounded", options={"xatol": 1e-10}
        ).x
        h = 1e-4
        info = (nll(k_hat + h) - 2 * nll(k_hat) + nll(k_hat - h)) / h**2
        se_c = (1 / math.sqrt(info)) / (1 - k_hat) ** 2
        half = (est.ci95[1] - est.ci95[0]) / 2
        assert half == pytest.approx(1.959963984540054 * se_c, rel=1e-3)

    def test_interval_coverage_is_nominal(self):
        k_true = 0.25
        c_true = k_true / (1 - k_true)
        hits = sum(
            (lambda e: e.ci95[0] <= c_true <= e.ci95[1])(fit_composite(*self._data(s), 0.15))
            for s in range(200)
        )
        # 95% nominal; binomial sd at n=200 is ~1.5 points. The pre-fix SE gave ~83%.
        assert 0.90 <= hits / 200 <= 0.99

    def test_boundary_estimate_is_flagged(self):
        x = np.ones(50)
        est = fit_composite(x, np.zeros(50), sigma_r=0.15)
        assert est.method == "behavior-ml-boundary" and est.ci95[0] == 0.0
        assert learning_gain(1.0, 1.0) == 0.5


# --------------------------------------------------------------------------- DEF-05
class TestDef05CheckpointCompatibility:
    def test_audit_probe_model_version_999_refused(self):
        eng = SessionEngine("apgi_reduced", seed=1)
        eng.run(np.linspace(0, 1, 2))
        ck = eng.checkpoint()
        ck.model_version = "999.0.0"
        with pytest.raises(IncompatibleCheckpointError, match="999.0.0"):
            SessionEngine.resume(ck)

    def test_inconsistent_history_refused(self, tmp_path):
        eng = SessionEngine("apgi_reduced", seed=1)
        eng.run(np.linspace(0, 1, 4))
        ck = eng.checkpoint()
        ck.history["S"][0] += 1.0
        with pytest.raises(IncompatibleCheckpointError, match="digest"):
            SessionEngine.resume(ck)
        ck2 = eng.checkpoint()
        ck2.x_e = ck2.x_e[:-1]
        with pytest.raises(IncompatibleCheckpointError, match="n_done"):
            SessionEngine.resume(ck2)

    def test_producer_mismatch_refused_and_roundtrip_ok(self, tmp_path):
        eng = SessionEngine("apgi_reduced", seed=1)
        eng.run(np.linspace(0, 1, 4))
        path = eng.save(tmp_path / "ck.json")
        assert SessionEngine.resume(path).output_digest() == eng.output_digest()
        ck = eng.checkpoint()
        ck.extra["producer"] = {**ck.extra["producer"], "checkpoint_format": "0"}
        save_checkpoint(ck, tmp_path / "ck2.json")
        with pytest.raises(IncompatibleCheckpointError, match="checkpoint_format"):
            SessionEngine.resume(tmp_path / "ck2.json")

    def test_replay_refuses_other_model_version(self):
        rec, _ = record_run("apgi_reduced", {}, 1, np.linspace(0, 1, 5))
        rec.model_version = "999.0.0"
        with pytest.raises(ValueError, match="999.0.0"):
            replay_run(rec, np.linspace(0, 1, 5))


# --------------------------------------------------------------------------- DEF-06
class TestDef06BatchCacheIntegrity:
    def _jobs(self):
        return make_jobs("apgi_reduced", np.linspace(-1, 1, 30), n_seeds=2, master_seed=4)

    def test_audit_probe_altered_mean_s_is_rejected(self, tmp_path):
        jobs = self._jobs()
        first = run_batch(jobs, output_dir=tmp_path)
        f = tmp_path / f"{jobs[0].job_id}.json"
        doc = json.loads(f.read_text())
        doc["body"]["summary"]["mean_S"] = 123456.0
        f.write_text(json.dumps(doc))
        again = run_batch(jobs, output_dir=tmp_path)
        r0 = again.by_id()[jobs[0].job_id]
        assert r0.from_cache is False and r0.mean_S == first.by_id()[jobs[0].job_id].mean_S
        assert "digest mismatch" in again.invalidated[jobs[0].job_id]
        assert again.by_id()[jobs[1].job_id].from_cache is True

    def test_consistent_but_forged_summary_rejected(self, tmp_path):
        """Even with a recomputed document digest, summary must match stored outputs."""
        from runtime.batch_runner import _doc_digest

        jobs = self._jobs()
        run_batch(jobs, output_dir=tmp_path)
        f = tmp_path / f"{jobs[0].job_id}.json"
        doc = json.loads(f.read_text())
        doc["body"]["summary"]["mean_S"] = 123456.0
        doc["sha256"] = _doc_digest(doc["body"])
        f.write_text(json.dumps(doc))
        again = run_batch(jobs, output_dir=tmp_path)
        assert "does not match its stored outputs" in again.invalidated[jobs[0].job_id]

    def test_stale_identity_and_legacy_files_invalidated(self, tmp_path, monkeypatch):
        import runtime.batch_runner as br

        jobs = self._jobs()
        run_batch(jobs, output_dir=tmp_path)
        monkeypatch.setattr(br, "_producer", lambda: {"package_version": "other"})
        again = run_batch(jobs, output_dir=tmp_path)
        assert all("stale cache" in why for why in again.invalidated.values())
        (tmp_path / f"{jobs[0].job_id}.json").write_text(json.dumps({"spec_hash": "x"}))
        assert "unreadable" in run_batch(jobs, output_dir=tmp_path).invalidated[jobs[0].job_id]


# --------------------------------------------------------------------------- DEF-07
def test_def07_qualification_resources_are_package_data():
    tomllib = pytest.importorskip("tomllib")  # stdlib on 3.11+

    import qualification.reference as q

    pkg = Path(q.__file__).resolve().parent
    assert q.FIXTURE_PATH.parent == pkg and q.RECORD_PATH.parent == pkg
    assert q.FIXTURE_PATH.is_file() and q.RECORD_PATH.is_file()
    cfg = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    assert "*.json" in cfg["tool"]["setuptools"]["package-data"]["qualification"]


# --------------------------------------------------------------------------- DEF-11 / K7
def _recovery(rhat_bad_for: tuple[str, ...]) -> RecoveryResult:
    rng = np.random.default_rng(0)
    n, d = 40, 4
    truth = rng.uniform(0.2, 0.8, size=(n, d))
    est = truth + 0.01 * rng.standard_normal((n, d))
    lo, hi = est - 0.05, est + 0.05
    names = ("alpha", "pi", "theta_base", "beta")
    rhat = np.ones((n, d))
    for nm in rhat_bad_for:
        rhat[:, names.index(nm)] = 1.5
    comp_t = truth[:, 0] * truth[:, 1]
    res = RecoveryResult(
        channels=("behavior",),
        estimator="mcmc",
        n_subjects=n,
        truth=truth,
        estimate=est,
        lower=lo,
        upper=hi,
        pair_abs_corr=np.zeros((d, d)),
        composite_truth=comp_t,
        composite_estimate=comp_t,
        composite_lower=comp_t - 0.05,
        composite_upper=comp_t + 0.05,
        rhat_max=rhat.max(axis=0),
        ess_min=np.full(d, 1e4),
        rhat_all=rhat,
        ess_all=np.full((n, d), 1e4),
    )
    from estimation.recovery_simulation import _metrics

    for j, nm in enumerate(names):
        res.metrics[nm] = _metrics(truth[:, j], est[:, j], lo[:, j], hi[:, j], 1.0)
    res.composite_metrics = {
        "r": 0.99,
        "coverage95": 0.95,
        "nrmse": 0.01,
        "bias": 0.0,
        **{
            k: v
            for k, v in _metrics(comp_t, comp_t, comp_t - 0.05, comp_t + 0.05, 1.0).items()
            if k not in ("r", "coverage95")
        },
    }
    return res


class TestK7CompositeConvergence:
    def test_unconverged_factor_makes_composite_indeterminate(self):
        cls = gt.classify_k7c(_recovery(("pi",)))
        assert cls["pi"]["status"] == "INDETERMINATE"
        assert cls[COMPOSITE_NAME]["converged"] is False
        assert cls[COMPOSITE_NAME]["status"] == "INDETERMINATE"

    def test_converged_composite_can_be_confirmed(self):
        cls = gt.classify_k7c(_recovery(()))
        assert cls[COMPOSITE_NAME]["converged"] is True
        assert cls[COMPOSITE_NAME]["status"] == "CONFIRMED"


class TestDef11GateApproval:
    def test_hand_edited_table_exposes_nothing(self):
        table = {
            "approval": None,
            "parameters": [
                {
                    "parameter": "alpha",
                    "k7a": "CONFIRMED",
                    "k7b": "CONFIRMED",
                    "k7c": "CONFIRMED",
                    "exposable": True,
                }
            ],
        }
        assert gt.allow_list(table) == []
        table["approval"] = {"approver": "a", "date": "d", "reference": "r"}
        assert gt.allow_list(table) == ["alpha"]
        table["parameters"][0]["k7b"] = "PENDING_DATA"
        assert gt.allow_list(table) == []

    def test_approval_requires_named_fields(self):
        with pytest.raises(ValueError, match="required"):
            gt.approve_gate_table({"parameters": []}, k7b={}, approver=" ", date="d", reference="r")


# --------------------------------------------------------------------------- DEF-12
class TestDef12EmpiricalLoader:
    def _loader(self):
        return EmpiricalDataLoader(DatasetConfig(name="t", data_type="eeg", fs=100.0))

    def test_numpy_records_provenance_and_refuses_pickles(self, tmp_path):
        good = tmp_path / "x.npy"
        np.save(good, np.zeros((2, 10)))
        out = self._loader().load_eeg_dataset(str(good))
        assert out["provenance"]["parser"] == "numpy" and len(out["provenance"]["sha256"]) == 64
        obj = tmp_path / "obj.npy"
        np.save(obj, np.array([{"a": 1}], dtype=object), allow_pickle=True)
        with pytest.raises(DatasetLoadError):
            self._loader().load_eeg_dataset(str(obj))

    def test_corrupt_file_is_a_load_error_not_a_fallback(self, tmp_path):
        bad = tmp_path / "x.npy"
        bad.write_bytes(b"garbage")
        with pytest.raises(DatasetLoadError):
            self._loader().load_eeg_dataset(str(bad))

    def test_missing_optional_parser_and_unsupported_format(self, tmp_path, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "mne", None)
        with pytest.raises(OptionalDependencyError, match="mne"):
            self._loader().load_eeg_dataset(str(tmp_path / "rec.fif"))
        with pytest.raises(UnsupportedFormatError):
            self._loader().load_eeg_dataset(str(tmp_path / "rec.xyz"))

    def test_mne_parse_failure_is_reported(self, tmp_path, monkeypatch):
        import sys
        from unittest.mock import MagicMock

        mne = MagicMock()
        mne.io.read_raw.side_effect = RuntimeError("corrupt header")
        monkeypatch.setitem(sys.modules, "mne", mne)
        with pytest.raises(DatasetLoadError, match="corrupt header"):
            self._loader().load_eeg_dataset(str(tmp_path / "rec.edf"))


# --------------------------------------------------------------------------- INT-001
def test_int001_exchange_payload_requires_all_versions():
    run = create_model("apgi_reduced", {}, seed=0).run(np.linspace(0, 1, 5))
    out = to_canonical_outputs(run, model_id="apgi_reduced", model_version="1.0.0")
    with pytest.raises(IncompletePayloadError, match="protocol_version"):
        out.exchange_payload()
    out.protocol_version = "1.2"
    out.producer = {"name": "apgi-implementation", "version": "1.1.0"}
    payload = out.exchange_payload()
    assert payload["protocol_version"] == "1.2" and payload["producer"]["version"] == "1.1.0"
    assert payload["schema_version"]


# --------------------------------------------------------------------------- DAT-002
class TestDat002Quarantine:
    def test_every_record_gets_an_outcome_and_counts_reconcile(self):
        recs = [
            {"trial_id": 1, "x_e": 0.1, "seq": 1},
            {"trial_id": 2, "x_e": math.nan, "seq": 2},
            {"trial_id": 1, "x_e": 0.1, "seq": 1},
            {"trial_id": 1, "x_e": 0.9, "seq": 3},
            {"trial_id": 4, "x_e": 0.2, "seq": 4},
            {"trial_id": 3, "x_e": 0.3, "seq": 0},
            {"trial_id": 9, "x_e": 0.3, "seq": 5},
            {"x_e": 0.3},
            "not-a-record",
        ]
        rep = screen_records(
            recs, numeric_fields=("x_e",), seq_field="seq", expected_ids=[1, 2, 3, 4, 5]
        )
        c = rep.counts
        assert (
            c["accepted"],
            c["malformed"],
            c["duplicate"],
            c["conflict"],
            c["delayed"],
            c["unmatched"],
            c["missing"],
        ) == (2, 3, 1, 1, 1, 1, 3)
        assert [r["trial_id"] for r in rep.accepted] == [1, 4]
        assert set(rep.missing) == {2, 3, 5}
        assert rep.as_dict()["n_received"] == len(recs)

    def test_reconcile_detects_lost_records(self):
        rep = ScreeningReport(
            accepted=[{}], quarantined=[], missing=[], n_received=2, n_expected=None
        )
        with pytest.raises(ReconciliationError):
            rep.reconcile()


# --------------------------------------------------------------------------- SCI-003 / 005
class TestSci003Levels:
    def test_canonical_bands_and_boundaries(self):
        assert levels.LEVEL_LABELS == ("L0", "L1", "L2", "L3", "L4")
        assert levels.level_of(0.010) == "L0" and levels.level_of(0.1) == "L0"
        assert levels.level_of(0.101) == "L1" and levels.level_of(3.0) == "L1"
        assert levels.level_of(3.01) is None  # specification gap
        assert levels.level_of(60.0) == "L2" and levels.level_of(1800.0) == "L2"
        assert levels.level_of(2 * 3600.0) == "L3" and levels.level_of(365 * 86400.0) == "L4"
        assert levels.level_of(0.005) is None
        with pytest.raises(ValueError):
            levels.level_of(0.0)
        levels.check_in_level("L2", 120.0)
        with pytest.raises(ValueError, match="outside L1"):
            levels.check_in_level("L1", 10.0)
        with pytest.raises(ValueError, match="unknown"):
            levels.check_in_level("L9", 1.0)

    def test_index_convention(self):
        assert levels.label_for_index(0) == "L1" and levels.index_for_label("L4") == 3
        with pytest.raises(ValueError):
            levels.index_for_label("L0")
        with pytest.raises(ValueError):
            levels.label_for_index(4)

    def test_demo_config_departure_is_reported(self):
        rows = levels.audit_tau_ms([10.0 * 1.6**i for i in range(3)])
        assert all(not r["conforms"] and r["actual"] == "L0" for r in rows)


class TestSci005DefaultProvenance:
    def test_every_default_has_a_registered_basis_and_source(self):
        rows = all_defaults()
        assert rows and all(r.basis in (*BASES, UNCLASSIFIED) and r.source for r in rows)

    def test_registry_matches_code_defaults(self):
        """Drift check: a changed default in code fails until the registry is updated."""
        import models.reduced as red

        p = ReducedParams()
        for name in ("alpha", "pi", "theta_base", "beta", "sigma_r", "sigma_y"):
            assert provenance_for(name).value == getattr(p, name)
        for const in ("LAM", "RHO", "KAPPA", "DELTA", "TAU_SIGMA", "GAMMA_PHI"):
            assert provenance_for(const).value == getattr(red, const)
        cfg_rows = {r.name for r in all_defaults() if r.scope == "config.CONFIG"}
        numeric = {
            k for k, v in CONFIG.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        assert cfg_rows == numeric
