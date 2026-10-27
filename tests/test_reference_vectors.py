"""Reference vectors for the qualified numerical reference ``apgi_reduced`` (S1-IMP-13.1, S1-IMP-15).

Three independent kinds of evidence:

* **Hand-derived values** — closed-form numbers computed from the equations;
* **Independent re-implementation** — plain-float code in
  :mod:`qualification.reference` that shares no helper with the production module;
* **Frozen fixture** — trajectory digests in ``tests/golden/reference_vectors.json``.
  A deliberate change to the dynamics fails the fixture; regenerate with
  ``python -m qualification.reference --update`` and say so in the commit message.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from engine.provenance import array_digest
from models.reduced import LAM, ReducedParams, simulate_reduced
from qualification import reference as qref

pytestmark = pytest.mark.conformance


class TestHandDerivedValues:
    def test_lambda_is_exact_discretisation(self):
        # λ = 1 − exp(−dt/τ_S) with dt=0.5, τ_S=5 (MathSpec §3), not dt/τ_S.
        assert pytest.approx(0.09516258196404048, rel=1e-15) == LAM

    def test_constant_stimulus_gain_sequence(self):
        # With no ignition, x̂_t = x(1 − (1−k)^t) exactly.
        p = ReducedParams(alpha=0.3, pi=1.5, theta_base=0.85)
        k = 0.45 / 1.45
        tr = simulate_reduced(p, np.full(40, 0.5), hard=True)
        assert tr.B.sum() == 0
        expected = 0.5 * (1.0 - (1.0 - k) ** np.arange(40))
        np.testing.assert_allclose(tr.xhat, expected, rtol=0, atol=1e-13)

    def test_first_two_trials(self):
        p = ReducedParams()
        tr = simulate_reduced(p, np.array([1.0, 1.0]), hard=True)
        k = 0.45 / 1.45
        d0 = math.tanh(3.0) + 0.05
        d1 = math.tanh(3.0 * (1.0 - k)) + 0.05
        s0 = LAM * d0
        s1 = (1 - LAM) * s0 + LAM * d1
        np.testing.assert_allclose(tr.S, [s0, s1], atol=1e-14)
        np.testing.assert_allclose(tr.eps, [1.0, 1.0 - k], atol=1e-14)

    def test_refractory_boost_and_decay_by_hand(self):
        p = ReducedParams(theta_base=0.3)
        forced = np.array([1, 0, 0])
        tr = simulate_reduced(p, np.zeros(3), B_forced=forced)
        th1 = 0.3 + 0.0 * math.exp(-0.15) + 0.5
        th2 = 0.3 + (th1 - 0.3) * math.exp(-0.15)
        np.testing.assert_allclose(tr.theta, [0.3, th1, th2], atol=1e-14)

    def test_post_ignition_reset_retains_ten_percent(self):
        p = ReducedParams(beta=0.0)
        tr = simulate_reduced(p, np.array([1.0, 0.0]), B_forced=np.array([1, 0]))
        # Trial 2: x̂=k, ε=−k ... S₂ = (1−λ)·ρ·S₁ + λ·tanh(3k) with ρ = 0.1.
        k = 0.45 / 1.45
        s1 = tr.S[0]
        assert tr.S[1] == pytest.approx((1 - LAM) * 0.1 * s1 + LAM * math.tanh(3.0 * k), abs=1e-14)


class TestIndependentReimplementation:
    @pytest.mark.parametrize("case", list(qref.CASES))
    def test_matches_production_code(self, case):
        p = ReducedParams(**qref.CASES[case])
        x = qref.fixture_stimulus()
        ref = qref.independent_trajectory(p, x)
        mine = simulate_reduced(p, x, hard=True)
        for key, arr in (
            ("xhat", mine.xhat),
            ("S", mine.S),
            ("theta", mine.theta),
            ("p", mine.p),
            ("B", mine.B),
        ):
            np.testing.assert_allclose(arr, ref[key], rtol=0, atol=1e-12, err_msg=key)

    def test_fixture_cases_exercise_both_regimes(self):
        counts = {
            c: int(
                qref.independent_trajectory(
                    ReducedParams(**qref.CASES[c]), qref.fixture_stimulus()
                )["B"].count(1.0)
            )
            for c in qref.CASES
        }
        assert any(v == 0 for v in counts.values()) and any(v > 0 for v in counts.values()), counts


class TestFrozenFixture:
    def test_fixture_matches_current_code(self):
        stored = json.loads(qref.FIXTURE_PATH.read_text(encoding="utf-8"))
        assert stored == qref.build_fixture(), (
            "run `python -m qualification.reference --update` if intended"
        )

    @pytest.mark.parametrize("case", list(qref.CASES))
    def test_case_digest_and_rows(self, case):
        stored = json.loads(qref.FIXTURE_PATH.read_text(encoding="utf-8"))["cases"][case]
        traj = qref._case_trajectory(stored["params"])
        assert array_digest(traj, qref.ROUND_DP) == stored["digest"]
        for key, rows in stored["rows"].items():
            np.testing.assert_allclose(traj[key][: len(rows)], rows, atol=1e-9)
        assert int(traj["B"].sum()) == stored["ignition_count"]


class TestQualification:
    def test_every_check_passes(self):
        record = qref.qualify()
        assert record["qualified"], [k for k, v in record["checks"].items() if not v]

    def test_committed_record_is_current(self):
        assert qref.verify_record() == []

    def test_missing_record_reported(self, tmp_path):
        problems = qref.verify_record(tmp_path / "nope.json")
        assert problems and "missing" in problems[0]

    def test_tampered_record_detected(self, tmp_path):
        rec = json.loads(qref.RECORD_PATH.read_text(encoding="utf-8"))
        rec["fixture_sha256"] = "0" * 64
        f = tmp_path / "r.json"
        f.write_text(json.dumps(rec))
        assert any("fixture_sha256" in p for p in qref.verify_record(f))

    def test_a_failing_check_unqualifies(self, monkeypatch):
        monkeypatch.setitem(qref.CHECKS, "determinism", lambda: False)
        record = qref.qualify()
        assert record["qualified"] is False and record["checks"]["determinism"] is False

    def test_a_crashing_check_counts_as_failed(self, monkeypatch):
        def boom() -> bool:
            raise RuntimeError("x")

        monkeypatch.setitem(qref.CHECKS, "fixed_point", boom)
        assert qref.qualify()["checks"]["fixed_point"] is False

    def test_main_reports_current(self, capsys):
        assert qref.main([]) == 0
        assert "current" in capsys.readouterr().out
