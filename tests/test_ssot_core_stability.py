"""SSOT §23 TC-13 for the v1.2 minimal core (core/ssot_core.py).

Overflow, NaN and division by zero have defined handling:
* non-finite inputs raise NonFiniteInputError before any state change (AUD-DEF-01);
* a step that would overflow from finite inputs raises NonFiniteStateError (an
  OverflowError) and leaves state, statistics, RNG and history unchanged;
* division by zero cannot occur: constant input drives σ_ref and Σ to 0 and every
  denominator stays ≥ ε_stab (apgi-lean: Apgi.traj_denoms_pos).
"""
from __future__ import annotations

import copy
import math
import pickle
import warnings

import pytest

import config as PCFG
from core.finite import NonFiniteInputError, NonFiniteStateError


def _pipe():
    import pipeline as PP

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = PP.APGIPipeline(copy.deepcopy(PCFG.CONFIG))
    assert p.model_form == "ssot_core"
    import numpy as np

    p.rng = np.random.default_rng(20261127)  # CONFIG seed is None; fix the stream for comparisons
    return p


def _snapshot(p) -> dict:
    out = {}
    for k, v in vars(p).items():
        try:
            out[k] = pickle.dumps(v)
        except Exception:  # noqa: BLE001 - unpicklable helpers are compared by repr
            out[k] = repr(v)
    return out


def _warm(p, n=20):
    for k in range(n):
        p.step(0.1 * k, -0.05 * k)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("channel", ["x_e", "x_i"])
def test_non_finite_input_rejected_atomically(bad, channel):
    p = _pipe()
    _warm(p)
    before = _snapshot(p)
    args = (bad, 0.0) if channel == "x_e" else (0.0, bad)
    with pytest.raises(NonFiniteInputError):
        p.step(*args)
    assert _snapshot(p) == before


@pytest.mark.parametrize("big", [1e200, -1e200, 1e308])
def test_overflow_rejected_atomically(big):
    p = _pipe()
    _warm(p)
    before = _snapshot(p)
    with pytest.raises(NonFiniteStateError) as exc:
        p.step(big, 0.0)
    assert isinstance(exc.value, OverflowError)
    assert _snapshot(p) == before
    # The pipeline is still usable and continues as if the rejected call never happened.
    q = _pipe()
    _warm(q)
    assert p.step(0.2, 0.1) == q.step(0.2, 0.1)


def test_constant_input_has_no_division_by_zero():
    p = _pipe()
    for _ in range(3000):
        r = p.step(1.0, 1.0)
    for key in ("S", "S_inst", "theta", "theta_eff", "B_prop", "pi_e", "pi_i", "z_e_norm", "z_i_norm"):
        assert math.isfinite(float(r[key])), key
    assert r["pi_e"] == pytest.approx(float(PCFG.CONFIG.get("pi_max", 10.0)))
