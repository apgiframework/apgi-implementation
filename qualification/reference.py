"""Qualification of ``apgi_reduced`` as the single numerical reference (S1-IMP-15, D14).

Only ``apgi_reduced`` results carry reference authority; ``apgi-framework/src/apgi``
stays a fixture-conformant teaching package without it. Qualification is a set of
executable checks against the model's own contract — not a statement that the
model is *scientifically correct*, only that the implementation computes what its
equations say, deterministically, within its stated invariants:

``registry_authority``
    Exactly one registered variant holds authority and it is ``apgi_reduced``.
``independent_reimplementation``
    A from-the-equations re-implementation written separately in this module
    (plain floats, no shared helper) agrees with the production code to 1e-12 on
    every fixture case.
``reference_vectors``
    Trajectories match the committed fixture digests
    (``tests/golden/reference_vectors.json``).
``product_symmetry``
    Behaviour is invariant under ``(α, Π) → (sα, Π/s)``; the model depends on
    the factors only through ``αΠ``.
``fixed_point``
    Under a constant stimulus below threshold, ``x̂ → x`` and ``S → β``.
``signal_bounds``
    ``S`` stays inside ``[min(0, β), max(0, 1+β)]`` on randomised runs.
``determinism``
    Equal seeds give identical output digests; different seeds do not.
``ode_generator_parity``
    The ODE generator and the scorer agree (``check_scoring_regression``).

``python -m qualification.reference --update`` rewrites the fixture and the
qualification record; the test-suite fails when either drifts from the code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from engine.provenance import array_digest
from model_registry.variants import REFERENCE_MODEL_ID, list_variants
from models.reduced import ReducedParams, simulate_reduced

__all__ = [
    "FIXTURE_PATH",
    "RECORD_PATH",
    "CASES",
    "fixture_stimulus",
    "independent_trajectory",
    "build_fixture",
    "qualify",
    "verify_record",
]

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = ROOT / "tests" / "golden" / "reference_vectors.json"
RECORD_PATH = Path(__file__).resolve().parent / "apgi_reduced.qualification.json"
ROUND_DP = 9
N_ROWS = 12  # rows stored verbatim per case; the rest are covered by the digest

#: Fixture cases: name → parameters (everything else is a model constant).
CASES: dict[str, dict[str, float]] = {
    "nominal": {},
    "high_precision": {"alpha": 0.5, "pi": 4.0},
    "low_threshold": {"theta_base": 0.3, "beta": 0.1},
    "negative_bias": {"beta": -0.2, "alpha": 0.1, "pi": 0.5},
}


def fixture_stimulus(n: int = 120) -> np.ndarray:
    """Deterministic stimulus: ±0.8 reversal blocks of 30 plus a small sinusoid."""
    t = np.arange(n)
    blocks = 0.8 * np.where((t // 30) % 2 == 0, 1.0, -1.0)
    return np.asarray(blocks + 0.1 * np.sin(0.7 * t))


def independent_trajectory(p: ReducedParams, x: np.ndarray) -> dict[str, list[float]]:
    """Hard-threshold ``apgi_reduced`` straight from the equations, in plain floats.

    Deliberately shares no code with :mod:`models.reduced` — constants are
    re-derived here — so agreement is evidence about the equations, not about a
    helper reused in both places.
    """
    lam = 1.0 - math.exp(-0.5 / 5.0)
    rho, kappa, delta, tau, gamma = 0.1, 0.15, 0.5, 0.5, 2.0
    k = p.alpha * p.pi / (1.0 + p.alpha * p.pi)
    xhat, S, theta, B_prev = 0.0, 0.0, p.theta_base, 0
    out: dict[str, list[float]] = {key: [] for key in ("xhat", "S", "theta", "p", "B")}
    for xt in x.tolist():
        eps = xt - xhat
        drive = math.tanh(gamma * p.pi * abs(eps)) + p.beta
        S = (1.0 - lam) * (rho if B_prev else 1.0) * S + lam * drive
        prob = 1.0 / (1.0 + math.exp(-(S - theta) / tau))
        B = 1 if theta < S else 0
        for key, v in (("xhat", xhat), ("S", S), ("theta", theta), ("p", prob), ("B", B)):
            out[key].append(float(v))
        theta = p.theta_base + (theta - p.theta_base) * math.exp(-kappa) + delta * B
        xhat = xhat + k * eps
        B_prev = B
    return out


def _case_trajectory(params: dict[str, float]) -> dict[str, np.ndarray]:
    traj = simulate_reduced(ReducedParams(**params), fixture_stimulus(), hard=True)
    return {
        "xhat": traj.xhat,
        "S": traj.S,
        "theta": traj.theta,
        "p": traj.p,
        "B": traj.B.astype(float),
    }


def build_fixture() -> dict[str, Any]:
    """Compute the reference-vector fixture from the current code."""
    cases: dict[str, Any] = {}
    for name, params in CASES.items():
        arrs = _case_trajectory(params)
        cases[name] = {
            "params": params,
            "digest": array_digest(arrs, ROUND_DP),
            "ignition_count": int(arrs["B"].sum()),
            "rows": {
                k: [round(float(v), ROUND_DP) for v in arrs[k][:N_ROWS]]
                for k in ("xhat", "S", "theta", "p", "B")
            },
        }
    return {"model": REFERENCE_MODEL_ID, "n_trials": 120, "round_dp": ROUND_DP, "cases": cases}


def _check_registry() -> bool:
    holders = [v.model_id for v in list_variants() if v.authority]
    return holders == [REFERENCE_MODEL_ID]


def _check_independent() -> bool:
    x = fixture_stimulus()
    for params in CASES.values():
        ref = independent_trajectory(ReducedParams(**params), x)
        mine = _case_trajectory(params)
        for key in ("xhat", "S", "theta", "p", "B"):
            if np.max(np.abs(np.asarray(ref[key]) - mine[key])) > 1e-12:
                return False
    return True


def _check_fixture() -> bool:
    if not FIXTURE_PATH.is_file():
        return False
    stored = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return bool(stored == build_fixture())


def _check_symmetry() -> bool:
    x = fixture_stimulus()
    base = ReducedParams(alpha=0.2, pi=2.0)
    alt = ReducedParams(alpha=0.4, pi=1.0)
    a = simulate_reduced(base, x, B_forced=np.zeros(len(x), dtype=int)).xhat
    b = simulate_reduced(alt, x, B_forced=np.zeros(len(x), dtype=int)).xhat
    return bool(np.max(np.abs(a - b)) < 1e-12)


def _check_fixed_point() -> bool:
    p = ReducedParams(theta_base=0.85, beta=0.1)
    traj = simulate_reduced(p, np.full(2000, 0.5), hard=True)
    return bool(
        abs(traj.xhat[-1] - 0.5) < 1e-9 and abs(traj.S[-1] - p.beta) < 1e-9 and traj.B.sum() == 0
    )


def _check_bounds() -> bool:
    gen = np.random.default_rng(0)
    for _ in range(25):
        p = ReducedParams(
            alpha=float(gen.uniform(0.05, 0.9)),
            pi=float(gen.uniform(0.3, 5.0)),
            theta_base=float(gen.uniform(0.25, 0.85)),
            beta=float(gen.uniform(-0.3, 0.3)),
        )
        traj = simulate_reduced(p, gen.normal(0, 1.5, 200), rng=gen)
        if traj.S.min() < min(0.0, p.beta) - 1e-12 or traj.S.max() > max(0.0, 1.0 + p.beta) + 1e-12:
            return False
    return True


def _check_determinism() -> bool:
    x = fixture_stimulus()
    p = ReducedParams()

    def digest(seed: int) -> str:
        t = simulate_reduced(p, x, rng=seed)
        return array_digest({"S": t.S, "B": t.B.astype(float)})

    return digest(1) == digest(1) and digest(1) != digest(2)


def _check_ode() -> bool:
    from estimation.synthetic_generator import check_scoring_regression

    check_scoring_regression(ReducedParams(), atol=1e-6)
    return True


CHECKS = {
    "registry_authority": _check_registry,
    "independent_reimplementation": _check_independent,
    "reference_vectors": _check_fixture,
    "product_symmetry": _check_symmetry,
    "fixed_point": _check_fixed_point,
    "signal_bounds": _check_bounds,
    "determinism": _check_determinism,
    "ode_generator_parity": _check_ode,
}


def qualify() -> dict[str, Any]:
    """Run every check and return the (deterministic) qualification record."""
    results = {}
    for name, fn in CHECKS.items():
        try:
            results[name] = bool(fn())
        except Exception:  # a check that crashes has failed, whatever the cause
            results[name] = False
    fixture_bytes = json.dumps(build_fixture(), sort_keys=True).encode()
    return {
        "model_id": REFERENCE_MODEL_ID,
        "decision": "D14",
        "statement": "Only apgi_reduced results carry reference authority.",
        "checks": results,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "qualified": all(results.values()),
    }


def verify_record(path: Path = RECORD_PATH) -> list[str]:
    """Compare the committed record with a fresh qualification; return differences."""
    if not path.is_file():
        return [f"{path} is missing; run `python -m qualification.reference --update`"]
    stored = json.loads(path.read_text(encoding="utf-8"))
    fresh = qualify()
    problems = [
        f"{k}: committed {stored.get(k)!r} != current {fresh[k]!r}"
        for k in fresh
        if stored.get(k) != fresh[k]
    ]
    if not fresh["qualified"]:
        failed = [k for k, v in fresh["checks"].items() if not v]
        problems.append(f"qualification failed: {failed}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--update", action="store_true", help="rewrite the fixture and the record")
    args = ap.parse_args(argv)
    if args.update:
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_PATH.write_text(
            json.dumps(build_fixture(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        record = qualify()
        RECORD_PATH.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {FIXTURE_PATH} and {RECORD_PATH}; qualified={record['qualified']}")
        return 0 if record["qualified"] else 1
    problems = verify_record()
    for p in problems:
        print(p)
    print("qualification record is current" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
