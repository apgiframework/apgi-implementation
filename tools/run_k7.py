"""Run the K7 desk-science chain: K7a → K7c → gate table (S1-K7-01).

    python tools/run_k7.py --out gates --subjects 60 --estimator mcmc

Writes, under ``--out``:

* ``k7a-results.json``  — identifiability tables for each channel set;
* ``k7c-results.json``  — recovery tables (production design and behaviour-only);
* ``gate-table.yaml`` / ``.json`` — the per-parameter gate table;
* ``k7-manifest.json``  — run manifest with seeds, settings and result digests.

The production design uses all three observation channels; the behaviour-only
results are included to document that the ``α``/``Π`` and ``θ₀``/``β``
degeneracies are properties of the design, not of the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.manifest import build_manifest  # noqa: E402
from estimation.gate_table import build_gate_table, write_gate_table  # noqa: E402
from estimation.recovery_simulation import run_recovery  # noqa: E402
from estimation.structural_identifiability import AnalysisSetup, run_k7a  # noqa: E402
from models.reduced import CHANNELS  # noqa: E402

CHANNEL_SETS: tuple[tuple[str, ...], ...] = (
    ("behavior",),
    ("behavior", "ignition"),
    CHANNELS,
)


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="gates", help="output directory")
    ap.add_argument("--subjects", type=int, default=60, help="synthetic subjects for K7c")
    ap.add_argument("--points", type=int, default=100, help="K7a plausible-range sweep points")
    ap.add_argument("--starts", type=int, default=16, help="K7a multi-start searches")
    ap.add_argument("--estimator", choices=("laplace", "mcmc"), default="mcmc")
    ap.add_argument("--draws", type=int, default=3000, help="MCMC draws per chain")
    ap.add_argument("--warmup", type=int, default=1500, help="MCMC warm-up per chain")
    ap.add_argument("--seed", type=int, default=20261116)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    setup = AnalysisSetup(seed=args.seed)

    k7a = {
        "+".join(ch): run_k7a(ch, setup, n_points=args.points, n_starts=args.starts, seed=args.seed)
        for ch in CHANNEL_SETS
    }
    (out / "k7a-results.json").write_text(json.dumps(k7a, indent=2) + "\n", encoding="utf-8")
    print(f"K7a done ({time.time() - t0:.0f}s)")

    k7c_runs = {}
    for ch in (("behavior",), CHANNELS):
        k7c_runs["+".join(ch)] = run_recovery(
            args.subjects,
            ch,
            estimator=args.estimator,
            rng=args.seed,
            n_draws=args.draws,
            n_warmup=args.warmup,
        )
        print(f"K7c {'+'.join(ch)} done ({time.time() - t0:.0f}s)")
    (out / "k7c-results.json").write_text(
        json.dumps({k: v.as_dict() for k, v in k7c_runs.items()}, indent=2) + "\n", encoding="utf-8"
    )

    production = "+".join(CHANNELS)
    manifest = build_manifest(
        {"tool": "run_k7", "seed": args.seed, "args": vars(args), "model": "apgi_reduced"},
        extra={
            "k7a_sha256": _digest(k7a),
            "k7c_sha256": _digest({k: v.as_dict() for k, v in k7c_runs.items()}),
            "elapsed_s": round(time.time() - t0, 1),
        },
    )
    table = build_gate_table(
        k7a[production],
        k7c_runs[production],
        meta={
            "seed": args.seed,
            "k7a_sha256": manifest["k7a_sha256"],
            "k7c_sha256": manifest["k7c_sha256"],
            "git_commit": manifest["git"]["commit"],
            "model": "apgi_reduced",
            "contract_status": "provisional: S1-REG-04 mathematical-contracts.yaml not yet qualified",
        },
    )
    write_gate_table(table, out / "gate-table.yaml")
    (out / "k7-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"gate table written; allow-list = {table['allow_list']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
