"""PyMC transcription of the reference-model likelihood (S1-IMP-09.1).

PyMC is an *optional* dependency (``pip install apgi[estimation]``); importing
this module never imports it. Everything that needs it goes through
:func:`require_pymc`, which raises a clear :class:`ImportError`.

The model is a line-by-line transcription of
:func:`estimation.likelihood.log_likelihood` (the numpy reference) with the
same priors on the same unconstrained scale. ``tests/test_estimation_pymc.py``
verifies that the two agree to machine precision at random points, so NUTS
output is a posterior over *the* reference model rather than a lookalike.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Sequence
from typing import Any

import numpy as np

from models.parameters import REDUCED_PARAMETER_NAMES
from models.reduced import (
    CHANNELS,
    DELTA,
    GAMMA_PHI,
    KAPPA,
    LAM,
    RHO,
    TAU_SIGMA,
    ReducedParams,
)

from .fitting import PosteriorSummary
from .likelihood import BOUNDS, PRIOR_MEAN_U, PRIOR_SD_U, Observed, check_channels

__all__ = ["pymc_available", "require_pymc", "build_pymc_model", "sample_pymc"]


def pymc_available() -> bool:
    """True if ``pymc`` can be imported (without importing it)."""
    return importlib.util.find_spec("pymc") is not None


def require_pymc() -> Any:
    """Import and return the ``pymc`` module, or raise a helpful ``ImportError``."""
    if not pymc_available():
        raise ImportError(
            "PyMC is required for NUTS estimation: install with `pip install apgi[estimation]`. "
            "The numpy estimators in estimation.fitting need no extra dependency."
        )
    return importlib.import_module("pymc")


def build_pymc_model(
    data: Observed,
    channels: Sequence[str] = CHANNELS,
    base: ReducedParams | None = None,
) -> Any:
    """Build the PyMC model for one subject.

    Args:
        data: Observations (``B`` is required; ``report``/``amplitude`` as needed).
        channels: Channels entering the likelihood.
        base: Supplies the known noise levels ``sigma_r`` and ``sigma_y``.

    Returns:
        A ``pymc.Model`` with free variable ``u`` (unconstrained) and
        deterministics named after :data:`models.parameters.REDUCED_PARAMETER_NAMES`.
    """
    pm = require_pymc()
    import pytensor
    import pytensor.tensor as pt

    ch = check_channels(channels)
    base = base or ReducedParams()
    x = np.asarray(data.stimulus, dtype="float64")
    B = np.asarray(data.B, dtype="float64")
    B_prev = np.concatenate([[0.0], B[:-1]])
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]

    with pm.Model() as model:
        u = pm.Normal("u", mu=PRIOR_MEAN_U, sigma=PRIOR_SD_U, shape=len(REDUCED_PARAMETER_NAMES))
        p = lo + (hi - lo) * pm.math.sigmoid(u)
        alpha = pm.Deterministic("alpha", p[0])
        pi = pm.Deterministic("pi", p[1])
        theta_base = pm.Deterministic("theta_base", p[2])
        beta = pm.Deterministic("beta", p[3])
        k = alpha * pi / (1.0 + alpha * pi)

        outs, _updates = pytensor.scan(
            fn=_scan_step,
            sequences=[pt.as_tensor_variable(a) for a in (x, B_prev, B)],
            outputs_info=[pt.zeros(()), pt.zeros(()), theta_base, None, None, None],
            non_sequences=[k, pi, beta, theta_base],
            n_steps=int(x.shape[0]),
        )
        xhat_pre, S_seq, theta_pre = outs[3], outs[4], outs[5]
        terms = []
        if "behavior" in ch:
            terms.append(
                pm.logp(pm.Normal.dist(xhat_pre, base.sigma_r), np.asarray(data.report)).sum()
            )
        if "ignition" in ch:
            terms.append(
                pm.logp(pm.Bernoulli.dist(logit_p=(S_seq - theta_pre) / TAU_SIGMA), B).sum()
            )
        if "amplitude" in ch:
            terms.append(
                pm.logp(pm.Normal.dist(S_seq, base.sigma_y), np.asarray(data.amplitude)).sum()
            )
        pm.Potential("loglik", sum(terms))
    return model


def _scan_step(x_t, b_prev, b_t, xhat, S, theta, k, pi, beta, theta_base):  # type: ignore[no-untyped-def]
    """One trial of the recurrence (shared by ``scan``); returns states then records."""
    import pytensor.tensor as pt

    eps = x_t - xhat
    carry = 1.0 + (RHO - 1.0) * b_prev
    S_new = (1.0 - LAM) * carry * S + LAM * (pt.tanh(GAMMA_PHI * pi * pt.abs(eps)) + beta)
    theta_next = theta_base + (theta - theta_base) * np.exp(-KAPPA) + DELTA * b_t
    return xhat + k * eps, S_new, theta_next, xhat, S_new, theta


def sample_pymc(
    data: Observed,
    channels: Sequence[str] = CHANNELS,
    base: ReducedParams | None = None,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int | None = 0,
    progressbar: bool = False,
) -> PosteriorSummary:
    """Run NUTS and return a :class:`~estimation.fitting.PosteriorSummary`.

    ``rhat`` and ``ess`` are ArviZ's rank-normalised R̂ and bulk ESS.
    """
    pm = require_pymc()
    model = build_pymc_model(data, channels, base)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            progressbar=progressbar,
            cores=1,
            compute_convergence_checks=False,
        )
    az = importlib.import_module("arviz")
    names = list(REDUCED_PARAMETER_NAMES)
    post = idata.posterior
    samples = np.stack([post[n].values.reshape(-1) for n in names], axis=1)
    summ = az.summary(idata, var_names=names, kind="diagnostics")
    return PosteriorSummary(
        "pymc",
        samples.mean(axis=0),
        samples,
        float("nan"),
        rhat=np.array([summ.loc[n, "r_hat"] for n in names]),
        ess=np.array([summ.loc[n, "ess_bulk"] for n in names]),
    )
