"""PyMC transcription of the reference likelihood (S1-IMP-09.1).

Skipped unless PyMC is importable (`pip install "apgi[estimation]"`). In CI it runs in a
dedicated job so the heavy PyMC install cannot disturb the other jobs.
"""

from __future__ import annotations

import numpy as np
import pytest

pm = pytest.importorskip("pymc")

from estimation import Observed, build_pymc_model, generate_dataset, sample_pymc  # noqa: E402
from estimation.likelihood import log_posterior_u  # noqa: E402
from models.reduced import ReducedParams  # noqa: E402

BASE = ReducedParams()
FULL = ["behavior", "ignition", "amplitude"]


@pytest.fixture(scope="module")
def data():
    ds = generate_dataset(BASE, rng=3)
    return Observed(ds.stimulus, ds.B, ds.report, ds.amplitude)


@pytest.mark.parametrize("channels", [["behavior"], ["ignition"], ["amplitude"], FULL])
def test_logp_equals_numpy_reference_up_to_a_constant(data, channels):
    """PyMC's joint logp and the numpy log posterior differ only by the prior's normalising constant."""
    model = build_pymc_model(data, channels, BASE)
    logp = model.compile_logp()
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(5):
        u = rng.normal(size=4)
        diffs.append(float(logp({"u": u})) - log_posterior_u(u, BASE, data, channels))
    assert np.ptp(diffs) < 1e-8, diffs


def test_nuts_posterior_is_close_to_the_truth(data):
    post = sample_pymc(data, FULL, BASE, draws=250, tune=250, chains=2, seed=1)
    assert post.samples.shape[1] == 4 and post.rhat is not None and post.ess is not None
    assert np.all(post.rhat < 1.1)
    truth = BASE.estimable()
    assert abs(post.mean[2] - truth[2]) < 0.2  # θ₀
    assert abs(post.mean[1] * post.mean[0] - truth[0] * truth[1]) < 0.3  # αΠ
