import numpy as np
import pytest

from core.thermodynamics import (
    compute_information_bits,
    compute_landauer_cost,
    compute_landauer_cost_batch,
    compute_metabolic_efficiency,
    estimate_temperature_from_cost,
    thermodynamic_cost_trajectory,
    validate_thermodynamic_constraint,
)


def test_compute_landauer_cost():
    # S <= eps
    assert compute_landauer_cost(0.01, 0.01) == 0.0
    # Simple case
    # S=1.0, eps=0.01 -> n_erase = log2(100) ~ 6.64
    # E = 6.64 * 1.38e-23 * 310 * ln2 ~ 1.97e-20 J
    res = compute_landauer_cost(1.0, 0.01)
    assert pytest.approx(res, rel=1e-2) == 1.97e-20
    # Calibrated mode
    # E = 6.64 * 2.0 = 13.28
    res_cal = compute_landauer_cost(1.0, 0.01, kappa_meta=2.0, kappa_units="joules_per_bit")
    assert pytest.approx(res_cal) == np.log2(100) * 2.0
    # Error cases
    with pytest.raises(ValueError, match="eps must be > 0"):
        compute_landauer_cost(1.0, 0.0)
    with pytest.raises(ValueError, match="k_b must be > 0"):
        compute_landauer_cost(1.0, 0.01, k_b=0.0)
    with pytest.raises(ValueError, match="T_env must be > 0"):
        compute_landauer_cost(1.0, 0.01, T_env=0.0)
    with pytest.raises(ValueError, match="r_l must be > 0"):
        compute_landauer_cost(1.0, 0.01, kappa_meta=0.0)
    with pytest.raises(ValueError, match="r_l_units must be"):
        compute_landauer_cost(1.0, 0.01, kappa_units="invalid")


def test_compute_landauer_cost_batch():
    S_vals = np.array([0.01, 1.0])
    res = compute_landauer_cost_batch(S_vals, 0.01)
    assert res[0] == 0.0
    assert pytest.approx(res[1], rel=1e-2) == 1.97e-20
    # Calibrated batch
    res_cal = compute_landauer_cost_batch(
        S_vals, 0.01, kappa_meta=1.0, kappa_units="joules_per_bit"
    )
    assert res_cal[1] == np.log2(100)


def test_validate_thermodynamic_constraint():
    # S <= eps case
    res_small = validate_thermodynamic_constraint(1.0, 0.01, 0.01)
    assert res_small["satisfied"] is True
    assert res_small["E_min"] == 0.0
    # Satisfied
    # cost = 2e-20 > 1.97e-20
    res_ok = validate_thermodynamic_constraint(2e-20, 1.0, 0.01)
    assert res_ok["satisfied"] is True
    # Violated
    res_bad = validate_thermodynamic_constraint(1e-20, 1.0, 0.01)
    assert res_bad["satisfied"] is False
    assert res_bad["violation"] > 0


def test_compute_information_bits():
    assert compute_information_bits(1.0, 0.01) == pytest.approx(np.log2(100))
    assert compute_information_bits(0.005, 0.01) == 0.0


def test_inverses():
    # Efficiency
    cost = compute_landauer_cost(1.0, 0.01)
    assert pytest.approx(compute_metabolic_efficiency(cost, 1.0, 0.01)) == 1.0
    # Temperature
    assert pytest.approx(estimate_temperature_from_cost(cost, 1.0, 0.01)) == 310.0
    # Error cases
    with pytest.raises(ValueError, match="S must be > eps"):
        compute_metabolic_efficiency(1.0, 0.01, 0.01)
    with pytest.raises(ValueError, match="S must be > eps"):
        estimate_temperature_from_cost(1.0, 0.01, 0.01)
    # Zero denominator cases
    with pytest.raises(ValueError, match="Denominator is zero"):
        compute_metabolic_efficiency(1.0, 1.0, 0.01, T_env=0.0)
    with pytest.raises(ValueError, match="Denominator is zero"):
        estimate_temperature_from_cost(1.0, 1.0, 0.01, kappa_meta=0.0)


def test_thermodynamic_cost_trajectory():
    S_hist = np.array([0.1, 0.5, 1.0])
    res = thermodynamic_cost_trajectory(S_hist, eps=0.01)
    assert len(res["costs"]) == 3
    assert res["total_cost"] == np.sum(res["costs"])
    assert res["total_bits"] == np.sum(res["bits_history"])
