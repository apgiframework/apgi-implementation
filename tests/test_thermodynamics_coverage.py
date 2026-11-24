import numpy as np
import pytest

from core.thermodynamics import (
    K_BOLTZMANN,
    LN2,
    T_ENV_DEFAULT,
    compute_information_bits,
    compute_landauer_cost,
    compute_landauer_cost_batch,
    compute_metabolic_efficiency,
    estimate_temperature_from_cost,
    thermodynamic_cost_trajectory,
    validate_thermodynamic_constraint,
)


def test_compute_landauer_cost():
    # S=1.0, eps=0.5 -> n_erase = log2(2) = 1.0
    # E_min = 1.0 * 1.0 * k_b * T * ln(2)
    expected = 1.0 * K_BOLTZMANN * T_ENV_DEFAULT * LN2
    assert pytest.approx(compute_landauer_cost(1.0, 0.5)) == expected
    # joules_per_bit mode
    assert compute_landauer_cost(1.0, 0.5, kappa_meta=2.0, kappa_units="joules_per_bit") == 2.0
    # S <= eps -> 0
    assert compute_landauer_cost(0.4, 0.5) == 0.0
    # Errors
    with pytest.raises(ValueError, match="eps must be > 0"):
        compute_landauer_cost(1.0, 0.0)
    with pytest.raises(ValueError, match="k_b must be > 0"):
        compute_landauer_cost(1.0, 0.5, k_b=0.0)
    with pytest.raises(ValueError, match="T_env must be > 0"):
        compute_landauer_cost(1.0, 0.5, T_env=0.0)
    with pytest.raises(ValueError, match="r_l must be > 0"):
        compute_landauer_cost(1.0, 0.5, kappa_meta=0.0)
    with pytest.raises(ValueError, match="r_l_units must be"):
        compute_landauer_cost(1.0, 0.5, kappa_units="invalid")


def test_compute_landauer_cost_batch():
    S_vals = np.array([0.4, 1.0])
    costs = compute_landauer_cost_batch(S_vals, eps=0.5)
    assert costs[0] == 0.0
    assert pytest.approx(costs[1]) == K_BOLTZMANN * T_ENV_DEFAULT * LN2
    # joules_per_bit batch
    costs_j = compute_landauer_cost_batch(
        S_vals, eps=0.5, kappa_meta=10.0, kappa_units="joules_per_bit"
    )
    assert costs_j[0] == 0.0
    assert costs_j[1] == 10.0


def test_validate_thermodynamic_constraint():
    # satisfied
    res = validate_thermodynamic_constraint(1e-10, 1.0, 0.5)
    assert res["satisfied"] is True
    # violated
    res_v = validate_thermodynamic_constraint(0.0, 1.0, 0.5)
    assert res_v["satisfied"] is False
    assert res_v["violation"] > 0
    # S <= eps case
    res_small = validate_thermodynamic_constraint(1.0, 0.1, 0.5)
    assert res_small["satisfied"] is True
    assert res_small["E_min"] == 0.0
    assert res_small["ratio"] == float("inf")
    res_zero = validate_thermodynamic_constraint(0.0, 0.1, 0.5)
    assert res_zero["ratio"] == 1.0


def test_compute_information_bits():
    assert compute_information_bits(1.0, 0.25) == 2.0
    assert compute_information_bits(0.1, 0.25) == 0.0


def test_inverse_calculations():
    # S=1.0, eps=0.5 -> n_erase=1
    # Cost = 1 * k_b * T * ln(2)
    cost = K_BOLTZMANN * T_ENV_DEFAULT * LN2
    assert pytest.approx(compute_metabolic_efficiency(cost, 1.0, 0.5)) == 1.0
    assert pytest.approx(estimate_temperature_from_cost(cost, 1.0, 0.5)) == T_ENV_DEFAULT
    with pytest.raises(ValueError, match="S must be > eps"):
        compute_metabolic_efficiency(1.0, 0.1, 0.5)
    with pytest.raises(ValueError, match="S must be > eps"):
        estimate_temperature_from_cost(1.0, 0.1, 0.5)
    # Zero denominator cases
    with pytest.raises(ValueError, match="Denominator is zero"):
        compute_metabolic_efficiency(1.0, 1.0, 0.5, k_b=0.0)
    with pytest.raises(ValueError, match="Denominator is zero"):
        estimate_temperature_from_cost(1.0, 1.0, 0.5, kappa_meta=0.0)


def test_thermodynamic_cost_trajectory():
    S_hist = np.array([0.1, 1.0, 2.0])
    res = thermodynamic_cost_trajectory(S_hist, eps=0.5)
    assert len(res["costs"]) == 3
    assert res["total_bits"] == compute_information_bits(1.0, 0.5) + compute_information_bits(
        2.0, 0.5
    )
    assert res["max_bits"] == 2.0  # log2(2.0/0.5) = 2.0
    assert "min_cost" in res
    assert "mean_bits" in res
