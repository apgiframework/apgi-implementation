import numpy as np
import pytest

from core.threshold import (
    apply_ne_threshold_modulation,
    apply_refractory_boost,
    apply_serotonin_threshold_offset,
    compute_information_value,
    compute_information_value_with_bias,
    compute_metabolic_cost,
    compute_metabolic_cost_realistic,
    threshold_decay,
    update_threshold_discrete,
    update_threshold_ode_deprecated,
)


def test_compute_metabolic_cost():
    # C(t) = c1·S + c2·B_prev: S=1.0, B_prev=1, c1=2.0, c2=0.5 → 2.5
    assert pytest.approx(compute_metabolic_cost(1.0, B_prev=1, c1=2.0, c2=0.5)) == 2.5


def test_compute_metabolic_cost_realistic():
    # Basic
    assert pytest.approx(compute_metabolic_cost_realistic(1.0, 1, c1=2.0, c2=0.5)) == 2.5
    # Landauer enforcement (kappa_meta=1e20 absorbs J→AU unit conversion)
    # S=1.0, eps=0.1 -> n_erase = log2(10) ~ 3.32
    # E_phys = 3.32 * k_b * T * ln(2) ~ 1e-20 J
    # E_scaled = 1e-20 * kappa_meta = 1e-20 * 1e20 = 1.0
    # base_cost = 0.1 * 1.0 + 0 = 0.1
    # max(0.1, 1.0) = 1.0
    cost = compute_metabolic_cost_realistic(
        1.0, 0, c1=0.1, c2=0.0, eps_stab=0.1, enforce_landauer=True, kappa_meta=1e20
    )
    assert cost > 0.1
    # BOLD calibration mock-like (real calls to energy.bold_calibration)
    bold_cfg = {
        "bold_signal_change": 2.0,
        "conversion_factor": 1.0e-6,
        "tissue_volume": 1.0,
        "ignition_spike_factor": 1.1,
    }
    cost_bold = compute_metabolic_cost_realistic(
        1.0, 0, enforce_landauer=True, bold_calibration=bold_cfg
    )
    assert cost_bold > 0


def test_information_value():
    assert pytest.approx(compute_information_value(0.5, -0.2, v1=2.0, v2=3.0)) == 1.6
    # compute_information_value_with_bias is now identical to compute_information_value
    # (beta bias is pre-integrated into phi_i)
    assert pytest.approx(compute_information_value_with_bias(0.5, -0.3, v1=2.0, v2=3.0)) == 1.9


def test_modulations():
    assert pytest.approx(apply_ne_threshold_modulation(1.0, 0.5, 0.2)) == 1.1
    assert pytest.approx(threshold_decay(2.0, 1.0, 1.0)) == 1.0 + (1.0 * np.exp(-1.0))
    with pytest.raises(ValueError, match="decay_exponent must be >= 0"):
        threshold_decay(1.0, 1.0, -1.0)
    assert pytest.approx(apply_serotonin_threshold_offset(1.0, 0.5)) == 1.5
    assert pytest.approx(apply_refractory_boost(1.0, 1, 0.5)) == 1.5


def test_updates():
    # 1.0 + 0.1*(1.0 - 0.5) + 0.5*1 = 1.0 + 0.05 + 0.5 = 1.55
    assert (
        pytest.approx(update_threshold_discrete(1.0, 1.0, 0.5, eta=0.1, delta=0.5, B_prev=1))
        == 1.55
    )
    # Deprecated
    # 0.1*(1.0-2.0) + 0.5*1 - 0.2*|0.5| = -0.1 + 0.5 - 0.1 = 0.3
    assert pytest.approx(update_threshold_ode_deprecated(2.0, 1.0, 0.5, 1, 0.1, 0.5, 0.2)) == 0.3
