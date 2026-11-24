"""Comprehensive unit tests for core/threshold.py module.
Tests cover:
- compute_metabolic_cost function
- compute_metabolic_cost_realistic function
- compute_information_value function
- apply_ne_threshold_modulation function
- threshold_decay function
- update_threshold_discrete function
- apply_refractory_boost function
"""

from __future__ import annotations

import numpy as np
import pytest

from core.threshold import (
    apply_ne_threshold_modulation,
    apply_refractory_boost,
    compute_information_value,
    compute_information_value_with_bias,
    compute_metabolic_cost,
    compute_metabolic_cost_realistic,
    threshold_decay,
    update_threshold_discrete,
    update_threshold_ode_deprecated,
)


class TestComputeMetabolicCost:
    """Tests for compute_metabolic_cost function."""

    def test_linear_cost(self):
        """C(t) = c1·S(t) + c2·B(t-1): with B_prev=1, c2=0.5, c1=0.5, S=2.0 → 1.5."""
        result = compute_metabolic_cost(S=2.0, B_prev=1, c1=0.5, c2=0.5)
        assert result == 1.5

    def test_zero_c2(self):
        """Without prior ignition (B_prev=0), cost is c1·S only."""
        result = compute_metabolic_cost(S=2.0, B_prev=0, c1=0.5, c2=0.5)
        assert result == 1.0

    def test_zero_signal(self):
        """With zero signal and prior ignition, cost = c2·B_prev."""
        result = compute_metabolic_cost(S=0.0, B_prev=1, c1=0.5, c2=0.5)
        assert result == 0.5


class TestComputeMetabolicCostRealistic:
    """Tests for compute_metabolic_cost_realistic function."""

    def test_realistic_cost(self):
        """Should compute realistic cost C(t) = c1*S(t) + c2*B(t-1)."""
        result = compute_metabolic_cost_realistic(S=2.0, B_prev=1, c1=0.5, c2=1.0)
        assert result == 2.0  # 0.5*2.0 + 1.0*1 = 2.0

    def test_no_previous_ignition(self):
        """Should exclude ignition cost when B_prev=0."""
        result = compute_metabolic_cost_realistic(S=2.0, B_prev=0, c1=0.5, c2=1.0)
        assert result == 1.0  # 0.5*2.0 + 0 = 1.0

    def test_previous_ignition(self):
        """Should add ignition cost when B_prev=1."""
        result = compute_metabolic_cost_realistic(S=2.0, B_prev=1, c1=0.5, c2=1.0)
        assert result == 2.0  # 0.5*2.0 + 1.0 = 2.0


class TestComputeInformationValue:
    """Tests for compute_information_value function."""

    def test_basic_value(self):
        """V(t) = v1|z_e| + v2|z_i_eff| uses raw prediction errors per spec."""
        result = compute_information_value(
            z_e=1.0,
            z_i_eff=0.5,
            v1=0.5,
            v2=0.5,
        )
        assert result == 0.75  # 0.5*1.0 + 0.5*0.5 = 0.75

    def test_negative_errors(self):
        """Should use absolute values of raw errors."""
        result = compute_information_value(
            z_e=-1.0,
            z_i_eff=-0.5,
            v1=0.5,
            v2=0.5,
        )
        assert result == 0.75

    def test_zero_weights(self):
        """Should return zero when weights are zero."""
        result = compute_information_value(1.0, 0.5, v1=0.0, v2=0.0)
        assert result == 0.0

    def test_different_v1_v2(self):
        """Should use different weights correctly."""
        result = compute_information_value(
            z_e=1.0,
            z_i_eff=0.5,
            v1=0.3,
            v2=0.7,
        )
        assert result == 0.3 * 1.0 + 0.7 * 0.5  # 0.3 + 0.35 = 0.65


class TestApplyNeThresholdModulation:
    """Tests for apply_ne_threshold_modulation function."""

    def test_ne_modulation(self):
        """Should apply NE modulation θ <- θ * (1 + γ_NE * g_NE)."""
        result = apply_ne_threshold_modulation(theta=1.0, g_ne=0.5, gamma_ne=0.1)
        # θ = 1.0 * (1 + 0.1 * 0.5) = 1.0 * 1.05 = 1.05
        assert result == 1.05

    def test_zero_gain(self):
        """Should return original theta when g_NE=0."""
        result = apply_ne_threshold_modulation(theta=1.0, g_ne=0.0, gamma_ne=0.1)
        assert result == 1.0

    def test_increase_threshold(self):
        """Should increase threshold for positive gain."""
        result = apply_ne_threshold_modulation(theta=1.0, g_ne=1.0, gamma_ne=0.1)
        assert result > 1.0

    def test_decrease_threshold(self):
        """Should decrease threshold for negative gain."""
        result = apply_ne_threshold_modulation(theta=1.0, g_ne=-1.0, gamma_ne=0.1)
        assert result < 1.0


class TestThresholdDecay:
    """Tests for threshold_decay function."""

    def test_exponential_decay(self):
        """Should compute exponential decay correctly."""
        result = threshold_decay(theta=2.0, theta_base=1.0, kappa=0.5)
        expected = 1.0 + (2.0 - 1.0) * np.exp(-0.5)
        assert pytest.approx(result, rel=1e-7) == expected

    def test_already_at_base(self):
        """Should return theta_base when theta equals base."""
        result = threshold_decay(theta=1.0, theta_base=1.0, kappa=0.5)
        assert result == 1.0

    def test_high_kappa_fast_decay(self):
        """Should decay faster with higher kappa."""
        result_fast = threshold_decay(theta=2.0, theta_base=1.0, kappa=2.0)
        result_slow = threshold_decay(theta=2.0, theta_base=1.0, kappa=0.1)
        assert result_fast < result_slow

    def test_zero_kappa(self):
        """Should return original theta when kappa=0."""
        result = threshold_decay(theta=2.0, theta_base=1.0, kappa=0.0)
        assert result == 2.0

    def test_invalid_kappa(self):
        """Should raise ValueError for negative kappa."""
        with pytest.raises(ValueError, match="decay_exponent must be >= 0"):
            threshold_decay(theta=2.0, theta_base=1.0, kappa=-0.1)


class TestUpdateThresholdDiscrete:
    """Tests for update_threshold_discrete function."""

    def test_basic_update(self):
        """Should compute threshold update correctly."""
        result = update_threshold_discrete(
            theta=1.0,
            metabolic_cost=1.5,
            information_value=1.0,
            eta=0.1,
            delta=0.5,
            B_prev=0,
        )
        # θ = 1.0 + 0.1*(1.5-1.0) + 0.5*0 = 1.0 + 0.05 = 1.05
        assert result == 1.05

    def test_with_ignition(self):
        """Should add refractory boost after ignition."""
        result = update_threshold_discrete(
            theta=1.0,
            metabolic_cost=1.5,
            information_value=1.0,
            eta=0.1,
            delta=0.5,
            B_prev=1,
        )
        # θ = 1.0 + 0.1*(1.5-1.0) + 0.5*1 = 1.0 + 0.05 + 0.5 = 1.55
        assert result == 1.55

    def test_cost_greater_than_value(self):
        """Should increase threshold when cost > value."""
        result = update_threshold_discrete(
            theta=1.0,
            metabolic_cost=2.0,
            information_value=1.0,
            eta=0.1,
            delta=0.5,
            B_prev=0,
        )
        assert result > 1.0

    def test_value_greater_than_cost(self):
        """Should decrease threshold when value > cost."""
        result = update_threshold_discrete(
            theta=1.0,
            metabolic_cost=1.0,
            information_value=2.0,
            eta=0.1,
            delta=0.5,
            B_prev=0,
        )
        assert result < 1.0


class TestApplyRefractoryBoost:
    """Tests for apply_refractory_boost function."""

    def test_refractory_boost(self):
        """Should add refractory boost after ignition."""
        result = apply_refractory_boost(theta_next=1.0, B=1, delta=0.5)
        assert result == 1.5

    def test_no_boost_without_ignition(self):
        """Should not add boost when no ignition."""
        result = apply_refractory_boost(theta_next=1.0, B=0, delta=0.5)
        assert result == 1.0

    def test_boost_with_integer_B(self):
        """Should handle B as integer."""
        result = apply_refractory_boost(theta_next=1.0, B=1, delta=0.5)
        assert result == 1.5

    def test_zero_delta(self):
        """Should not change theta when delta=0."""
        result = apply_refractory_boost(theta_next=1.0, B=1, delta=0.0)
        assert result == 1.0


class TestComputeMetabolicCostRealisticExtended:
    """Extended tests for Landauer enforcement."""

    def test_enforce_landauer_threshold_low_signal(self):
        """Should not apply Landauer constraint when S <= eps_stab."""
        result = compute_metabolic_cost_realistic(
            S=0.001,
            B_prev=0,
            c1=1.0,
            c2=1.0,
            eps_stab=0.01,
            enforce_landauer=True,
        )
        # When S <= eps, no Landauer constraint is applied
        assert result == 0.001  # Just c1 * S

    def test_enforce_landauer_active(self):
        """Should enforce Landauer minimum when signal is large enough."""
        result = compute_metabolic_cost_realistic(
            S=1.0,
            B_prev=0,
            c1=0.01,  # Very small cost coefficient
            c2=1.0,
            eps_stab=0.001,
            enforce_landauer=True,
            kappa_meta=1.0,
        )
        # Landauer cost should dominate the small base cost
        assert result >= 0.01

    def test_kappa_meta_parameter(self):
        """Should use kappa_meta to scale Landauer cost (κ_meta absorbs unit conversion)."""
        # Use kappa_meta values large enough for the Landauer term to dominate
        # base_cost (c1*S ≈ 1.0).  With the scale_factor removed, κ_meta carries
        # the J→AU unit conversion; canonical default is 1e20.
        result1 = compute_metabolic_cost_realistic(
            S=1.0,
            B_prev=0,
            enforce_landauer=True,
            kappa_meta=1e20,
        )
        result2 = compute_metabolic_cost_realistic(
            S=1.0,
            B_prev=0,
            enforce_landauer=True,
            kappa_meta=2e20,
        )
        # Higher kappa_meta should increase Landauer cost
        assert result2 > result1


class TestComputeInformationValueWithBias:
    """Tests for compute_information_value_with_bias function."""

    def test_basic_value_with_bias(self):
        """Alias for compute_information_value; uses raw errors per spec."""
        result = compute_information_value_with_bias(
            z_e=1.0,
            z_i_eff=0.5,
            v1=0.5,
            v2=0.5,
        )
        assert result == 0.75

    def test_negative_bias(self):
        """Should handle negative z_i_eff values."""
        result = compute_information_value_with_bias(
            z_e=1.0,
            z_i_eff=0.3,
            v1=0.5,
            v2=0.5,
        )
        assert result == 0.65

    def test_zero_bias(self):
        """Should equal regular compute_information_value (identical function)."""
        z_e = 1.0
        z_i_eff = 0.5
        v1 = 0.5
        v2 = 0.5
        result_with_bias = compute_information_value_with_bias(
            z_e=z_e, z_i_eff=z_i_eff, v1=v1, v2=v2
        )
        result_without_bias = compute_information_value(z_e=z_e, z_i_eff=z_i_eff, v1=v1, v2=v2)
        assert result_with_bias == result_without_bias


class TestUpdateThresholdOdeDeprecated:
    """Tests for deprecated ODE threshold update function."""

    def test_deprecated_ode_function(self):
        """Should still work even though deprecated."""
        result = update_threshold_ode_deprecated(
            theta=1.0,
            theta_0=0.8,
            dS_dt=0.1,
            B_prev=1,
            gamma=0.1,
            delta=0.5,
            lam=0.2,
        )
        # Compute expected: gamma * (theta_0 - theta) + delta * B_prev - lam * abs(dS_dt)
        # = 0.1 * (0.8 - 1.0) + 0.5 * 1 - 0.2 * 0.1
        # = 0.1 * (-0.2) + 0.5 - 0.02
        # = -0.02 + 0.5 - 0.02 = 0.46
        expected = 0.1 * (0.8 - 1.0) + 0.5 * 1 - 0.2 * 0.1
        assert pytest.approx(result, rel=1e-6) == expected

    def test_deprecated_ode_no_ignition(self):
        """Should work without ignition."""
        result = update_threshold_ode_deprecated(
            theta=1.0,
            theta_0=0.8,
            dS_dt=0.1,
            B_prev=0,  # No ignition
            gamma=0.1,
            delta=0.5,
            lam=0.2,
        )
        # = 0.1 * (0.8 - 1.0) + 0 - 0.02 = -0.04
        expected = -0.04
        assert pytest.approx(result, rel=1e-6) == expected
