"""Tests for core/config_schema.py - Pydantic configuration validation."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from core.config_schema import APGIConfig, create_production_config


class TestAPGIConfigAcceptsRuntimeConfig:
    """The shipped runtime CONFIG dict (config.py) must validate cleanly
    through this schema — it's an optional validator layered on top of the
    actual runtime config, not a competing source of truth, so it must never
    reject values the pipeline itself ships and uses by default."""

    def test_from_dict_accepts_shipped_config(self):
        from config import CONFIG

        config = APGIConfig.from_dict(CONFIG)
        assert config.kuramoto_reset_amount is None


class TestAPGIConfigDefaults:
    """Tests for APGIConfig default values."""

    def test_default_initial_states(self):
        """Should have correct default initial states."""
        config = APGIConfig()
        assert config.S0 == 0.0
        assert config.theta_0 == 1.0
        assert config.theta_base == 1.0
        assert config.sigma2_e0 == 1.0
        assert config.sigma2_i0 == 1.0

    def test_default_numerical_stability(self):
        """Should have correct default numerical stability params."""
        config = APGIConfig()
        assert config.eps == 1e-8
        assert config.pi_min == 0.01
        assert config.pi_max == 10.0

    def test_default_ema_rates(self):
        """Should have correct default EMA rates."""
        config = APGIConfig()
        assert config.alpha_e == 0.05
        assert config.alpha_i == 0.05

    def test_default_variance_method(self):
        """Should default to ema method."""
        config = APGIConfig()
        assert config.variance_method == "ema"
        assert config.T_win == 50

    def test_default_neuromodulation(self):
        """Should have correct default neuromodulation params."""
        config = APGIConfig()
        assert config.g_ach == 1.0
        assert config.g_ne == 1.0

    def test_default_dopaminergic_bias(self):
        """Should have correct default dopaminergic bias."""
        config = APGIConfig()
        assert config.beta == 1.15
        assert config.beta_da is None

    def test_default_ne_settings(self):
        """Should have correct default NE settings."""
        config = APGIConfig()
        assert config.ne_on_precision is True
        assert config.ne_on_threshold is True  # SSOT §11: both pathways (SSOT-P13)
        assert config.gamma_ne == 0.0  # R6: no default sign
        assert not hasattr(config, "kappa")  # bare κ is reserved for ATP/bit

    def test_default_signal_accumulation(self):
        """Should have correct default signal accumulation."""
        config = APGIConfig()
        assert config.lam == pytest.approx(1 - math.exp(-config.dt / config.tau_s))
        assert config.signal_log_nonlinearity is True
        assert config.use_canonical_discrete_mode is False

    def test_default_threshold_dynamics(self):
        """Should have correct default threshold dynamics."""
        config = APGIConfig()
        assert config.eta == 0.002  # per step: 0.2 s⁻¹ × dt
        assert config.delta == 0.5
        assert config.reset_factor == 0.1

    def test_default_timescales(self):
        """Should have correct default timescales."""
        config = APGIConfig()
        assert config.tau_theta == 20.0
        assert config.tau_theta_recovery == 0.45
        assert config.tau_s == 0.3
        assert config.dt == 0.01
        assert config.r_max == 10.0  # OP-V12 design value (B-13)

    def test_default_hierarchical(self):
        """Should have correct default hierarchical settings."""
        config = APGIConfig()
        assert config.KAPPA_UP == 0.1
        assert config.KAPPA_DOWN == 0.1
        assert config.use_hierarchical is False
        assert config.n_levels == 3

    def test_default_cost_value(self):
        """Should have correct default cost-value model."""
        config = APGIConfig()
        assert config.use_realistic_cost is True
        assert config.c0 == 0.0
        assert config.c1 == 0.2
        assert config.c2 == 0.5
        assert config.v1 == 0.5
        assert config.v2 == 0.5

    def test_default_ignition(self):
        """Should have correct default ignition settings."""
        config = APGIConfig()
        assert config.ignite_tau == 1.0 / 7.5  # τ_σ = 1/γ_sig at OP-V12 (B-13)
        assert config.tau_sigma is None
        assert config.stochastic_ignition is False

    def test_default_noise(self):
        """Should have correct default noise settings."""
        config = APGIConfig()
        assert config.noise_std == 0.01

    def test_default_predictions(self):
        """Should have correct default prediction settings."""
        config = APGIConfig()
        assert config.use_internal_predictions is True
        assert config.eta_pred_e == 0.01
        assert config.eta_pred_i == 0.01

    def test_default_multiscale(self):
        """Should have correct default multiscale settings."""
        config = APGIConfig()
        assert config.timescale_k == 1.6

    def test_default_thermodynamic(self):
        """Should have correct default thermodynamic settings."""
        config = APGIConfig()
        assert config.use_thermodynamic_cost is False
        assert config.k_boltzmann == 1.380649e-23
        assert config.T_env == 310.0
        assert config.r_l == 1e20
        assert config.r_l_units == "dimensionless"

    def test_default_bold(self):
        """Should have correct default BOLD settings."""
        config = APGIConfig()
        assert config.use_bold_calibration is False
        assert config.bold_conversion_factor == 1.2e-18
        assert config.bold_tissue_volume == 1.0
        assert config.bold_ignition_spike_factor == 1.075

    def test_default_reservoir(self):
        """Should have correct default reservoir settings."""
        config = APGIConfig()
        assert config.use_reservoir is False
        assert config.reservoir_size == 100
        assert config.reservoir_tau == 1.0
        assert config.reservoir_spectral_radius == 0.9
        assert config.reservoir_input_scale == 0.1
        assert config.reservoir_readout_method == "linear"
        assert config.reservoir_amplification == 0.0

    def test_default_kuramoto(self):
        """Should have correct default Kuramoto settings."""
        config = APGIConfig()
        assert config.use_kuramoto is False
        assert config.kuramoto_tau_xi == 1.0
        assert config.kuramoto_sigma_xi == 0.1
        assert config.kuramoto_reset_amount is None


class TestAPGIConfigValidation:
    """Tests for APGIConfig validation rules."""

    def test_valid_config_passes(self):
        """Should accept valid configuration."""
        config = APGIConfig(theta_0=2.0, beta=1.2)
        assert config.theta_0 == 2.0
        assert config.beta == 1.2

    def test_theta_0_must_be_positive(self):
        """Should reject non-positive theta_0."""
        with pytest.raises(ValidationError):
            APGIConfig(theta_0=0)
        with pytest.raises(ValidationError):
            APGIConfig(theta_0=-1)

    def test_theta_base_must_be_positive(self):
        """Should reject non-positive theta_base."""
        with pytest.raises(ValidationError):
            APGIConfig(theta_base=0)

    def test_sigma2_e0_must_be_positive(self):
        """Should reject non-positive sigma2_e0."""
        with pytest.raises(ValidationError):
            APGIConfig(sigma2_e0=0)

    def test_sigma2_i0_must_be_positive(self):
        """Should reject non-positive sigma2_i0."""
        with pytest.raises(ValidationError):
            APGIConfig(sigma2_i0=0)

    def test_eps_range(self):
        """Should enforce eps > 0 and < 1."""
        with pytest.raises(ValidationError):
            APGIConfig(eps=0)
        with pytest.raises(ValidationError):
            APGIConfig(eps=1.0)
        with pytest.raises(ValidationError):
            APGIConfig(eps=-1e-9)

    def test_pi_min_must_be_positive(self):
        """Should reject non-positive pi_min."""
        with pytest.raises(ValidationError):
            APGIConfig(pi_min=0)

    def test_pi_max_must_be_positive(self):
        """Should reject non-positive pi_max."""
        with pytest.raises(ValidationError):
            APGIConfig(pi_max=0)

    def test_pi_max_must_be_greater_than_pi_min(self):
        """Should reject pi_max <= pi_min."""
        with pytest.raises(ValidationError):
            APGIConfig(pi_min=10, pi_max=5)
        with pytest.raises(ValidationError):
            APGIConfig(pi_min=10, pi_max=10)

    def test_alpha_e_range(self):
        """Should enforce 0 < alpha_e < 1."""
        with pytest.raises(ValidationError):
            APGIConfig(alpha_e=0)
        with pytest.raises(ValidationError):
            APGIConfig(alpha_e=1.0)
        with pytest.raises(ValidationError):
            APGIConfig(alpha_e=-0.1)

    def test_alpha_i_range(self):
        """Should enforce 0 < alpha_i < 1."""
        with pytest.raises(ValidationError):
            APGIConfig(alpha_i=0)
        with pytest.raises(ValidationError):
            APGIConfig(alpha_i=1.0)

    def test_variance_method_enum(self):
        """Should only accept valid variance methods."""
        APGIConfig(variance_method="ema")
        APGIConfig(variance_method="sliding_window")
        with pytest.raises(ValidationError):
            APGIConfig(variance_method="invalid")

    def test_T_win_must_be_positive(self):
        """Should reject non-positive T_win."""
        with pytest.raises(ValidationError):
            APGIConfig(T_win=0)

    def test_g_ach_must_be_non_negative(self):
        """Should reject negative g_ach."""
        with pytest.raises(ValidationError):
            APGIConfig(g_ach=-0.1)
        # Zero should be allowed
        config = APGIConfig(g_ach=0)
        assert config.g_ach == 0

    def test_g_ne_must_be_non_negative(self):
        """Should reject negative g_ne."""
        with pytest.raises(ValidationError):
            APGIConfig(g_ne=-0.1)

    def test_beta_must_be_non_negative(self):
        """Should reject negative beta."""
        with pytest.raises(ValidationError):
            APGIConfig(beta=-0.1)

    def test_gamma_ne_range(self):
        """γ_NE is signed with no default sign (R6); its bound is the pipeline clamp
        |γ_NE|·g_NE ≤ 0.5 (EQ-NE-1), not a schema range."""
        for v in (-0.1, 0.0, 1.0, 1.1):
            assert APGIConfig(gamma_ne=v).gamma_ne == v

    def test_kappa_field_retired(self):
        """The stored per-step exponent kappa is gone: τ_θ (validated > 0) replaces it."""
        assert "kappa" not in APGIConfig.model_fields

    def test_lam_range(self):
        """Should enforce 0 < lam < 1."""
        with pytest.raises(ValidationError):
            APGIConfig(lam=0)
        with pytest.raises(ValidationError):
            APGIConfig(lam=1.0)

    def test_eta_range(self):
        """Should enforce 0 < eta <= 1."""
        with pytest.raises(ValidationError):
            APGIConfig(eta=0)
        with pytest.raises(ValidationError):
            APGIConfig(eta=1.1)
        APGIConfig(eta=1.0)

    def test_delta_must_be_non_negative(self):
        """Should reject negative delta."""
        with pytest.raises(ValidationError):
            APGIConfig(delta=-0.1)

    def test_reset_factor_range(self):
        """Should enforce 0 < reset_factor < 1."""
        with pytest.raises(ValidationError):
            APGIConfig(reset_factor=0)
        with pytest.raises(ValidationError):
            APGIConfig(reset_factor=1.0)

    def test_tau_theta_must_be_positive(self):
        """Should reject non-positive tau_theta."""
        with pytest.raises(ValidationError):
            APGIConfig(tau_theta=0)

    def test_tau_theta_recovery_must_be_positive(self):
        """Should reject non-positive tau_theta_recovery."""
        with pytest.raises(ValidationError):
            APGIConfig(tau_theta_recovery=0)

    def test_KAPPA_UP_range(self):
        """Should enforce 0 <= KAPPA_UP <= 1."""
        with pytest.raises(ValidationError):
            APGIConfig(KAPPA_UP=-0.1)
        with pytest.raises(ValidationError):
            APGIConfig(KAPPA_UP=1.1)

    def test_KAPPA_DOWN_range(self):
        """Should enforce 0 <= KAPPA_DOWN <= 1."""
        with pytest.raises(ValidationError):
            APGIConfig(KAPPA_DOWN=-0.1)
        with pytest.raises(ValidationError):
            APGIConfig(KAPPA_DOWN=1.1)

    def test_cost_coefficients_non_negative(self):
        """Should reject negative cost coefficients."""
        with pytest.raises(ValidationError):
            APGIConfig(c0=-0.1)
        with pytest.raises(ValidationError):
            APGIConfig(c1=-0.1)
        with pytest.raises(ValidationError):
            APGIConfig(c2=-0.1)

    def test_value_weights_non_negative(self):
        """Should reject negative value weights."""
        with pytest.raises(ValidationError):
            APGIConfig(v1=-0.1)
        with pytest.raises(ValidationError):
            APGIConfig(v2=-0.1)

    def test_ignite_tau_must_be_positive(self):
        """Should reject non-positive ignite_tau."""
        with pytest.raises(ValidationError):
            APGIConfig(ignite_tau=0)

    def test_tau_s_must_be_positive(self):
        """Should reject non-positive tau_s."""
        with pytest.raises(ValidationError):
            APGIConfig(tau_s=0)

    def test_dt_must_be_positive(self):
        """Should reject non-positive dt."""
        with pytest.raises(ValidationError):
            APGIConfig(dt=0)

    def test_noise_std_must_be_non_negative(self):
        """Should reject negative noise_std."""
        with pytest.raises(ValidationError):
            APGIConfig(noise_std=-0.1)

    def test_kappa_e_must_be_non_negative(self):
        """Should reject negative eta_pred_e."""
        with pytest.raises(ValidationError):
            APGIConfig(eta_pred_e=-0.1)

    def test_kappa_i_must_be_non_negative(self):
        """Should reject negative eta_pred_i."""
        with pytest.raises(ValidationError):
            APGIConfig(eta_pred_i=-0.1)

    def test_timescale_k_range(self):
        """Should enforce 1 < timescale_k <= 1000 (spec canonical inter-level ratio is ~133)."""
        with pytest.raises(ValidationError):
            APGIConfig(timescale_k=1.0)
        with pytest.raises(ValidationError):
            APGIConfig(timescale_k=0.5)
        with pytest.raises(ValidationError):
            APGIConfig(timescale_k=1001.0)
        APGIConfig(timescale_k=1.5)
        # Canonical Notation Appendix inter-level ratio must be configurable
        # (previously capped at <3, making the spec value impossible to set).
        APGIConfig(timescale_k=133.0)
        APGIConfig(timescale_k=2.5)

    def test_k_boltzmann_must_be_positive(self):
        """Should reject non-positive k_boltzmann."""
        with pytest.raises(ValidationError):
            APGIConfig(k_boltzmann=0)

    def test_T_env_must_be_positive(self):
        """Should reject non-positive T_env."""
        with pytest.raises(ValidationError):
            APGIConfig(T_env=0)

    def test_r_l_must_be_positive(self):
        """Should reject non-positive r_l (formerly kappa_meta)."""
        with pytest.raises(ValidationError):
            APGIConfig(r_l=0)

    def test_r_l_units_enum(self):
        """Should only accept valid r_l_units (formerly kappa_units)."""
        APGIConfig(r_l_units="dimensionless")
        APGIConfig(r_l_units="joules_per_bit")
        with pytest.raises(ValidationError):
            APGIConfig(r_l_units="invalid")

    def test_bold_conversion_factor_must_be_positive(self):
        """Should reject non-positive bold_conversion_factor."""
        with pytest.raises(ValidationError):
            APGIConfig(bold_conversion_factor=0)

    def test_bold_tissue_volume_must_be_positive(self):
        """Should reject non-positive bold_tissue_volume."""
        with pytest.raises(ValidationError):
            APGIConfig(bold_tissue_volume=0)

    def test_bold_ignition_spike_factor_must_be_positive(self):
        """Should reject non-positive bold_ignition_spike_factor."""
        with pytest.raises(ValidationError):
            APGIConfig(bold_ignition_spike_factor=0)

    def test_reservoir_size_must_be_positive(self):
        """Should reject non-positive reservoir_size."""
        with pytest.raises(ValidationError):
            APGIConfig(reservoir_size=0)

    def test_reservoir_tau_must_be_positive(self):
        """Should reject non-positive reservoir_tau."""
        with pytest.raises(ValidationError):
            APGIConfig(reservoir_tau=0)

    def test_reservoir_spectral_radius_range(self):
        """Should enforce 0 < reservoir_spectral_radius < 1."""
        with pytest.raises(ValidationError):
            APGIConfig(reservoir_spectral_radius=0)
        with pytest.raises(ValidationError):
            APGIConfig(reservoir_spectral_radius=1.0)

    def test_reservoir_input_scale_must_be_positive(self):
        """Should reject non-positive reservoir_input_scale."""
        with pytest.raises(ValidationError):
            APGIConfig(reservoir_input_scale=0)

    def test_reservoir_readout_method_enum(self):
        """Should only accept valid readout methods."""
        APGIConfig(reservoir_readout_method="linear")
        APGIConfig(reservoir_readout_method="energy")
        with pytest.raises(ValidationError):
            APGIConfig(reservoir_readout_method="invalid")

    def test_reservoir_amplification_must_be_non_negative(self):
        """Should reject negative amplification."""
        with pytest.raises(ValidationError):
            APGIConfig(reservoir_amplification=-0.1)

    def test_kuramoto_tau_xi_must_be_positive(self):
        """Should reject non-positive tau_xi."""
        with pytest.raises(ValidationError):
            APGIConfig(kuramoto_tau_xi=0)

    def test_kuramoto_sigma_xi_must_be_non_negative(self):
        """Should reject negative sigma_xi."""
        with pytest.raises(ValidationError):
            APGIConfig(kuramoto_sigma_xi=-0.1)

    def test_kuramoto_reset_amount_accepts_none_for_canonical_reset(self):
        """None (the default) selects the canonical zero-phase reset and must
        validate — it must not be rejected as if it were the legacy
        positive-float-only field."""
        config = APGIConfig(kuramoto_reset_amount=None)
        assert config.kuramoto_reset_amount is None

    def test_kuramoto_reset_amount_accepts_legacy_float(self):
        """An explicit float still selects the legacy additive reset."""
        config = APGIConfig(kuramoto_reset_amount=3.14159)
        assert config.kuramoto_reset_amount == 3.14159

    def test_n_levels_range(self):
        """Should enforce 1 <= n_levels <= 10."""
        with pytest.raises(ValidationError):
            APGIConfig(n_levels=0)
        with pytest.raises(ValidationError):
            APGIConfig(n_levels=11)
        APGIConfig(n_levels=1)
        APGIConfig(n_levels=10)


class TestAPGIConfigModelValidators:
    """Tests for APGIConfig model-level validators."""

    def test_both_ne_pathways_are_accepted(self):
        """SSOT §11 / EQ-THEFF-1 (SSOT-P13): both NE pathways share one clamped g_NE,
        so the schema accepts them together, and the shipped default enables both."""
        cfg = APGIConfig(ne_on_precision=True, ne_on_threshold=True, gamma_ne=0.01, kappa=0.15)
        assert cfg.ne_on_precision and cfg.ne_on_threshold
        assert APGIConfig().gamma_ne == 0.0  # R6: no default sign

    @pytest.mark.parametrize("gamma_ne", [0.2, -0.3, 1.0])
    def test_gamma_ne_is_signed_and_not_rejected(self, gamma_ne):
        """γ_NE is fitted with no default sign (R6); its only bound is the clamp
        |γ_NE|·g_NE ≤ 0.5 (EQ-NE-1), which the pipeline applies by rescaling g_NE."""
        cfg = APGIConfig(ne_on_precision=True, ne_on_threshold=True, gamma_ne=gamma_ne, kappa=0.15)
        assert cfg.gamma_ne == gamma_ne

    def test_dt_stability_validation(self):
        """Should validate dt against minimum timescale."""
        # Valid: dt is small enough
        APGIConfig(tau_s=5.0, tau_theta=20.0, dt=0.1)
        # Invalid: dt too large
        with pytest.raises(ValidationError) as exc_info:
            APGIConfig(tau_s=1.0, tau_theta=1.0, dt=0.2)
        assert "exceeds max" in str(exc_info.value)

    def test_learning_rates_validation(self):
        """Should validate eta_pred_e and eta_pred_i against pi_max."""
        # Valid: kappa < 2/pi_max
        APGIConfig(use_internal_predictions=True, pi_max=100.0, eta_pred_e=0.01)
        # Invalid: eta_pred_e too large
        with pytest.raises(ValidationError) as exc_info:
            APGIConfig(use_internal_predictions=True, pi_max=1.0, eta_pred_e=5.0)
        assert "eta_pred_e" in str(exc_info.value)
        # Invalid: eta_pred_i too large
        with pytest.raises(ValidationError) as exc_info:
            APGIConfig(use_internal_predictions=True, pi_max=1.0, eta_pred_i=5.0)
        assert "eta_pred_i" in str(exc_info.value)

    def test_somatic_bias_consistency_beta_da_precedence(self):
        """Should use beta_da when both set with different values."""
        config = APGIConfig(beta=1.15, beta_da=1.25)
        # beta_da takes precedence
        assert config.beta == 1.25

    def test_somatic_bias_consistency_same_values(self):
        """Should handle identical beta and beta_da."""
        config = APGIConfig(beta=1.2, beta_da=1.2)
        assert config.beta == 1.2

    def test_backward_compat_beta_da(self):
        """Should apply beta_da when beta was default."""
        config = APGIConfig(beta_da=1.25)  # beta defaults to 1.15
        assert config.beta == 1.25

    def test_backward_compat_tau_sigma(self):
        """Should apply tau_sigma when ignite_tau was default."""
        config = APGIConfig(tau_sigma=0.7)  # ignite_tau defaults to 1/7.5
        assert config.ignite_tau == 0.7

    def test_backward_compat_tau_sigma_identical(self):
        """Should skip applying tau_sigma when it's already identical and not default."""
        config = APGIConfig(ignite_tau=0.8, tau_sigma=0.805)
        assert config.ignite_tau == 0.8

    def test_learning_rates_validation_disabled(self):
        """Should skip validation when use_internal_predictions is False."""
        config = APGIConfig(
            use_internal_predictions=False, pi_max=1.0, eta_pred_e=5.0, eta_pred_i=5.0
        )
        assert config.eta_pred_e == 5.0
        assert config.eta_pred_i == 5.0


class TestAPGIConfigConversion:
    """Tests for APGIConfig conversion methods."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        config = APGIConfig(theta_0=2.0, beta=1.2)
        d = config.to_dict()
        assert d["theta_0"] == 2.0
        assert d["beta"] == 1.2
        assert "lam" in d

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {"theta_0": 2.0, "beta": 1.2}
        config = APGIConfig.from_dict(data)
        assert config.theta_0 == 2.0
        assert config.beta == 1.2


class TestCreateProductionConfig:
    """Tests for create_production_config function."""

    def test_strict_mode_returns_apgiconfig(self):
        """Should return APGIConfig in strict mode."""
        config = create_production_config(strict=True)
        assert isinstance(config, APGIConfig)

    def test_non_strict_mode_returns_dict(self):
        """Should return dict in non-strict mode."""
        config = create_production_config(strict=False)
        assert isinstance(config, dict)
        assert "theta_0" in config

    def test_with_overrides(self):
        """Should apply overrides."""
        config = create_production_config(strict=True, theta_0=5.0, beta=1.3)
        assert config.theta_0 == 5.0
        assert config.beta == 1.3

    def test_validation_errors_propagate(self):
        """Should raise ValidationError for invalid overrides."""
        with pytest.raises(ValidationError):
            create_production_config(theta_0=0)
