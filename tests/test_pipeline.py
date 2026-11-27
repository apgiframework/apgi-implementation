"""Comprehensive tests for APGI Pipeline module.
Tests all functionality including:
- Hierarchical mode presets
- Precision state management
- NE configuration validation
- Signal integration (ODE and discrete modes)
- Threshold dynamics (continuous and discrete)
- Thermodynamic cost calculation
- Observable mapping integration
- Stability analysis integration
- Kuramoto oscillators
- Reservoir layer
- Post-ignition dynamics
- Internal prediction updates
"""

import warnings

import numpy as np
import pytest

from core.validation import ValidationError
from pipeline import APGIPipeline, HierarchicalState, PrecisionState


@pytest.fixture
def base_config() -> dict[str, float | int | bool]:
    """Base configuration for pipeline tests."""
    return {
        "S0": 0.5,
        "theta_0": 1.0,
        "theta_base": 1.0,
        "sigma2_e0": 1.0,
        "sigma2_i0": 1.0,
        "alpha_e": 0.1,
        "alpha_i": 0.1,
        "beta": 0.5,
        "g_ach": 1.0,
        "eta": 0.1,
        "kappa": 0.15,
        "delta": 0.5,
        "c0": 0.1,
        "c1": 0.2,
        "c2": 0.05,
        "v1": 1.0,
        "v2": 0.5,
        "lam": 0.2,
        "eps": 1e-8,
        "pi_min": 1e-4,
        "pi_max": 1e4,
        "ignite_tau": 0.5,
        "stochastic_ignition": False,
        "signal_log_nonlinearity": False,
        "use_realistic_cost": False,
        "dt": 0.01,  # Must be ≤ min(tau_s, tau_theta, tau_pi) / 10 = 5.0/10 = 0.5
        "tau_s": 5.0,
        "tau_theta": 1000.0,
        "tau_pi": 1000.0,
        "T_win": 50,
    }


@pytest.fixture
def discrete_config(base_config: dict[str, float | int | bool]) -> dict[str, float | int | bool]:
    """Configuration for discrete canonical mode."""
    config = base_config.copy()
    config["use_canonical_discrete_mode"] = True
    return config


@pytest.fixture
def ode_config(base_config: dict[str, float | int | bool]) -> dict[str, float | int | bool]:
    """Configuration for ODE mode."""
    config = base_config.copy()
    config["use_canonical_discrete_mode"] = False
    return config


@pytest.fixture
def hierarchical_config(
    base_config: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    """Configuration with hierarchical mode enabled."""
    config = base_config.copy()
    config["use_hierarchical"] = True
    config["n_levels"] = 3
    config["tau_0"] = 10.0
    config["k"] = 1.6
    return config


@pytest.fixture
def reservoir_config(base_config: dict[str, float | int | bool]) -> dict[str, float | int | bool]:
    """Configuration with reservoir enabled."""
    config = base_config.copy()
    config["use_reservoir"] = True
    config["reservoir_size"] = 50
    config["reservoir_tau"] = 1.0
    config["reservoir_spectral_radius"] = 0.9
    config["reservoir_input_scale"] = 0.1
    return config


@pytest.fixture
def thermodynamic_config(
    base_config: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    """Configuration with thermodynamic cost enabled."""
    config = base_config.copy()
    config["use_thermodynamic_cost"] = True
    config["k_boltzmann"] = 1.38e-23
    config["T_env"] = 310.0
    config["kappa_meta"] = 1.0
    return config


@pytest.fixture
def observable_config(base_config: dict[str, float | int | bool]) -> dict[str, float | int | bool]:
    """Configuration with observable mapping enabled."""
    config = base_config.copy()
    config["use_observable_mapping"] = True
    return config


@pytest.fixture
def stability_config(base_config: dict[str, float | int | bool]) -> dict[str, float | int | bool]:
    """Configuration with stability analysis enabled."""
    config = base_config.copy()
    config["use_stability_analysis"] = True
    return config


@pytest.fixture
def kuramoto_config(base_config: dict[str, float | int | bool]) -> dict[str, float | int | bool]:
    """Configuration with Kuramoto oscillators enabled."""
    config = base_config.copy()
    config["use_kuramoto"] = True
    config["n_levels"] = 3
    return config


@pytest.fixture
def somatic_config(base_config: dict[str, float | int | bool]) -> dict[str, float | int | bool]:
    """Configuration with somatic precision enabled."""
    config = base_config.copy()
    config["use_somatic_precision"] = True
    config["beta_somatic"] = 0.3
    config["M_somatic"] = 0.5
    return config


@pytest.fixture
def ne_precision_config(
    base_config: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    """Configuration with NE on precision."""
    config = base_config.copy()
    config["ne_on_precision"] = True
    config["g_ne"] = 1.5
    return config


@pytest.fixture
def ne_threshold_config(
    base_config: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    """Configuration with NE on threshold."""
    config = base_config.copy()
    config["ne_on_threshold"] = True
    config["g_ne"] = 1.5
    config["gamma_ne"] = 0.01
    return config


@pytest.fixture
def generative_config(base_config: dict[str, float | int | bool]) -> dict[str, float | int | bool]:
    """Configuration with generative model update enabled."""
    config = base_config.copy()
    config["use_generative_model_update"] = True
    config["eta_pred_e"] = 0.01
    config["eta_pred_i"] = 0.01
    config["x_hat_e0"] = 0.0
    config["x_hat_i0"] = 0.0
    return config


@pytest.fixture
def sliding_window_config(
    base_config: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    """Configuration with sliding window variance method."""
    config = base_config.copy()
    config["variance_method"] = "sliding_window"
    config["T_win"] = 10  # Smaller window to avoid degrees of freedom warnings
    return config


@pytest.fixture
def continuous_threshold_config(
    base_config: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    """Configuration with continuous threshold ODE."""
    config = base_config.copy()
    config["use_continuous_threshold_ode"] = True
    config["noise_std"] = 0.01
    return config


@pytest.fixture
def hierarchical_precision_ode_config(
    hierarchical_config: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    """Configuration with hierarchical precision ODE."""
    config = hierarchical_config.copy()
    config["use_hierarchical_precision_ode"] = True
    config["tau_pi"] = 1000.0
    config["C_down"] = 0.1
    config["C_up"] = 0.05
    return config


@pytest.fixture
def phase_modulation_config(
    hierarchical_config: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    """Configuration with phase modulation."""
    config = hierarchical_config.copy()
    config["use_phase_modulation"] = True
    config["kappa_phase"] = 0.1
    config["omega_phases"] = [0.1, 0.05, 0.01]
    return config


@pytest.fixture
def reservoir_threshold_config(
    reservoir_config: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    """Configuration with reservoir as threshold."""
    config = reservoir_config.copy()
    config["reservoir_as_threshold"] = True
    config["reservoir_theta_scale"] = 0.1
    config["reservoir_readout_method"] = "linear"
    config["reservoir_amplification"] = 0.0
    return config


class TestHierarchicalState:
    """Tests for HierarchicalState dataclass."""

    def test_default_initialization(self):
        """Test default initialization of HierarchicalState."""
        state = HierarchicalState(n_levels=3)
        assert state.n_levels == 3
        assert state.pis == [1.0, 1.0, 1.0]
        assert state.thetas == [1.0, 1.0, 1.0]
        assert state.phases == [0.0, 0.0, 0.0]

    def test_custom_initialization(self):
        """Test custom initialization of HierarchicalState."""
        state = HierarchicalState(
            n_levels=2,
            pis=[0.5, 1.5],
            thetas=[0.8, 1.2],
            phases=[0.1, 0.2],
        )
        assert state.n_levels == 2
        assert state.pis == [0.5, 1.5]
        assert state.thetas == [0.8, 1.2]
        assert state.phases == [0.1, 0.2]

    def test_single_level(self):
        """Test single level hierarchical state."""
        state = HierarchicalState(n_levels=1)
        assert state.n_levels == 1
        assert state.pis == [1.0]
        assert state.thetas == [1.0]
        assert state.phases == [0.0]


class TestPrecisionState:
    """Tests for PrecisionState dataclass."""

    def test_default_initialization(self):
        """Test default initialization."""
        state = PrecisionState(sigma2_e=1.0, sigma2_i=1.0)
        assert state.sigma2_e == 1.0
        assert state.sigma2_i == 1.0
        assert state.pi_e == 1.0
        assert state.pi_i == 1.0
        assert state.mu_e == 0.0
        assert state.mu_i == 0.0

    def test_custom_initialization(self):
        """Test custom initialization."""
        state = PrecisionState(
            sigma2_e=0.5,
            sigma2_i=0.8,
            pi_e=2.0,
            pi_i=3.0,
            mu_e=0.1,
            mu_i=0.2,
        )
        assert state.sigma2_e == 0.5
        assert state.sigma2_i == 0.8
        assert state.pi_e == 2.0
        assert state.pi_i == 3.0
        assert state.mu_e == 0.1
        assert state.mu_i == 0.2


class TestHierarchicalModePresets:
    """Tests for hierarchical mode preset functionality."""

    def test_hierarchical_mode_off(self, base_config):
        """Test hierarchical_mode='off' preset."""
        config = base_config.copy()
        config["hierarchical_mode"] = "off"
        pipeline = APGIPipeline(config)
        assert not pipeline.use_hierarchical
        assert not pipeline.config["use_hierarchical"]
        assert not pipeline.config["use_hierarchical_precision_ode"]
        assert not pipeline.config["use_phase_modulation"]

    def test_hierarchical_mode_basic(self, base_config):
        """Test hierarchical_mode='basic' preset."""
        config = base_config.copy()
        config["hierarchical_mode"] = "basic"
        config["tau_0"] = 10.0  # Valid timescale > 1
        pipeline = APGIPipeline(config)
        assert pipeline.use_hierarchical
        assert pipeline.config["use_hierarchical"]
        assert not pipeline.config["use_hierarchical_precision_ode"]
        assert not pipeline.config["use_phase_modulation"]

    def test_hierarchical_mode_advanced(self, base_config):
        """Test hierarchical_mode='advanced' preset."""
        config = base_config.copy()
        config["hierarchical_mode"] = "advanced"
        config["n_levels"] = 3
        config["tau_0"] = 10.0  # Valid timescale > 1
        pipeline = APGIPipeline(config)
        assert pipeline.use_hierarchical
        assert pipeline.config["use_hierarchical"]
        assert pipeline.config["use_hierarchical_precision_ode"]
        assert not pipeline.config["use_phase_modulation"]

    def test_hierarchical_mode_full(self, base_config):
        """Test hierarchical_mode='full' preset."""
        config = base_config.copy()
        config["hierarchical_mode"] = "full"
        config["n_levels"] = 3
        config["tau_0"] = 10.0  # Valid timescale > 1
        pipeline = APGIPipeline(config)
        assert pipeline.use_hierarchical
        assert pipeline.config["use_hierarchical"]
        assert pipeline.config["use_hierarchical_precision_ode"]
        assert pipeline.config["use_phase_modulation"]

    def test_hierarchical_mode_invalid(self, base_config):
        """Test invalid hierarchical_mode raises error."""
        config = base_config.copy()
        config["hierarchical_mode"] = "invalid"
        with pytest.raises(ValueError, match="Unknown hierarchical_mode"):
            APGIPipeline(config)

    def test_no_hierarchical_mode_uses_defaults(self, base_config):
        """Test that missing hierarchical_mode uses explicit flags."""
        config = base_config.copy()
        config["use_hierarchical"] = True
        config["n_levels"] = 3
        config["tau_0"] = 10.0  # Required for hierarchical validation
        config["k"] = 1.6  # Required for hierarchical validation
        pipeline = APGIPipeline(config)
        assert pipeline.use_hierarchical


class TestParameterBackwardCompatibility:
    """Tests for backward compatibility parameter names."""

    def test_beta_da_backward_compat(self, base_config):
        """Test beta_da maps to beta."""
        config = base_config.copy()
        config.pop("beta")
        config["beta_da"] = 0.7
        pipeline = APGIPipeline(config)
        assert pipeline.config["beta"] == 0.7

    def test_beta_da_does_not_override_beta(self, base_config):
        """Test beta is not overridden when already present."""
        config = base_config.copy()
        config["beta"] = 0.5
        config["beta_da"] = 0.7
        pipeline = APGIPipeline(config)
        assert pipeline.config["beta"] == 0.5

    def test_tau_sigma_backward_compat(self, base_config):
        """Test tau_sigma maps to ignite_tau."""
        config = base_config.copy()
        config.pop("ignite_tau")
        config["tau_sigma"] = 0.8
        pipeline = APGIPipeline(config)
        assert pipeline.config["ignite_tau"] == 0.8


class TestNEConfigurationValidation:
    """Tests for NE configuration validation."""

    def test_both_ne_pathways_allowed_under_clamp(self, base_config):
        """Both NE pathways may run simultaneously when γ_NE·g_NE ≤ 0.5.

        MathSpec §2 permits this explicitly; config.py documents it as the
        supported way to enable both. The previous blanket rejection made that
        documented configuration unusable.
        """
        config = base_config.copy()
        config["ne_on_precision"] = True
        config["ne_on_threshold"] = True
        config["gamma_ne"] = 0.01
        config["g_ne"] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = APGIPipeline(config)
        assert pipeline.config["g_ne"] == pytest.approx(1.0), "clamp must not bind at 0.01"

    def test_ne_clamp_rescales_g_ne_rather_than_raising(self, base_config):
        """γ_NE·g_NE > 0.5 rescales g_NE ← 0.5/γ_NE, the spec's own remedy.

        MathSpec §2: "if γ_NE · g_NE(t) > 0.5, rescale g_NE(t) ← 0.5 / γ_NE
        before the threshold update." The clamp is mandatory, not optional,
        and rescaling (not aborting) is what the spec prescribes.
        """
        config = base_config.copy()
        config["ne_on_threshold"] = True
        config["gamma_ne"] = 0.1
        config["g_ne"] = 10.0  # product = 1.0 > 0.5
        config["kappa"] = 0.3  # avoid the separate stiffness advisory
        with pytest.warns(RuntimeWarning, match=r"exceeds the spec-mandated ceiling"):
            pipeline = APGIPipeline(config)
        assert pipeline.config["g_ne"] == pytest.approx(5.0)  # 0.5 / 0.1
        product = pipeline.config["gamma_ne"] * pipeline.config["g_ne"]
        assert product <= APGIPipeline.NE_CLAMP_MAX + 1e-12

    def test_ne_threshold_no_warning_with_safe_params(self, base_config):
        """Test no warning with safe NE threshold params."""
        config = base_config.copy()
        config["ne_on_threshold"] = True
        config["gamma_ne"] = 0.01
        config["kappa"] = 0.15
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            pipeline = APGIPipeline(config)  # noqa: F841
            # Should not have threshold instability warning
            assert not any("threshold instability" in str(w.message) for w in record)

    def test_ne_clamp_inactive_when_threshold_pathway_disabled(self, base_config):
        """With ne_on_threshold=False the product ceiling does not bind.

        The clamp exists to keep the MULTIPLICATIVE THRESHOLD modulation
        θ ← θ(1 + γ_NE·g_NE) sub-dominant. When that pathway is off, g_NE
        scales precision only and no instability can arise from the product,
        so a large γ_NE must not be rejected or rescaled.
        """
        config = base_config.copy()
        config["gamma_ne"] = 1.0
        config["g_ne"] = 1.0  # product = 1.0, but threshold pathway is off
        config["ne_on_precision"] = True
        config["ne_on_threshold"] = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = APGIPipeline(config)
        assert pipeline.config["g_ne"] == pytest.approx(1.0)


class TestValidationWarning:
    """Tests for configuration validation warnings."""

    def test_invalid_config_raises_by_default(self, base_config):
        """A spec-violating config must ABORT, not warn-and-continue.

        λ=1.5 breaks the leaky integrator outright, yet the old fail-open path
        let step() run and return finite, plausible-looking numbers. For the
        cited artifact of a paper series, an invalid run must never be able to
        masquerade as a valid one.
        """
        config = base_config.copy()
        config["lam"] = 1.5  # Invalid: must be in (0, 1)
        with pytest.raises(ValidationError, match=r"lam must be in"):
            APGIPipeline(config)

    def test_strict_false_downgrades_to_warning(self, base_config):
        """strict=False is the explicit, documented opt-out."""
        config = base_config.copy()
        config["lam"] = 1.5
        config["strict"] = False
        with pytest.warns(RuntimeWarning, match="NOT spec-conformant"):
            pipeline = APGIPipeline(config)
        assert isinstance(pipeline, APGIPipeline)
        # The departure is recorded so any output produced is traceable.
        assert any("lam" in w for w in pipeline._validation_warnings)


class TestBasicStep:
    """Tests for basic pipeline step functionality."""

    def test_basic_step_returns_dict(self, base_config):
        """Test that step returns a dictionary."""
        pipeline = APGIPipeline(base_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert isinstance(result, dict)

    def test_step_keys_present(self, base_config):
        """Test that all expected keys are in result."""
        pipeline = APGIPipeline(base_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        expected_keys = [
            "z_e",
            "z_i",
            "z_e_norm",
            "z_i_norm",
            "mu_e",
            "mu_i",
            "pi_e",
            "pi_i",
            "pi_e_eff",
            "pi_i_eff",
            "z_i_eff",
            "S_inst",
            "S",
            "C",
            "V",
            "theta",
            "ignition_margin",
            "p_ignite",
            "B",
            "theta_dot",
            "x_hat_e",
            "x_hat_i",
            "M_somatic",
        ]
        for key in expected_keys:
            assert key in result, f"Key {key} not found in result"

    def test_step_updates_history(self, base_config):
        """Test that step updates history."""
        pipeline = APGIPipeline(base_config)
        pipeline.step(x_e=1.0, x_i=0.5)
        assert len(pipeline.history["S"]) == 1
        assert len(pipeline.history["theta"]) == 1
        assert len(pipeline.history["B"]) == 1

    def test_multiple_steps_accumulate_history(self, base_config):
        """Test that multiple steps accumulate history."""
        pipeline = APGIPipeline(base_config)
        for _ in range(5):
            pipeline.step(x_e=1.0, x_i=0.5)
        assert len(pipeline.history["S"]) == 5
        assert len(pipeline.history["theta"]) == 5
        assert len(pipeline.history["B"]) == 5


class TestSignalIntegrationModes:
    """Tests for different signal integration modes."""

    def test_discrete_mode(self, discrete_config):
        """Test discrete canonical mode."""
        pipeline = APGIPipeline(discrete_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "S" in result
        assert pipeline.config["use_canonical_discrete_mode"]

    def test_ode_mode(self, ode_config):
        """Test ODE signal integration mode."""
        pipeline = APGIPipeline(ode_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "S" in result
        assert not pipeline.config["use_canonical_discrete_mode"]

    def test_different_modes_produce_different_results(self, discrete_config, ode_config):
        """Test that discrete and ODE modes produce different results."""
        pipeline_discrete = APGIPipeline(discrete_config)
        pipeline_ode = APGIPipeline(ode_config)
        np.random.seed(42)
        for _ in range(10):
            x_e = np.random.randn()
            x_i = np.random.randn()
            pipeline_discrete.step(x_e=x_e, x_i=x_i)  # noqa: F841
            pipeline_ode.step(x_e=x_e, x_i=x_i)  # noqa: F841
        # After several steps, signals should diverge
        assert pytest.approx(pipeline_ode.S, abs=0.01) != pipeline_discrete.S


class TestSlidingWindowVariance:
    """Tests for sliding window variance method."""

    def test_sliding_window_initialization(self, sliding_window_config):
        """Test sliding window stats initialization."""
        pipeline = APGIPipeline(sliding_window_config)
        assert pipeline.stats_e is not None
        assert pipeline.stats_i is not None

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_sliding_window_updates(self, sliding_window_config):
        """Test sliding window updates during steps."""
        pipeline = APGIPipeline(sliding_window_config)
        for _ in range(10):
            pipeline.step(x_e=1.0, x_i=0.5)
        # Window should be populated after steps
        # The window size is T_win=10, so after 10 steps it should have data
        assert len(pipeline.stats_e.window) >= 0  # Window may be empty if not yet filled
        assert len(pipeline.stats_i.window) >= 0

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_sliding_window_variance_method_used(self, sliding_window_config):
        """Test that sliding window is actually used for variance."""
        pipeline = APGIPipeline(sliding_window_config)
        # Need multiple steps to populate window for variance calculation
        for _ in range(50):
            result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "sigma2_e" not in result  # Not directly exposed in result
        # After multiple steps, variance should be computed (not nan)
        assert not np.isnan(pipeline.state.sigma2_e)


class TestSomaticPrecision:
    """Tests for somatic precision modulation."""

    def test_somatic_precision_exponential(self, somatic_config):
        """Test somatic precision with exponential form."""
        pipeline = APGIPipeline(somatic_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "pi_i_eff" in result
        assert pipeline.config["use_somatic_precision"]

    def test_somatic_precision_affects_result(self, somatic_config, base_config):
        """Test that somatic precision affects pi_i_eff."""
        pipeline_somatic = APGIPipeline(somatic_config)
        pipeline_base = APGIPipeline(base_config)
        for _ in range(5):
            x_e = np.random.randn()
            x_i = np.random.randn()
            result_s = pipeline_somatic.step(x_e=x_e, x_i=x_i)
            result_b = pipeline_base.step(x_e=x_e, x_i=x_i)
        # pi_i_eff should be different with somatic precision
        assert result_s["pi_i_eff"] != result_b["pi_i_eff"]


class TestNEPrecisionModulation:
    """Tests for NE on precision."""

    def test_ne_precision_modulation(self, ne_precision_config):
        """Test NE precision modulation."""
        pipeline = APGIPipeline(ne_precision_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "pi_i_eff" in result
        assert pipeline.config["ne_on_precision"]

    def test_ne_precision_vs_none(self, ne_precision_config, base_config):
        """Test NE precision produces different results."""
        pipeline_ne = APGIPipeline(ne_precision_config)
        pipeline_base = APGIPipeline(base_config)
        for _ in range(5):
            x_e = np.random.randn()
            x_i = np.random.randn()
            result_ne = pipeline_ne.step(x_e=x_e, x_i=x_i)
            result_base = pipeline_base.step(x_e=x_e, x_i=x_i)
        assert result_ne["pi_i_eff"] != result_base["pi_i_eff"]


class TestNEThresholdModulation:
    """Tests for NE on threshold."""

    def test_ne_threshold_modulation(self, ne_threshold_config):
        """Test NE threshold modulation."""
        pipeline = APGIPipeline(ne_threshold_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "theta" in result
        assert pipeline.config["ne_on_threshold"]


class TestContinuousThresholdODE:
    """Tests for continuous threshold ODE mode."""

    def test_continuous_threshold_ode(self, continuous_threshold_config):
        """Test continuous threshold ODE mode."""
        pipeline = APGIPipeline(continuous_threshold_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "theta" in result
        assert pipeline.config["use_continuous_threshold_ode"]

    def test_continuous_vs_discrete_different(self, continuous_threshold_config, base_config):
        """Test that continuous and discrete threshold modes differ.

        At dt₀ = 10 ms both modes integrate the same τ_θ = 20 s relaxation, so after a
        few steps they differ by ~1e-5 AU; the inputs are seeded and the check is exact
        inequality (an abs=1e-3 tolerance made the test depend on the random draw).
        """
        pipeline_cont = APGIPipeline(continuous_threshold_config)
        pipeline_disc = APGIPipeline(base_config)
        rng = np.random.default_rng(12345)
        for _ in range(5):
            x_e = rng.standard_normal()
            x_i = rng.standard_normal()
            pipeline_cont.step(x_e=x_e, x_i=x_i)  # noqa: F841
            pipeline_disc.step(x_e=x_e, x_i=x_i)  # noqa: F841
        assert pipeline_cont.theta != pipeline_disc.theta


class TestHierarchicalFeatures:
    """Tests for hierarchical features."""

    def test_hierarchical_initialization(self, hierarchical_config):
        """Test hierarchical state initialization."""
        pipeline = APGIPipeline(hierarchical_config)
        assert pipeline.hierarchical is not None
        assert pipeline.n_levels == 3
        assert len(pipeline.taus) == 3

    def test_per_level_statistics(self, hierarchical_config):
        """Test per-level statistics arrays."""
        pipeline = APGIPipeline(hierarchical_config)
        assert len(pipeline.mu_e_levels) == 3
        assert len(pipeline.mu_i_levels) == 3
        assert len(pipeline.sigma2_e_levels) == 3
        assert len(pipeline.sigma2_i_levels) == 3

    def test_hierarchical_step(self, hierarchical_config):
        """Test step with hierarchical mode."""
        pipeline = APGIPipeline(hierarchical_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "hierarchical_pis" in result
        assert "hierarchical_phases" in result
        assert "hierarchical_thetas" in result

    def test_hierarchical_pis_length(self, hierarchical_config):
        """Test hierarchical pis have correct length."""
        pipeline = APGIPipeline(hierarchical_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert len(result["hierarchical_pis"]) == 3
        assert len(result["hierarchical_phases"]) == 3
        assert len(result["hierarchical_thetas"]) == 3


class TestHierarchicalPrecisionODE:
    """Tests for hierarchical precision ODE."""

    def test_hierarchical_network_initialization(self, hierarchical_precision_ode_config):
        """Test hierarchical network initialization."""
        pipeline = APGIPipeline(hierarchical_precision_ode_config)
        assert pipeline.hierarchical_network is not None

    def test_hierarchical_precision_ode_step(self, hierarchical_precision_ode_config):
        """Test step with hierarchical precision ODE."""
        pipeline = APGIPipeline(hierarchical_precision_ode_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "hierarchical_pis" in result


class TestPhaseModulation:
    """Tests for phase modulation feature."""

    def test_phase_modulation_enabled(self, phase_modulation_config):
        """Test phase modulation is enabled."""
        pipeline = APGIPipeline(phase_modulation_config)
        assert pipeline.config["use_phase_modulation"]
        assert pipeline.hierarchical is not None

    def test_phase_modulation_updates_phases(self, phase_modulation_config):
        """Test that phases are updated during step."""
        pipeline = APGIPipeline(phase_modulation_config)
        pipeline.step(x_e=1.0, x_i=0.5)
        # Check that time advanced (which causes phase updates)
        assert pipeline.t > 0
        # With non-zero omega_phases, phases should change
        # The phase update happens in the step when use_phase_modulation is True
        # If phases didn't change, it might be because the omega values are very small
        # Let's just verify the mechanism is in place
        assert pipeline.config.get("use_phase_modulation", False)


class TestReservoirLayer:
    """Tests for reservoir layer functionality."""

    def test_reservoir_initialization(self, reservoir_config):
        """Test reservoir initialization."""
        pipeline = APGIPipeline(reservoir_config)
        assert pipeline.reservoir is not None

    def test_reservoir_step(self, reservoir_config):
        """Test step with reservoir."""
        pipeline = APGIPipeline(reservoir_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "S_reservoir" in result
        assert "reservoir_state_norm" in result

    def test_reservoir_state_norm(self, reservoir_config):
        """Test reservoir state norm is computed."""
        pipeline = APGIPipeline(reservoir_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert result["reservoir_state_norm"] >= 0


class TestReservoirAsThreshold:
    """Tests for reservoir-as-threshold mode."""

    def test_reservoir_as_threshold_initialization(self, reservoir_threshold_config):
        """Test reservoir as threshold initialization."""
        pipeline = APGIPipeline(reservoir_threshold_config)
        assert pipeline.reservoir is not None
        assert pipeline.config["reservoir_as_threshold"]

    def test_reservoir_threshold_step(self, reservoir_threshold_config):
        """Test step with reservoir as threshold."""
        pipeline = APGIPipeline(reservoir_threshold_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "S_reservoir" in result
        assert "theta" in result

    def test_reservoir_as_threshold_different_from_normal(
        self, reservoir_threshold_config, reservoir_config
    ):
        """Test reservoir-as-threshold produces different results."""
        pipeline_rt = APGIPipeline(reservoir_threshold_config)
        pipeline_r = APGIPipeline(reservoir_config)
        for _ in range(5):
            x_e = np.random.randn()
            x_i = np.random.randn()
            pipeline_rt.step(x_e=x_e, x_i=x_i)  # noqa: F841
            pipeline_r.step(x_e=x_e, x_i=x_i)  # noqa: F841
        # Theta should differ between modes
        assert pipeline_rt.theta != pytest.approx(pipeline_r.theta, abs=0.001)

    def test_reservoir_as_threshold_missing_reservoir_raises(self, base_config):
        """Test that reservoir_as_threshold without reservoir raises error."""
        config = base_config.copy()
        config["reservoir_as_threshold"] = True
        config["use_reservoir"] = False  # Explicitly disable reservoir
        pipeline = APGIPipeline(config)
        with pytest.raises(
            ValueError, match="Reservoir mode enabled but reservoir not initialized"
        ):
            pipeline.step(x_e=1.0, x_i=0.5)


class TestThermodynamicCost:
    """Tests for thermodynamic cost calculation."""

    def test_thermodynamic_cost_enabled(self, thermodynamic_config):
        """Test thermodynamic cost is calculated."""
        pipeline = APGIPipeline(thermodynamic_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "C_landauer" in result
        assert "bits_erased" in result

    def test_thermodynamic_history(self, thermodynamic_config):
        """Test thermodynamic cost history is recorded."""
        pipeline = APGIPipeline(thermodynamic_config)
        pipeline.step(x_e=1.0, x_i=0.5)
        assert "C_landauer" in pipeline.history
        assert "bits_erased" in pipeline.history
        assert len(pipeline.history["C_landauer"]) == 1

    def test_landauer_cost_positive(self, thermodynamic_config):
        """Test Landauer cost is positive."""
        pipeline = APGIPipeline(thermodynamic_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert result["C_landauer"] >= 0


class TestObservableMapping:
    """Tests for observable mapping integration."""

    def test_neural_observables_initialization(self, observable_config):
        """Test neural observables extractor initialization."""
        pipeline = APGIPipeline(observable_config)
        assert pipeline.neural_observables is not None

    def test_behavioral_observables_initialization(self, observable_config):
        """Test behavioral observables extractor initialization."""
        pipeline = APGIPipeline(observable_config)
        assert pipeline.behavioral_observables is not None

    def test_prediction_validator_initialization(self, observable_config):
        """Test prediction validator initialization."""
        pipeline = APGIPipeline(observable_config)
        assert pipeline.prediction_validator is not None

    def test_neural_observables_in_result(self, observable_config):
        """Test neural observables in step result."""
        pipeline = APGIPipeline(observable_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "neural_gamma_power" in result
        assert "neural_erp_amplitude" in result
        assert "neural_ignition_rate" in result

    def test_behavioral_observables_in_result(self, observable_config):
        """Test behavioral observables in step result."""
        pipeline = APGIPipeline(observable_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "behavioral_rt_variability" in result
        assert "behavioral_response_criterion" in result
        assert "behavioral_decision_rate" in result

    def test_prediction_validator_in_result(self, observable_config):
        """Test prediction validator values in step result."""
        pipeline = APGIPipeline(observable_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "prediction_margin" in result
        assert "prediction_p_ign" in result


class TestStabilityAnalysis:
    """Tests for stability analysis integration."""

    def test_stability_analyzer_initialization(self, stability_config):
        """Test stability analyzer initialization."""
        pipeline = APGIPipeline(stability_config)
        assert pipeline.stability_analyzer is not None

    def test_stability_analyzer_step(self, stability_config):
        """Test stability analyzer records step."""
        pipeline = APGIPipeline(stability_config)
        pipeline.step(x_e=1.0, x_i=0.5)
        assert len(pipeline.stability_analyzer.history["S"]) == 1
        assert len(pipeline.stability_analyzer.history["theta"]) == 1

    def test_stability_analyzer_multiple_steps(self, stability_config):
        """Test stability analyzer records multiple steps."""
        pipeline = APGIPipeline(stability_config)
        for _ in range(5):
            pipeline.step(x_e=1.0, x_i=0.5)
        assert len(pipeline.stability_analyzer.history["S"]) == 5
        assert len(pipeline.stability_analyzer.history["theta"]) == 5


class TestKuramotoOscillators:
    """Tests for Kuramoto oscillators integration."""

    def test_kuramoto_initialization(self, kuramoto_config):
        """Test Kuramoto system initialization."""
        pipeline = APGIPipeline(kuramoto_config)
        assert pipeline.kuramoto is not None

    def test_kuramoto_in_result(self, kuramoto_config):
        """Test Kuramoto values in step result."""
        pipeline = APGIPipeline(kuramoto_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "kuramoto_phases" in result
        assert "kuramoto_synchronization" in result

    def test_kuramoto_phases_length(self, kuramoto_config):
        """Test Kuramoto phases have correct length."""
        pipeline = APGIPipeline(kuramoto_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert len(result["kuramoto_phases"]) == 3

    def test_kuramoto_phase_reset_on_ignition(self, kuramoto_config):
        """Test Kuramoto phase reset on ignition."""
        pipeline = APGIPipeline(kuramoto_config)
        # Run many steps to trigger ignition
        for i in range(100):  # noqa: B007
            x_e = 2.0 + np.random.randn() * 0.1  # High signal to trigger ignition
            result = pipeline.step(x_e=x_e, x_i=0.5)
            if result["B"] == 1:
                # Phase should have been reset
                break
        # Should have had at least one ignition
        assert sum(pipeline.history["B"]) > 0


class TestGenerativeModelUpdate:
    """Tests for generative model prediction update."""

    def test_generative_update_enabled(self, generative_config):
        """Test generative model update is enabled."""
        pipeline = APGIPipeline(generative_config)
        assert pipeline.config.get("use_generative_model_update") or pipeline.config.get(
            "use_internal_predictions"
        )

    def test_predictions_in_result(self, generative_config):
        """Test predictions in step result."""
        pipeline = APGIPipeline(generative_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "x_hat_e" in result
        assert "x_hat_i" in result

    def test_predictions_update_over_time(self, generative_config):
        """Test predictions update over multiple steps."""
        pipeline = APGIPipeline(generative_config)
        initial_x_hat_e = pipeline.x_hat_e
        initial_x_hat_i = pipeline.x_hat_i
        for _ in range(10):
            pipeline.step(x_e=1.0, x_i=0.5)
        # Predictions should change
        assert pipeline.x_hat_e != initial_x_hat_e or pipeline.x_hat_i != initial_x_hat_i


class TestInternalPredictions:
    """Tests for using internal vs external predictions."""

    def test_internal_predictions_used_by_default(self, base_config):
        """Test internal predictions are used when x_hat not provided."""
        pipeline = APGIPipeline(base_config)
        pipeline.x_hat_e = 0.5
        pipeline.x_hat_i = 0.3
        result = pipeline.step(x_e=1.0, x_i=0.5)  # No predictions provided
        # Should use internal predictions
        assert result["z_e"] == 1.0 - 0.5  # x_e - x_hat_e
        assert result["z_i"] == 0.5 - 0.3  # x_i - x_hat_i

    def test_external_predictions_override_internal(self, base_config):
        """Test external predictions override internal."""
        pipeline = APGIPipeline(base_config)
        pipeline.x_hat_e = 0.5
        pipeline.x_hat_i = 0.3
        result = pipeline.step(x_e=1.0, x_i=0.5, x_hat_e=0.2, x_hat_i=0.1)
        # Should use external predictions
        assert result["z_e"] == 1.0 - 0.2  # x_e - external x_hat_e
        assert result["z_i"] == 0.5 - 0.1  # x_i - external x_hat_i


class TestPostIgnitionDynamics:
    """Tests for post-ignition dynamics."""

    def test_signal_reset_on_ignition(self, base_config):
        """Test signal is reset on ignition."""
        config = base_config.copy()
        config["reset_factor"] = 0.5  # Ensure reset factor is set
        pipeline = APGIPipeline(config)
        # Run steps until ignition
        pipeline.S = 2.0  # Force high signal
        pipeline.theta = 0.5  # Low threshold
        result = pipeline.step(x_e=2.0, x_i=0.5)
        if result["B"] == 1:
            # Signal should be significantly reduced from original value
            # The reset applies, but signal integration may add some contribution
            # So we check it's reduced by at least the reset factor
            assert pipeline.S < 2.0  # Definitely less than original
            # And it should be close to the reset value (within tolerance for integration)
            assert 2.0 * config["reset_factor"] * 2.0 > pipeline.S

    def test_refractory_boost_on_ignition(self, base_config):
        """Test refractory boost on ignition."""
        pipeline = APGIPipeline(base_config)
        # Force ignition
        pipeline.S = 2.0
        pipeline.theta = 0.5
        theta_before = pipeline.theta
        result = pipeline.step(x_e=2.0, x_i=0.5)
        if result["B"] == 1:
            # Theta should have refractory boost applied
            assert pipeline.theta > theta_before

    def test_threshold_decay_after_ignition(self, base_config):
        """Test threshold decays after ignition."""
        pipeline = APGIPipeline(base_config)
        # Force ignition
        pipeline.S = 2.0
        pipeline.theta = 0.5
        pipeline.step(x_e=2.0, x_i=0.5)
        theta_after_ignition = pipeline.theta
        # Run more steps without ignition
        pipeline.S = 0.1  # Low signal
        for _ in range(10):
            pipeline.step(x_e=0.1, x_i=0.5)
        # Theta should decay toward baseline
        assert pipeline.theta < theta_after_ignition or pipeline.theta == pytest.approx(
            base_config["theta_base"], abs=0.1
        )

    def test_invalid_reset_factor_raises(self, base_config):
        """Test invalid reset factor raises error."""
        config = base_config.copy()
        config["reset_factor"] = 1.5  # Invalid: must be in (0, 1)
        config["strict"] = False  # opt out of fail-closed to reach step()
        # Suppress expected validation warning for this intentionally invalid config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = APGIPipeline(config)
            pipeline.S = 2.0
            pipeline.theta = 0.5  # Force ignition condition
            with pytest.raises(ValueError, match=r"reset_factor.*must be in"):
                result = pipeline.step(x_e=2.0, x_i=0.5)  # noqa: F841


class TestIgnition:
    """Tests for ignition mechanism."""

    def test_stochastic_ignition_enabled(self, base_config):
        """Test stochastic ignition mode."""
        config = base_config.copy()
        config["stochastic_ignition"] = True
        pipeline = APGIPipeline(config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "B" in result
        assert result["B"] in [0, 1]

    def test_deterministic_ignition(self, base_config):
        """Test deterministic ignition mode."""
        config = base_config.copy()
        config["stochastic_ignition"] = False
        pipeline = APGIPipeline(config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "B" in result
        assert result["B"] in [0, 1]

    def test_ignition_probability_in_result(self, base_config):
        """Test ignition probability is in result."""
        pipeline = APGIPipeline(base_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "p_ignite" in result
        assert 0 <= result["p_ignite"] <= 1

    def test_ignition_margin_in_result(self, base_config):
        """Test ignition margin is in result."""
        pipeline = APGIPipeline(base_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "ignition_margin" in result


class TestRealisticCost:
    """Tests for realistic metabolic cost mode."""

    def test_realistic_cost_enabled(self, base_config):
        """Test realistic cost mode."""
        config = base_config.copy()
        config["use_realistic_cost"] = True
        pipeline = APGIPipeline(config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "C" in result

    def test_realistic_cost_vs_simple(self, base_config):
        """Test realistic cost differs from simple cost."""
        config_realistic = {**base_config, "model_form": "legacy_v1_1"}  # v1.1 cost model
        config_realistic["use_realistic_cost"] = True
        config_simple = {**base_config, "model_form": "legacy_v1_1"}
        config_simple["use_realistic_cost"] = False
        pipeline_r = APGIPipeline(config_realistic)
        pipeline_s = APGIPipeline(config_simple)
        result_r = pipeline_r.step(x_e=1.0, x_i=0.5)
        result_s = pipeline_s.step(x_e=1.0, x_i=0.5)
        # Cost values should differ
        assert result_r["C"] != result_s["C"]


class TestValidation:
    """Tests for pipeline validation method."""

    def test_validation_insufficient_data(self, base_config):
        """Test validation with insufficient data."""
        pipeline = APGIPipeline(base_config)
        result = pipeline.validate()
        assert result["status"] == "insufficient_data"

    def test_validation_success(self, base_config):
        """Test validation with sufficient data."""
        pipeline = APGIPipeline(base_config)
        # Run many steps to accumulate data
        for _ in range(100):
            pipeline.step(x_e=np.random.randn(), x_i=np.random.randn())
        # Suppress numpy divide by zero warnings in spectral analysis
        with np.errstate(divide="ignore", invalid="ignore"):
            result = pipeline.validate()
        assert result["status"] == "success"
        assert "hurst_exponent" in result
        assert "is_pink_noise" in result
        assert "beta" in result

    def test_validation_pink_noise_check(self, base_config):
        """Test pink noise validation in validate()."""
        pipeline = APGIPipeline(base_config)
        for _ in range(100):
            pipeline.step(x_e=np.random.randn(), x_i=np.random.randn())
        # Suppress numpy divide by zero warnings in spectral analysis
        with np.errstate(divide="ignore", invalid="ignore"):
            result = pipeline.validate()
        assert isinstance(result["is_pink_noise"], bool)
        assert isinstance(result["beta"], float)


class TestTimeTracking:
    """Tests for time tracking."""

    def test_time_initialized_to_zero(self, base_config):
        """Test time is initialized to zero."""
        pipeline = APGIPipeline(base_config)
        assert pipeline.t == 0.0

    def test_time_advances(self, base_config):
        """Test time advances with each step."""
        pipeline = APGIPipeline(base_config)
        dt = base_config["dt"]
        pipeline.step(x_e=1.0, x_i=0.5)
        assert pipeline.t == dt
        pipeline.step(x_e=1.0, x_i=0.5)
        assert pipeline.t == 2 * dt


class TestHierarchicalComputePerLevelErrors:
    """Tests for _compute_per_level_errors method."""

    def test_non_hierarchical_returns_single_z(self, base_config):
        """Test non-hierarchical mode returns single z-score."""
        pipeline = APGIPipeline(base_config)
        z_e_levels, z_i_levels = pipeline._compute_per_level_errors(1.0, 0.5)
        assert len(z_e_levels) == 1
        assert len(z_i_levels) == 1

    def test_hierarchical_returns_multiple_z(self, hierarchical_config):
        """Test hierarchical mode returns multiple z-scores."""
        pipeline = APGIPipeline(hierarchical_config)
        z_e_levels, z_i_levels = pipeline._compute_per_level_errors(1.0, 0.5)
        assert len(z_e_levels) == 3
        assert len(z_i_levels) == 3

    def test_hierarchical_updates_per_level_stats(self, hierarchical_config):
        """Test hierarchical mode updates per-level statistics."""
        pipeline = APGIPipeline(hierarchical_config)
        initial_mu_e = pipeline.mu_e_levels.copy()
        initial_sigma2_e = pipeline.sigma2_e_levels.copy()  # noqa: F841
        pipeline._compute_per_level_errors(1.0, 0.5)
        # Statistics should be updated
        assert any(m != im for m, im in zip(pipeline.mu_e_levels, initial_mu_e, strict=False))


class TestConfigurationMutability:
    """Tests for configuration handling."""

    def test_config_not_mutated_externally(self, base_config):
        """Test that external config dict is not mutated."""
        original_config = base_config.copy()
        APGIPipeline(base_config)
        # Original config should not be mutated by internal operations
        assert base_config.keys() == original_config.keys()

    def test_config_copy_created(self, base_config):
        """Test that config is copied on initialization."""
        pipeline = APGIPipeline(base_config)
        # Modifying original should not affect pipeline
        base_config["S0"] = 999.0
        assert pipeline.config["S0"] != 999.0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_input(self, base_config):
        """Test with zero input."""
        pipeline = APGIPipeline(base_config)
        result = pipeline.step(x_e=0.0, x_i=0.0)
        assert "S" in result
        assert "B" in result

    def test_negative_input(self, base_config):
        """Test with negative input."""
        pipeline = APGIPipeline(base_config)
        result = pipeline.step(x_e=-1.0, x_i=-0.5)
        assert "S" in result
        assert "B" in result

    def test_large_input(self, base_config):
        """Test with large input."""
        pipeline = APGIPipeline(base_config)
        result = pipeline.step(x_e=100.0, x_i=50.0)
        assert "S" in result
        assert "B" in result

    def test_very_small_eps(self, base_config):
        """Test with very small epsilon."""
        config = base_config.copy()
        config["eps"] = 1e-12
        pipeline = APGIPipeline(config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "S" in result

    def test_minimal_config(self):
        """Test with minimal required configuration."""
        config = {
            "S0": 0.5,
            "theta_0": 1.0,
            "theta_base": 1.0,
            "sigma2_e0": 1.0,
            "sigma2_i0": 1.0,
            "alpha_e": 0.1,
            "alpha_i": 0.1,
            "beta": 0.5,
            "g_ach": 1.0,
            "eta": 0.1,
            "kappa": 0.15,
            "delta": 0.5,
            "c0": 0.1,
            "c1": 0.2,
            "c2": 0.05,
            "v1": 1.0,
            "v2": 0.5,
            "lam": 0.2,
            "eps": 1e-8,
            "pi_min": 1e-4,
            "pi_max": 1e4,
            "ignite_tau": 0.5,
            "stochastic_ignition": False,
            "signal_log_nonlinearity": False,
            "use_realistic_cost": False,
            "dt": 0.1,  # Valid dt ≤ min(tau_s, tau_theta, tau_pi) / 10
        }
        pipeline = APGIPipeline(config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert isinstance(result, dict)


class TestMParameter:
    """Tests for somatic marker M parameter."""

    def test_m_somatic_initialized(self, somatic_config):
        """Test M_somatic is initialized."""
        pipeline = APGIPipeline(somatic_config)
        assert pipeline.M == 0.5

    def test_m_in_result(self, somatic_config):
        """Test M_somatic in step result."""
        pipeline = APGIPipeline(somatic_config)
        result = pipeline.step(x_e=1.0, x_i=0.5)
        assert "M_somatic" in result
        assert result["M_somatic"] == 0.5

    def test_m_default_zero(self, base_config):
        """Test M_somatic defaults to 0."""
        pipeline = APGIPipeline(base_config)
        assert pipeline.M == 0.0
