import pytest

from core.validation import (
    ValidationError,
    format_constraint_summary,
    get_constraint_summary,
    print_constraint_summary,
    validate_config,
    validate_parameter,
    validate_reset_factor,
)


def test_validate_config_success():
    config = {
        "lam": 0.2,
        "kappa": 0.15,
        "ignite_tau": 0.5,
        "dt": 0.5,  # dt <= min(tau)/10 per MathSpec §7.4
        "tau_s": 50.0,
        "tau_theta": 100.0,
        "tau_pi": 100.0,
        "pi_min": 1e-4,
        "pi_max": 1e4,
        "eps": 1e-8,
        "eta": 0.1,
        "noise_std": 0.01,
        "g_ach": 1.0,
        "g_ne": 1.0,
        "beta": 1.15,
        "reset_factor": 0.5,
    }
    validate_config(config)


def test_validate_config_failures():
    # Base valid config to avoid unrelated errors
    base = {
        "dt": 0.1,
        "tau_s": 10.0,
        "tau_theta": 100.0,
        "tau_pi": 100.0,
        "lam": 0.5,
        "kappa": 0.1,
        "ignite_tau": 0.5,
        "pi_min": 0.1,
        "pi_max": 10.0,
    }

    def check(update):
        cfg = base.copy()
        cfg.update(update)
        validate_config(cfg)

    # Neuromodulator separation
    with pytest.warns(RuntimeWarning, match="sub-dominant"):
        check(
            {
                "ne_on_precision": True,
                "ne_on_threshold": True,
                "gamma_ne": 1.0,
                "g_ne": 1.0,  # product = 1.0 > 0.5 ceiling
            }
        )
    # Signal accumulation
    with pytest.raises(ValidationError, match="lam must be in"):
        check({"lam": 1.5})
    # Threshold dynamics
    with pytest.raises(ValidationError, match="decay exponent dt/tau_theta must be > 0"):
        check({"kappa": 0.0})
    # Ignition dynamics
    with pytest.raises(ValidationError, match="ignite_tau"):
        check({"ignite_tau": 0.0})
    # Continuous-time SDE (dt)
    with pytest.raises(ValidationError, match="dt must be > 0"):
        check({"dt": 0.0})
    with pytest.raises(ValidationError, match="dt=2.0 exceeds max"):
        check({"dt": 2.0, "tau_s": 5.0})
    # Hierarchical
    with pytest.raises(ValidationError, match="timescale_k must be > 1"):
        check({"use_hierarchical": True, "timescale_k": 1.0})
    with pytest.raises(ValidationError, match="τ_0 = 1.00 ≤ 1"):
        check({"use_hierarchical": True, "timescale_k": 1.5, "tau_0": 1.0})
    # Precision
    with pytest.raises(ValidationError, match="pi_min must be > 0"):
        check({"pi_min": 0.0})
    with pytest.raises(ValidationError, match="pi_max must be > pi_min"):
        check({"pi_max": 1.0, "pi_min": 2.0})
    # Learning rates (Stability)
    with pytest.raises(ValidationError, match="kappa_e=1.0 >= 0.0002"):
        check({"use_internal_predictions": True, "kappa_e": 1.0, "pi_max": 10000.0, "dt": 0.000001})
    with pytest.raises(ValidationError, match="kappa_i=1.0 >= 0.0002"):
        check(
            {
                "use_internal_predictions": True,
                "kappa_e": 0.0001,  # must be valid to reach kappa_i check
                "kappa_i": 1.0,
                "pi_max": 10000.0,
                "dt": 0.000001,
            }
        )
    # EMA
    with pytest.raises(ValidationError, match="alpha_e=0.0 must be in"):
        check({"variance_method": "ema", "alpha_e": 0.0})
    with pytest.raises(ValidationError, match="alpha_i=0.0 must be in"):
        check({"variance_method": "ema", "alpha_i": 0.0})
    # Sliding window
    with pytest.raises(ValidationError, match="T_win=0 must be positive integer"):
        check({"variance_method": "sliding_window", "T_win": 0})
    # Numerical stability
    with pytest.raises(ValidationError, match="eps must be in"):
        check({"eps": 1.5})
    with pytest.raises(ValidationError, match="eta must be in"):
        check({"eta": 1.5})
    with pytest.raises(ValidationError, match="noise_std must be"):
        check({"noise_std": -1.0})
    with pytest.raises(ValidationError, match="c1=-1.0 must be"):
        check({"c1": -1.0})
    with pytest.raises(ValidationError, match="delta=-1.0 must be"):
        check({"delta": -1.0})
    with pytest.raises(ValidationError, match="signal_log_nonlinearity must be boolean"):
        check({"signal_log_nonlinearity": "yes"})
    # Neuromodulator gains
    with pytest.raises(ValidationError, match="g_ach"):
        check({"g_ach": -1.0})
    with pytest.raises(ValidationError, match="g_ne"):
        check({"g_ne": -1.0})
    # β_DA is an unconstrained real (MathSpec glossary: "ℝ"); negative values
    # model dopaminergic depletion and must be accepted, not rejected.
    check({"beta": -1.0})


def test_validate_reset_factor():
    validate_reset_factor(0.5)
    with pytest.raises(ValidationError, match="reset_factor"):
        validate_reset_factor(0.0)
    with pytest.raises(ValidationError, match="reset_factor"):
        validate_reset_factor(1.0)


def test_validate_parameter():
    validate_parameter("x", 0.5, "in (0, 1)")
    with pytest.raises(ValidationError):
        validate_parameter("x", 1.5, "in (0, 1)")
    validate_parameter("x", 10.0, ">= 10")
    with pytest.raises(ValidationError):
        validate_parameter("x", 9.0, ">= 10")
    validate_parameter("x", 11.0, "> 10")
    with pytest.raises(ValidationError):
        validate_parameter("x", 10.0, "> 10")
    validate_parameter("x", 9.0, "<= 10")
    with pytest.raises(ValidationError):
        validate_parameter("x", 11.0, "<= 10")
    validate_parameter("x", 9.0, "< 10")
    with pytest.raises(ValidationError):
        validate_parameter("x", 10.0, "< 10")
    # Unsupported constraint format falls through without error
    validate_parameter("x", 1.0, "unsupported")


def test_warnings():
    # Precision range warning
    with pytest.warns(RuntimeWarning, match="Precision range very large"):
        validate_config(
            {
                "pi_min": 1e-5,
                "pi_max": 1e4,
                "dt": 0.0001,
                "tau_s": 10.0,
                "tau_theta": 100.0,
                "tau_pi": 100.0,
            }
        )
    # Small window warning
    with pytest.warns(RuntimeWarning, match="T_win=4 is very small"):
        validate_config(
            {
                "variance_method": "sliding_window",
                "T_win": 4,
                "dt": 0.1,
                "tau_s": 10.0,
                "tau_theta": 100.0,
                "tau_pi": 100.0,
            }
        )


def test_summary(capsys):
    summary = get_constraint_summary()
    assert "Signal Accumulation" in summary
    fmt = format_constraint_summary()
    assert "APGI PARAMETER CONSTRAINTS" in fmt
    print_constraint_summary()
    captured = capsys.readouterr()
    assert "APGI PARAMETER CONSTRAINTS" in captured.out
