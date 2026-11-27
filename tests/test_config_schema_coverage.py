import pytest
from pydantic import ValidationError

from core.config_schema import APGIConfig, create_production_config


def test_apgi_config_defaults():
    cfg = APGIConfig()
    assert cfg.S0 == 0.0
    assert cfg.theta_0 == 1.0
    assert cfg.variance_method == "ema"


def test_apgi_config_validation_failures():
    # pi_max > pi_min
    with pytest.raises(ValidationError, match="pi_max must be greater than pi_min"):
        APGIConfig(pi_min=100, pi_max=50)
    # NE: both pathways together are canonical (SSOT-P13), so neither raises
    APGIConfig(ne_on_precision=True, ne_on_threshold=True)
    APGIConfig(ne_on_precision=False, ne_on_threshold=True, gamma_ne=0.2)
    # dt stability
    with pytest.raises(ValidationError, match="exceeds max"):
        APGIConfig(dt=2.0, tau_s=5.0)
    # Learning rate stability
    with pytest.raises(ValidationError, match="Spec §1.4 requires η_pred,e < 2/Π_max"):
        APGIConfig(use_internal_predictions=True, eta_pred_e=1.0, pi_max=100)
    with pytest.raises(ValidationError, match="Spec §1.4 requires η_pred,i < 2/Π_max"):
        APGIConfig(use_internal_predictions=True, eta_pred_e=0.001, eta_pred_i=1.0, pi_max=100)
    # EMA bounds
    with pytest.raises(ValidationError, match="less than 1"):
        APGIConfig(variance_method="ema", alpha_e=1.5)
    with pytest.raises(ValidationError, match="greater than 0"):
        APGIConfig(variance_method="ema", alpha_i=0.0)


def test_apgi_config_backward_compat():
    # beta_da alias
    cfg = APGIConfig(beta_da=2.5)
    assert cfg.beta == 2.5
    # tau_sigma alias
    cfg2 = APGIConfig(tau_sigma=0.8)
    assert cfg2.ignite_tau == 0.8
    # Explicit mismatch (beta_da takes precedence)
    cfg3 = APGIConfig(beta=1.0, beta_da=2.0)
    assert cfg3.beta == 2.0


def test_to_from_dict():
    cfg = APGIConfig(lam=0.5)
    d = cfg.to_dict()
    assert d["lam"] == 0.5
    cfg2 = APGIConfig.from_dict(d)
    assert cfg2.lam == 0.5


def test_create_production_config():
    # Strict
    cfg = create_production_config(strict=True, lam=0.3)
    assert isinstance(cfg, APGIConfig)
    assert cfg.lam == 0.3
    # Non-strict (returns dict)
    d = create_production_config(strict=False, lam=0.4)
    assert isinstance(d, dict)
    assert d["lam"] == 0.4
