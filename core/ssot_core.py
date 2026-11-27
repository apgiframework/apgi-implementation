"""APGI v1.2 minimal core: the canonical 15-step pipeline (SSOT §8, Math Spec v1.2 §13 Table 3).

This is the default `model_form="ssot_core"` path of `pipeline.APGIPipeline.step`. It follows the
reference implementation `apgi_ref.core_step` in apgi-lean (verification/python/apgi_ref.py) step
for step, and apgi-lean's test_production.py holds this module to it (SSOT-P1…P9).

Equations (apgi-lean specification/equations.json):
  2  EQ-ERR-1      ε = x − x̂ (previous predictions)
  3  EQ-STD-1      z = (ε − μ_ref)/(σ_ref + ε_stab); reference windows exclude ε(t)
  4  EQ-PREC-2     Π = clip(1/(Σ(t) + ε_stab)); Σ(t) is the variance BEFORE ε(t) enters it
  5  EQ-NE-1, EQ-PREC-ACH-1, EQ-PREC-SM-1, EQ-DA-1
  6  EQ-DRIVE-1    S_inst = Πᵉ_rel·(|zᵉ| − m₀) + Πⁱ_rel·(|zⁱ_eff| − m₀)   (signed, centered)
  7  EQ-THEFF-1    θ_eff = θ·(1 + γ_NE·g_NE) + r(t) + β_5HT − u(t)      (read-out only)
  8  EQ-IGN-2      Bₜ = σ(γ_sig(S(t) − θ_eff)) from the EXISTING S
  9  EQ-HAZARD-1   B(t) ~ Bernoulli(1 − exp(−r_max·Bₜ·dt)); hard limit S > θ_eff when not stochastic
 10  EQ-COST-1, EQ-VAL-1, EQ-ALLO-1   θ' = θ + η_θ(C − V) + δ_reset·B; exact decay; floor
 11  EQ-REF-1      r ← r·exp(−dt/τ_ref) + δ_ref·B
 12  EQ-RESET-1    S ← ρ_retain·S on B = 1, exactly once
 13  EQ-LEAK-1     S ← (1 − λ)S + λ·S_inst
 14  EQ-EMA-1/2, EQ-PRED-1, reference windows updated with ε(t)
 15  B_prev ← B
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from core.finite import NonFiniteStateError

M0 = math.sqrt(2.0 / math.pi)  # m₀ = E|z| for z ~ N(0, 1) (Lean: Apgi.meanAbsStdNormal_eq)
NE_BOUND = 0.5


def ne_clamp(gamma_ne: float, g_ne: float, bound: float = NE_BOUND) -> float:
    """EQ-NE-1 for either sign of γ_NE (Lean: Apgi.neClamp_bound)."""
    return g_ne if abs(gamma_ne) * g_ne <= bound else bound / abs(gamma_ne)


def clip(lo: float, hi: float, x: float) -> float:
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def ema_alpha(dt: float, window: float) -> float:
    """EQ-EMA-2: α_EMA = 1 − exp(−dt/T_window)."""
    return -math.expm1(-dt / window)


def operating_point(gamma_sig: float, theta0: float, r_max: float) -> dict:
    """R1 / TC-14: γ_sig·θ₀, the soft floor σ(−γ_sig·θ₀) and the stimulus-free hazard (s⁻¹)."""
    floor = sigmoid(-gamma_sig * theta0)
    return {
        "gamma_sig_theta0": gamma_sig * theta0,
        "propensity_floor": floor,
        "baseline_rate": r_max * floor,
    }


class SSOTCore:
    """Per-pipeline state that the SSOT core needs beyond APGIPipeline's own (reference windows, r(t))."""

    def __init__(self, config: dict):
        dt = float(config.get("dt", 0.01))
        self.n_e = max(2, round(float(config.get("T_window_e", 5.0)) / dt))
        self.n_i = max(2, round(float(config.get("T_window_i", 10.0)) / dt))
        self.ref_e: deque[float] = deque(maxlen=self.n_e)
        self.ref_i: deque[float] = deque(maxlen=self.n_i)
        self.r = 0.0
        self.t_ep = 0.0

    @staticmethod
    def ref_stats(window: deque[float]) -> tuple[float, float]:
        """Sliding-window reference statistics; N(0, 1) until two samples exist (EQ-STD-1)."""
        n = len(window)
        if n < 2:
            return 0.0, 1.0
        mu = sum(window) / n
        return mu, math.sqrt(sum((w - mu) ** 2 for w in window) / n)


def _rng_state(rng: Any) -> Any:
    bg = getattr(rng, "bit_generator", None)
    return None if bg is None else bg.state


def _restore_rng(rng: Any, state: Any) -> None:
    if state is not None:
        rng.bit_generator.state = state


def step(
    pipe: Any, x_e: float, x_i: float, x_hat_e: float | None = None, x_hat_i: float | None = None
) -> dict:
    """One canonical timestep on an APGIPipeline instance `pipe` (SSOT §8).

    TC-13 (SSOT §23): the step is computed into locals and committed only if every new state
    value is finite. An overflow (OverflowError, or a non-finite result from finite inputs)
    raises NonFiniteStateError and leaves the pipeline, its RNG and its history unchanged: the
    same atomic contract APGIPipeline.step gives non-finite inputs (AUD-DEF-01). Division by
    zero cannot occur: every denominator is σ_ref + ε_stab or Σ + ε_stab with σ_ref, Σ ≥ 0
    (apgi-lean: Apgi.traj_denoms_pos).
    """
    rng_state = _rng_state(pipe.rng)
    try:
        new, B, out = _compute(pipe, x_e, x_i, x_hat_e, x_hat_i)
    except OverflowError as exc:
        _restore_rng(pipe.rng, rng_state)
        raise NonFiniteStateError(f"ssot_core step overflowed: {exc}") from exc
    bad = sorted(k for k, v in new.items() if not math.isfinite(v))
    if bad:
        _restore_rng(pipe.rng, rng_state)
        raise NonFiniteStateError(
            "ssot_core step would leave a non-finite state: " + ", ".join(bad)
        )
    st, core = pipe.state, pipe._ssot  # commit
    core.r, core.t_ep = new["r"], new["t_ep"]
    st.mu_e, st.mu_i, st.sigma2_e, st.sigma2_i = (
        new["mu_e"],
        new["mu_i"],
        new["sigma2_e"],
        new["sigma2_i"],
    )
    core.ref_e.append(new["eps_e"])
    core.ref_i.append(new["eps_i"])
    pipe.x_hat_e, pipe.x_hat_i = new["x_hat_e"], new["x_hat_i"]
    pipe.S, pipe.theta, pipe.B_prev = new["S"], new["theta"], B  # 15
    pipe.t += new["dt"]
    pipe.history["S"].append(new["S"])
    pipe.history["theta"].append(new["theta"])
    pipe.history["B"].append(float(B))
    return out


def _compute(
    pipe: Any, x_e: float, x_i: float, x_hat_e: float | None, x_hat_i: float | None
) -> tuple[dict, int, dict]:
    """Steps 2-15 into locals; nothing on `pipe` is mutated except the RNG draw (restored on failure)."""
    cfg, st, core = pipe.config, pipe.state, pipe._ssot
    eps_stab, dt = float(cfg.get("eps", 1e-8)), float(cfg.get("dt", 0.01))
    pi_min, pi_max = float(cfg.get("pi_min", 0.1)), float(cfg.get("pi_max", 10.0))
    rho = float(cfg.get("reset_factor", cfg.get("RHO_RETAIN", 0.1)))
    if not 0.1 <= rho <= 0.5:  # fail closed (MathSpec §6, SSOT EQ-RESET-1 canonical band)
        raise ValueError(f"reset_factor (ρ_retain) = {rho} must be in [0.1, 0.5]")
    xh_e = pipe.x_hat_e if x_hat_e is None else x_hat_e
    xh_i = pipe.x_hat_i if x_hat_i is None else x_hat_i
    eps_e, eps_i = x_e - xh_e, x_i - xh_i  # 2
    (me, se), (mi, si) = core.ref_stats(core.ref_e), core.ref_stats(core.ref_i)  # 3
    z_e = (eps_e - me) / (se + eps_stab)
    z_i = (eps_i - mi) / (si + eps_stab)
    pi_e = clip(pi_min, pi_max, 1.0 / (st.sigma2_e + eps_stab))  # 4
    pi_i = clip(pi_min, pi_max, 1.0 / (st.sigma2_i + eps_stab))
    gamma_ne = float(cfg.get("gamma_ne") or 0.0)  # 5
    g_ne = float(cfg.get("g_ne", 1.0))
    if gamma_ne:
        g_ne = ne_clamp(gamma_ne, g_ne)
    pi_e_eff = clip(pi_min, pi_max, float(cfg.get("g_ach", 1.0)) * pi_e)
    gain_i = g_ne if cfg.get("ne_on_precision", True) else 1.0
    M = float(getattr(pipe, "M", 0.0) or 0.0)
    pi_i_eff = clip(pi_min, pi_max, gain_i * pi_i * math.exp(float(cfg.get("beta_sm", 0.0)) * M))
    z_i_eff = z_i + float(cfg.get("beta", 0.0))
    pr_e = pi_e_eff / float(cfg.get("pi_bar_e", 1.0))
    pr_i = pi_i_eff / float(cfg.get("pi_bar_i", 1.0))
    s_inst = pr_e * (abs(z_e) - M0) + pr_i * (abs(z_i_eff) - M0)  # 6
    ne_factor = 1.0 + gamma_ne * g_ne if cfg.get("ne_on_threshold", False) else 1.0  # 7
    u_max, tau_u = float(cfg.get("u_max", 0.0)), float(cfg.get("tau_u", 1.0))
    u = u_max * -math.expm1(-core.t_ep / tau_u) if u_max else 0.0
    theta = pipe.theta
    theta_eff = theta * ne_factor + core.r + float(cfg.get("beta_5ht", 0.0)) - u
    S = pipe.S
    gamma_sig = 1.0 / float(cfg.get("ignite_tau", cfg.get("tau_sigma", 0.5)))
    margin = S - theta_eff
    b_prop = sigmoid(gamma_sig * margin)  # 8
    if cfg.get("stochastic_ignition", True):  # 9
        p_hit = -math.expm1(-float(cfg.get("r_max", 100.0)) * b_prop * dt)
        B = int(pipe.rng.random() < p_hit)
    else:
        p_hit = float(margin > 0)
        B = int(margin > 0)
    C = float(cfg.get("c1", 0.0)) * S + float(cfg.get("c2", 0.0)) * pipe.B_prev  # 10
    V = float(cfg.get("v1", 0.0)) * abs(z_e) + float(cfg.get("v2", 0.0)) * abs(z_i_eff)
    theta0 = float(cfg.get("theta_base", cfg.get("theta_0", 0.35)))
    pre = theta + float(cfg.get("eta", 0.0)) * (C - V) + float(cfg.get("delta", 0.0)) * B
    theta_next = max(
        float(cfg.get("theta_clip_min", 0.0)),
        theta0 + (pre - theta0) * math.exp(-dt / float(cfg.get("tau_theta", 20.0))),
    )
    r_next = (
        core.r * math.exp(-dt / float(cfg.get("tau_ref", 0.3)))
        + float(cfg.get("delta_ref", 0.0)) * B
    )  # 11
    S_at_ignition = S
    if B:  # 12
        S = rho * S
        t_ep = 0.0
    else:
        t_ep = core.t_ep + dt
    lam = float(cfg["lam"]) if "lam" in cfg else -math.expm1(-dt / float(cfg.get("tau_s", 0.3)))
    S = (1.0 - lam) * S + lam * s_inst  # 13
    a_e = (
        float(cfg["alpha_e"])
        if "alpha_e" in cfg
        else ema_alpha(dt, float(cfg.get("T_window_e", 5.0)))
    )  # 14
    a_i = (
        float(cfg["alpha_i"])
        if "alpha_i" in cfg
        else ema_alpha(dt, float(cfg.get("T_window_i", 10.0)))
    )
    mu_e0, mu_i0 = st.mu_e, st.mu_i
    mu_e = (1 - a_e) * mu_e0 + a_e * eps_e
    mu_i = (1 - a_i) * mu_i0 + a_i * eps_i
    s2_e = (1 - a_e) * st.sigma2_e + a_e * (eps_e - mu_e0) ** 2
    s2_i = (1 - a_i) * st.sigma2_i + a_i * (eps_i - mu_i0) ** 2
    xhat_e, xhat_i = pipe.x_hat_e, pipe.x_hat_i
    if cfg.get("use_internal_predictions", True) and x_hat_e is None:
        xhat_e = xhat_e + float(cfg.get("eta_pred_e", 0.0)) * pi_e * eps_e
    if cfg.get("use_internal_predictions", True) and x_hat_i is None:
        xhat_i = xhat_i + float(cfg.get("eta_pred_i", 0.0)) * pi_i * eps_i
    new = {
        "S": S,
        "theta": theta_next,
        "r": r_next,
        "t_ep": t_ep,
        "mu_e": mu_e,
        "mu_i": mu_i,
        "sigma2_e": s2_e,
        "sigma2_i": s2_i,
        "eps_e": eps_e,
        "eps_i": eps_i,
        "x_hat_e": xhat_e,
        "x_hat_i": xhat_i,
        "dt": dt,
    }
    return (
        new,
        B,
        {
            "model_form": "ssot_core",
            "z_e": eps_e,
            "z_i": eps_i,
            "z_e_norm": z_e,
            "z_i_norm": z_i,
            "z_i_eff": z_i_eff,
            "mu_e": mu_e,
            "mu_i": mu_i,
            "pi_e": pi_e,
            "pi_i": pi_i,
            "pi_e_eff": pi_e_eff,
            "pi_i_eff": pi_i_eff,
            "pi_rel_e": pr_e,
            "pi_rel_i": pr_i,
            "S_inst": s_inst,
            "S": S,
            "S_decided": S_at_ignition,
            "C": C,
            "V": V,
            "theta": theta_next,
            "theta_decided": theta,
            "theta_eff": theta_eff,
            "r_refractory": r_next,
            "ignition_margin": margin,
            "B_prop": b_prop,
            "p_ignite": p_hit,
            "B": B,
            "theta_dot": 0.0,
            "x_hat_e": xhat_e,
            "x_hat_i": xhat_i,
            "M_somatic": M,
            "metabolic_state": getattr(pipe, "metabolic_state", None),
        },
    )
