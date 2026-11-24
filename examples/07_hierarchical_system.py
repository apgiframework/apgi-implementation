"""Example: Hierarchical Multi-Timescale APGI System
Demonstrates the full hierarchical APGI system with:
- Per-level error computation at different timescales
- Precision coupling ODE across levels
- Phase-amplitude coupling for threshold modulation
- Multi-scale signal aggregation
- Spectral validation (1/f signature)
Spec References:
- §7: Hierarchical Multi-Timescale Architecture
- §8: Oscillatory Phase Coupling
- §12: Statistical Validation — Spectral Signatures
"""

from __future__ import annotations

# flake8: noqa=E402 (module level import not at top of file - needed for sys.path manipulation)
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure matplotlib with centralized font fallbacks FIRST
from core.matplotlib_config import configure_matplotlib_fonts, suppress_font_warnings

configure_matplotlib_fonts(use_tex=False, font_family="sans-serif")
suppress_font_warnings()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pipeline import APGIPipeline
from stats.spectral_model import validate_pink_noise


def generate_hierarchical_input(
    n_steps: int = 10000,
    dt: float = 1.0,
    noise_level: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sensory input with hierarchical structure.
    Args:
        n_steps: Number of timesteps
        dt: Sampling interval (ms)
        noise_level: Input noise amplitude
    Returns:
        (x_e, x_i) — Exteroceptive and interoceptive inputs
    """
    t = np.arange(n_steps) * dt
    # Exteroceptive input: multi-scale oscillations
    x_e = (
        np.sin(2 * np.pi * 0.01 * t)  # Slow oscillation (100ms period)
        + 0.5 * np.sin(2 * np.pi * 0.05 * t)  # Medium oscillation (20ms period)
        + 0.25 * np.sin(2 * np.pi * 0.1 * t)  # Fast oscillation (10ms period)
        + noise_level * np.random.randn(n_steps)
    )
    # Interoceptive input: slower modulation
    x_i = (
        0.5 * np.sin(2 * np.pi * 0.005 * t)  # Very slow (200ms period)
        + 0.3 * np.sin(2 * np.pi * 0.02 * t)  # Slow (50ms period)
        + noise_level * np.random.randn(n_steps)
    )
    return x_e, x_i


def run_hierarchical_simulation(
    hierarchical_mode: str = "full",
    n_steps: int = 10000,
    plot: bool = True,
) -> dict:
    """Run hierarchical APGI simulation.
    Args:
        hierarchical_mode: 'off', 'basic', 'advanced', or 'full'
        n_steps: Number of timesteps
        plot: Whether to plot results
    Returns:
        Dictionary with simulation results
    """
    print(f"\n{'=' * 70}")
    print(f"Hierarchical APGI Simulation: {hierarchical_mode.upper()}")
    print(f"{'=' * 70}\n")
    # Configuration
    # Hierarchical modes use phase-modulated thresholds (θ_l = θ_0·(1+κ·Π·cos(φ))).
    # The phase dips lower the effective threshold → ~2× more ignitions than OFF mode
    # at the same θ_0.  Raise θ_0 for hierarchical modes so ignition rate stays ~6-8%
    # (comparable to OFF), which keeps β_spec in the pink-noise band [0.7, 1.3].
    _theta = 0.3 if hierarchical_mode == "off" else 0.55
    config = {
        # Initial states
        "S0": 0.0,
        "theta_0": _theta,
        "theta_base": _theta,
        "sigma2_e0": 1.0,
        "sigma2_i0": 1.0,
        # Numerical stability
        "eps": 1e-8,
        "pi_min": 1e-4,
        "pi_max": 1e4,
        # Signal accumulation
        "alpha_e": 0.2,
        "alpha_i": 0.2,
        "lam": 0.1,
        "eta": 0.01,
        "delta": 0.5,
        "tau_theta": 0.02,  # example-specific: per-step decay exponent dt/τ_θ = 0.1 at dt = 2 ms
        # Cost-value
        "use_realistic_cost": True,
        "c1": 0.1,
        "c2": 0.1,  # Lower ignition cost
        "v1": 1.0,  # Increased for stronger information value
        "v2": 1.0,
        # Ignition
        "ignite_tau": 0.3,  # Sharper ignition transition
        "stochastic_ignition": False,
        # Hierarchical mode
        "hierarchical_mode": hierarchical_mode,
        "n_levels": 4,
        "tau_0": 5.0,  # Lower base timescale for better fast dynamics
        "k": 1.8,  # Increased separation for better 1/f slope
        # Neuromodulation
        "g_ach": 1.0,
        "g_ne": 1.0,
        "beta": 0.0,
        # Precision ODE (if advanced/full)
        "tau_pi": 1000.0,
        "alpha_pi": 0.1,
        "C_down": 0.1,
        "C_up": 0.05,
        # Phase coupling (if full)
        "kappa_down": 0.1,
        "kappa_up": 0.05,
        "kappa_phase": 0.03,  # Further reduced for more ignitions
        "omega_phases": [0.1, 0.05, 0.025, 0.0125],  # Must match n_levels
        # Post-ignition reset
        "reset_factor": 0.5,
        # Signal ODE
        "tau_s": 0.15,  # Balanced for universal 1/f validation
        "dt": 0.002,  # Finer step for numerical stability
        "signal_log_nonlinearity": True,
    }
    # Create pipeline
    pipeline = APGIPipeline(config)
    # Generate input
    x_e, x_i = generate_hierarchical_input(n_steps=n_steps)
    # Simple generative model (constant predictions)
    x_hat_e = np.mean(x_e)
    x_hat_i = np.mean(x_i)
    # Run simulation
    outputs = []
    print(f"Running {n_steps} timesteps...")
    for t in range(n_steps):
        output = pipeline.step(x_e[t], x_i[t], x_hat_e, x_hat_i)
        outputs.append(output)
        if (t + 1) % 2000 == 0:
            print(f"  Step {t + 1}/{n_steps}")
    print("✅ Simulation complete\n")
    # Extract results — including precision variables Πᵉ and Πⁱ_eff (paper §4.2)
    results = {
        "config": config,
        "outputs": outputs,
        "x_e": x_e,
        "x_i": x_i,
        "S": np.array([o["S"] for o in outputs]),
        "theta": np.array([o["theta"] for o in outputs]),
        "B": np.array([o["B"] for o in outputs]),
        "ignition_margin": np.array([o["ignition_margin"] for o in outputs]),
        "p_ignite": np.array([o["p_ignite"] for o in outputs]),
        "pi_e_eff": np.array([o.get("pi_e_eff", o.get("pi_e", 1.0)) for o in outputs]),
        "pi_i_eff": np.array([o.get("pi_i_eff", o.get("pi_i", 1.0)) for o in outputs]),
    }
    # Validate spectral signature using correct sampling frequency (paper §12)
    print("Validating spectral signature...")
    from scipy.signal import welch  # type: ignore[import-untyped]

    _dt = float(config.get("dt", 1.0))
    _fs = 1.0 / _dt  # Convert dt (seconds) to sampling frequency (Hz)
    freqs, power = welch(results["S"], fs=_fs, nperseg=512)
    spectral_result = validate_pink_noise(freqs, power)
    results["spectral_result"] = spectral_result
    print(spectral_result.get("message", "Validation complete"))
    print()
    # Statistics
    n_ignitions = int(np.sum(results["B"]))  # type: ignore[call-overload]
    ignition_rate = n_ignitions / n_steps * 100
    print("Statistics:")
    print(f"  Ignitions: {n_ignitions} ({ignition_rate:.1f}%)")
    print(f"  Signal range: [{results['S'].min():.3f}, {results['S'].max():.3f}]")  # type: ignore[attr-defined]
    print(
        f"  Threshold range: [{results['theta'].min():.3f}, {results['theta'].max():.3f}]"  # type: ignore[attr-defined]
    )
    print(
        f"  Margin range: [{results['ignition_margin'].min():.3f}, {results['ignition_margin'].max():.3f}]"  # type: ignore[attr-defined]
    )
    print()
    # Plot if requested
    if plot:
        plot_results(results, hierarchical_mode)
    return results


def plot_results(results: dict, title: str = "", save_path: Path | None = None) -> None:
    """Plot simulation results with calibrated time axis and precision traces.

    Five panels matching the paper's observable variables:
      1. Sₜ and θₜ (signal vs threshold, paper Eq. 1 and 3)
      2. Πᵉ and Πⁱ_eff (precision weighting, paper §4.2)
      3. Ignition margin Δ(t) = Sₜ − θₜ
      4. P(ignition|Sₜ,θₜ) with ignition events (paper Eq. 2)
      5. Power spectrum with 1/fᵝ reference (paper §7)

    Args:
        results: Dictionary with simulation results
        title: Plot title
        save_path: If provided, save plot to this path instead of showing
    """
    _dt = float(results["config"].get("dt", 1.0))
    _fs = 1.0 / _dt
    n = len(results["S"])
    t_ms = np.arange(n) * _dt * 1000.0  # calibrated time axis in milliseconds

    fig, axes = plt.subplots(5, 1, figsize=(14, 14))
    fig.suptitle(f"Hierarchical APGI: {title.upper()}", fontsize=14, fontweight="bold")

    # Panel 1: Signal and Threshold
    ax = axes[0]
    ax.plot(t_ms, results["S"], label="Sₜ (signal)", linewidth=1, alpha=0.8)
    ax.plot(t_ms, results["theta"], label="θₜ (threshold)", linewidth=1, alpha=0.8)
    ax.fill_between(
        t_ms,
        results["S"],
        results["theta"],
        where=(results["S"] > results["theta"]),
        alpha=0.3,
        color="red",
        label="Superthreshold",
    )
    ax.set_ylabel("Amplitude (AU)")
    ax.set_xlabel("Time (ms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Signal Sₜ and Dynamic Threshold θₜ  (paper Eq. 1, 3)")

    # Panel 2: Effective precisions Πᵉ and Πⁱ_eff (paper §4.2)
    ax = axes[1]
    ax.plot(
        t_ms,
        results["pi_e_eff"],
        label="Πᵉ (exteroceptive precision)",
        linewidth=1,
        alpha=0.8,
        color="steelblue",
    )
    ax.plot(
        t_ms,
        results["pi_i_eff"],
        label="Πⁱ_eff (interoceptive precision)",
        linewidth=1,
        alpha=0.8,
        color="darkorange",
    )
    ax.set_ylabel("Precision (AU)")
    ax.set_xlabel("Time (ms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Precision Weights Πᵉ and Πⁱ_eff  (paper §4.2)")

    # Panel 3: Ignition Margin
    ax = axes[2]
    ax.plot(t_ms, results["ignition_margin"], label="Δ(t) = Sₜ − θₜ", linewidth=1)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.fill_between(
        t_ms,
        0,
        results["ignition_margin"],
        where=(results["ignition_margin"] > 0),
        alpha=0.3,
        color="green",
        label="Superthreshold region",
    )
    ax.set_ylabel("Margin (AU)")
    ax.set_xlabel("Time (ms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Ignition Margin Δ(t)")

    # Panel 4: Ignition Probability and Events (paper Eq. 2)
    ax = axes[3]
    ax.plot(t_ms, results["p_ignite"], label="P(ignition|Sₜ,θₜ)", linewidth=1, alpha=0.7)
    ignition_times_ms = t_ms[results["B"] == 1]
    ax.scatter(
        ignition_times_ms,
        np.ones(len(ignition_times_ms)),
        color="red",
        s=50,
        marker="v",
        label="Ignition events Bₜ=1",
        zorder=5,
    )
    ax.set_ylabel("P(ignition)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylim([-0.1, 1.1])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Ignition Probability P(Bₜ=1|Sₜ,θₜ)  (paper Eq. 2, γ_sig=1/τ_σ)")

    # Panel 5: Power Spectrum with correct fs and 1/fᵝ reference
    ax = axes[4]
    from scipy.signal import welch  # type: ignore[import-untyped]

    freqs, power = welch(results["S"], fs=_fs, nperseg=512)
    ax.loglog(freqs[1:], power[1:], "b-", linewidth=2, label="Observed PSD")
    f_ref = freqs[len(freqs) // 2]
    p_ref = power[len(power) // 2]
    f_theory = np.logspace(np.log10(freqs[1]), np.log10(freqs[-1]), 200)
    p_theory = p_ref * (f_theory / f_ref) ** (-results["spectral_result"]["beta"])
    ax.loglog(
        f_theory,
        p_theory,
        "r--",
        linewidth=2,
        label=f"1/fᵝ fit  β={results['spectral_result']['beta']:.2f}",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(
        f"Power Spectrum  H={results['spectral_result']['hurst_exponent']:.2f}  (paper §7 Lorentzian superposition)"
    )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def compare_modes(save_path: Path | None = None) -> None:
    """Compare different hierarchical modes.
    Demonstrates the effect of hierarchical_mode on system behavior.

    Args:
        save_path: If provided, save comparison plot to this path instead of showing
    """
    print("\n" + "=" * 70)
    print("COMPARING HIERARCHICAL MODES")
    print("=" * 70)
    modes = ["off", "basic", "advanced", "full"]
    results_dict = {}
    for mode in modes:
        results = run_hierarchical_simulation(
            hierarchical_mode=mode,
            n_steps=5000,
            plot=False,
        )
        results_dict[mode] = results
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70 + "\n")
    print(f"{'Mode':<12} {'β_spec':<10} {'H':<10} {'Valid':<8} {'Ignitions':<12}")
    print("-" * 70)
    for mode in modes:
        result = results_dict[mode]["spectral_result"]
        n_ignitions = int(np.sum(results_dict[mode]["B"]))  # type: ignore[call-overload]
        print(
            f"{mode:<12} {result['beta']:<10.2f} "
            f"{result['hurst_exponent']:<10.2f} "
            f"{'✅' if result.get('is_pink_noise', False) else '❌':<8} "
            f"{n_ignitions:<12}"
        )
    print()
    # Plot comparison — calibrated time axis in ms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Hierarchical Mode Comparison", fontsize=14, fontweight="bold")
    for idx, mode in enumerate(modes):
        ax = axes[idx // 2, idx % 2]
        results = results_dict[mode]
        _dt = float(results["config"].get("dt", 1.0))
        window = min(2000, len(results["S"]))
        t_ms = np.arange(window) * _dt * 1000.0
        ax.plot(t_ms, results["S"][:window], label="Sₜ", linewidth=1, alpha=0.8)
        ax.plot(t_ms, results["theta"][:window], label="θₜ", linewidth=1, alpha=0.8)
        ax.set_title(f"Mode: {mode.upper()}  β={results['spectral_result']['beta']:.2f}")
        ax.set_ylabel("Amplitude (AU)")
        ax.set_xlabel("Time (ms)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    """Main function that saves plots to outputs directory."""
    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("HIERARCHICAL APGI SYSTEM EXAMPLES")
    print("=" * 70)

    # Example 1: Single-scale (baseline)
    results_off = run_hierarchical_simulation(
        hierarchical_mode="off",
        n_steps=5000,
        plot=False,
    )
    plot_results(
        results_off,
        "off",
        save_path=out_dir / "figure_07_off.png",
    )

    # Example 2: Basic hierarchical
    results_basic = run_hierarchical_simulation(
        hierarchical_mode="basic",
        n_steps=5000,
        plot=False,
    )
    plot_results(
        results_basic,
        "basic",
        save_path=out_dir / "figure_07_basic.png",
    )

    # Example 3: Full hierarchical
    results_full = run_hierarchical_simulation(
        hierarchical_mode="full",
        n_steps=5000,
        plot=False,
    )
    plot_results(
        results_full,
        "full",
        save_path=out_dir / "figure_07_full.png",
    )

    # Example 4: Compare all modes
    compare_modes(save_path=out_dir / "figure_07_comparison.png")

    print("\n" + "=" * 70)
    print("✅ Examples complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
