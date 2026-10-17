"""APGI (Allostatic Precision-Gated Ignition) - Main Entry Point
Available Options:
    --demo, --steps, --output, --multiscale, --levels, --k
    --ne-on-threshold, --gamma-ne, --beta, --stochastic, --seed
    --log-level, --json-logs, --max-history, --strict-mode
Example usage:
    python main.py --steps 1000 --output results.json
    python main.py --demo  # Run demonstration with synthetic data
    python main.py --multiscale --levels 5  # Enable multi-scale hierarchy
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from config import CONFIG
from core.logging_config import configure_logging, get_logger
from core.phi_transform import phi_transform_array
from core.rng import RNGLike, resolve_rng, seed_everything
from hierarchy.multiscale import build_timescales, multiscale_weights
from pipeline import APGIPipeline
from stats.hurst import estimate_hurst_robust

# Release version; keep in step with pyproject.toml and CITATION.cff.
APGI_VERSION = "1.1.0"

# Initialize logger at module level
logger = get_logger("apgi.main")


def generate_synthetic_input(
    t: int, noise_std: float = 0.1, rng: RNGLike = None
) -> tuple[float, float, float, float]:
    """Generate synthetic exteroceptive and interoceptive signals.

    Args:
        t: Timestep index.
        noise_std: Standard deviation of the additive observation noise.
        rng: Generator/seed; ``None`` uses the process-global APGI generator,
            which ``--seed`` makes deterministic.

    Returns:
        (x_e, x_hat_e, x_i, x_hat_i) - actual and predicted values
    """
    generator = resolve_rng(rng)
    # Exteroceptive: external sensory signal with periodic component
    x_e = np.sin(0.05 * t) + 0.5 * np.sin(0.01 * t) + generator.normal(0, noise_std)
    x_hat_e = np.sin(0.05 * t)  # Predictor sees only fast component
    # Interoceptive: internal signal (e.g., heart rate variability)
    x_i = 0.5 + 0.3 * np.cos(0.02 * t) + generator.normal(0, noise_std * 0.5)
    x_hat_i = 0.5  # Baseline prediction
    return float(x_e), float(x_hat_e), float(x_i), float(x_hat_i)


def run_standard_pipeline(
    n_steps: int = 1000,
    config: dict[str, Any] | None = None,
    progress_interval: int = 100,
    max_history: int | None = None,
) -> dict[str, Any]:
    """Run standard single-scale APGI pipeline.
    Args:
        n_steps: Number of simulation steps
        config: Optional custom configuration (uses default CONFIG if None)
        progress_interval: Print progress every N steps
        max_history: Maximum history size to prevent unbounded memory growth (None = unlimited)
    Returns:
        Dictionary with simulation results and statistics
    """
    cfg = config or CONFIG.copy()
    pipeline = APGIPipeline(cfg)
    # Pre-allocate history with optional bounded memory
    history: dict[str, list[float] | deque[float]]
    if max_history is None or n_steps <= max_history:
        history = {
            "S": [],
            "theta": [],
            "B": [],
            "z_e": [],
            "z_i": [],
            "p_ignite": [],
            "C": [],
            "V": [],
        }
        use_ring_buffer = False
    else:
        from collections import deque

        history = {
            "S": deque(maxlen=max_history),
            "theta": deque(maxlen=max_history),
            "B": deque(maxlen=max_history),
            "z_e": deque(maxlen=max_history),
            "z_i": deque(maxlen=max_history),
            "p_ignite": deque(maxlen=max_history),
            "C": deque(maxlen=max_history),
            "V": deque(maxlen=max_history),
        }
        use_ring_buffer = True
    logger.info(
        "starting_standard_pipeline",
        n_steps=n_steps,
        max_history=max_history,
        initial_threshold=pipeline.theta,
        initial_signal=pipeline.S,
    )
    ignition_count = 0
    for t in range(n_steps):
        x_e, x_hat_e, x_i, x_hat_i = generate_synthetic_input(t)
        result = pipeline.step(x_e, x_i, x_hat_e, x_hat_i)
        for key in history:
            history[key].append(result[key])
        if result["B"] == 1:
            ignition_count += 1
        if (t + 1) % progress_interval == 0:
            logger.debug(
                "step_progress",
                step=t + 1,
                total_steps=n_steps,
                signal=result["S"],
                threshold=result["theta"],
                ignite_probability=result["p_ignite"],
                ignition=result["B"],
            )
    logger.info(
        "pipeline_completed",
        ignition_count=ignition_count,
        ignition_rate=ignition_count / n_steps,
        final_signal=pipeline.S,
        final_threshold=pipeline.theta,
    )
    # Convert deque to list if using ring buffer
    if use_ring_buffer:
        history = {k: list(v) for k, v in history.items()}
    return {
        "config": cfg,
        "n_steps": n_steps,
        "ignition_count": ignition_count,
        "ignition_rate": ignition_count / n_steps,
        "history": history,
        "final_state": {
            "S": pipeline.S,
            "theta": pipeline.theta,
            "sigma2_e": pipeline.state.sigma2_e,
            "sigma2_i": pipeline.state.sigma2_i,
        },
    }


def run_multiscale_pipeline(
    n_steps: int = 1000,
    n_levels: int = 5,
    timescale_k: float = 1.6,
    config: dict[str, Any] | None = None,
    max_history: int | None = None,
) -> dict[str, Any]:
    """Run multi-scale APGI pipeline (hierarchical timescales).
    Args:
        n_steps: Number of simulation steps
        n_levels: Number of timescale hierarchy levels
        timescale_k: Timescale expansion factor (recommended: 1.3-2.0)
        config: Optional custom configuration
        max_history: Maximum history size to prevent unbounded memory growth (None = unlimited)
    Returns:
        Dictionary with simulation results
    """
    cfg = config or CONFIG.copy()
    cfg["timescale_k"] = timescale_k
    tau0 = 10.0  # Spec appendix: τ_0 > 1; canonical default matches CONFIG
    taus = build_timescales(tau0, timescale_k, n_levels)
    weights = multiscale_weights(n_levels, timescale_k)
    logger.info(
        "starting_multiscale_pipeline",
        n_steps=n_steps,
        n_levels=n_levels,
        timescale_k=timescale_k,
        timescales=taus.tolist(),
        weights=weights.tolist(),
        max_history=max_history,
    )
    # Initialize multi-scale feature buffers and per-level EMA variances
    phi_e = np.zeros(n_levels)
    phi_i = np.zeros(n_levels)
    sigma2_levels = np.ones(n_levels)
    # Pre-compute EMA rates for vectorized updates
    alphas = np.minimum(1.0, 1.0 / taus)
    pipeline = APGIPipeline(cfg)
    # Pre-allocate history arrays for better memory efficiency
    history: dict[str, list[float] | deque[float]]
    if max_history is None or n_steps <= max_history:
        history = {
            "S_multiscale": [],
            "S_standard": [],
            "B": [],
            "theta": [],
        }
        use_ring_buffer = False
    else:
        # Use ring buffer for bounded memory
        from collections import deque

        history = {
            "S_multiscale": deque(maxlen=max_history),
            "S_standard": deque(maxlen=max_history),
            "B": deque(maxlen=max_history),
            "theta": deque(maxlen=max_history),
        }
        use_ring_buffer = True
    ignition_count = 0
    # Pre-extract config values for performance
    eps = float(cfg.get("eps", 1e-8))  # type: ignore[arg-type]
    pi_min = float(cfg.get("pi_min", 0.01))  # type: ignore[arg-type]
    pi_max = float(cfg.get("pi_max", 10.0))  # type: ignore[arg-type]
    for t in range(n_steps):
        x_e, x_hat_e, x_i, x_hat_i = generate_synthetic_input(t)
        z_e = x_e - x_hat_e
        z_i = x_i - x_hat_i
        # Vectorized multi-scale feature updates (much faster than loop)
        # phi_e[i] = (1 - alphas[i]) * phi_e[i] + alphas[i] * z_e
        phi_e = (1.0 - alphas) * phi_e + alphas * z_e
        phi_i = (1.0 - alphas) * phi_i + alphas * z_i
        # Vectorized variance updates
        sigma2_levels = (1.0 - alphas) * sigma2_levels + alphas * (phi_e**2 + phi_i**2)
        # Vectorized precision computation
        pi_levels = np.clip(1.0 / (sigma2_levels + eps), pi_min, pi_max)
        # Aggregate signal using φ(ε) — asymmetric valence-specific transform (§6)
        _a_pos = float(cfg.get("alpha_plus", 1.0))  # type: ignore[arg-type]
        _a_neg = float(cfg.get("alpha_minus", 1.0))  # type: ignore[arg-type]
        _g_pos = float(cfg.get("gamma_plus", 2.0))  # type: ignore[arg-type]
        _g_neg = float(cfg.get("gamma_minus", 2.0))  # type: ignore[arg-type]
        phi_combined = phi_transform_array(
            phi_e, _a_pos, _a_neg, _g_pos, _g_neg
        ) + phi_transform_array(phi_i, _a_pos, _a_neg, _g_pos, _g_neg)
        S_multiscale = float(np.sum(weights * pi_levels * phi_combined))
        # Run standard pipeline for comparison
        result = pipeline.step(x_e, x_i, x_hat_e, x_hat_i)
        # Store results
        history["S_multiscale"].append(S_multiscale)
        history["S_standard"].append(result["S"])
        history["B"].append(result["B"])
        history["theta"].append(result["theta"])
        if result["B"] == 1:
            ignition_count += 1
        # Periodic progress logging for long runs
        if (t + 1) % 1000 == 0:
            logger.debug("multiscale_progress", step=t + 1, total=n_steps)
    # Convert deque to list for output if using ring buffer
    if use_ring_buffer:
        history = {k: list(v) for k, v in history.items()}
    logger.info(
        "multiscale_pipeline_completed",
        n_steps=n_steps,
        ignition_count=ignition_count,
        ignition_rate=ignition_count / n_steps,
        multiscale_s_range=[min(history["S_multiscale"]), max(history["S_multiscale"])],
        standard_s_range=[min(history["S_standard"]), max(history["S_standard"])],
    )
    return {
        "config": cfg,
        "n_steps": n_steps,
        "n_levels": n_levels,
        "timescales": taus.tolist(),
        "weights": weights.tolist(),
        "ignition_count": ignition_count,
        "history": history,
    }


def analyze_signal_statistics(
    signal_history: list[float], label: str = "Signal"
) -> dict[str, float]:
    """Compute statistics and estimate Hurst exponent from signal history.
    Args:
        signal_history: Time series of signal values
        label: Label for print output
    Returns:
        Dictionary with computed statistics
    """
    signal = np.array(signal_history)
    stats = {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "range": float(np.max(signal) - np.min(signal)),
    }
    # Estimate Hurst exponent if enough data
    if len(signal) >= 256:
        try:
            H = estimate_hurst_robust(signal, fs=1.0, method="welch")
            stats["hurst_exponent"] = H
            logger.info(
                "signal_statistics",
                label=label,
                mean=stats["mean"],
                std=stats["std"],
                range_min=stats["min"],
                range_max=stats["max"],
                hurst_exponent=H,
                process_type=(
                    "persistent" if H > 0.5 else "anti_persistent" if H < 0.5 else "random_walk"
                ),
            )
        except Exception as e:
            logger.warning("hurst_estimation_failed", label=label, error=str(e))
    else:
        logger.info(
            "signal_statistics_insufficient_data",
            label=label,
            data_points=len(signal),
            required_points=256,
            **stats,
        )
    return stats


def save_results(results: dict[str, Any], filepath: str) -> None:
    """Save simulation results to JSON file.
    Args:
        results: Dictionary with simulation results
        filepath: Output file path
    """

    # Convert numpy arrays to lists for JSON serialization
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    serializable = convert(results)
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("results_saved", filepath=filepath)


def show_info() -> None:
    """Display system information and configuration."""
    print("\n" + "=" * 60)
    print("APGI: Allostatic Precision-Gated Ignition")
    print("=" * 60)
    print(f"\nVersion: {APGI_VERSION}")
    print("Description: Unified computational framework for modeling")
    print("             allostatic threshold dynamics in biological systems")
    print("\n" + "-" * 60)
    print("Default Configuration:")
    print("-" * 60)
    for key, value in CONFIG.items():
        print(f"  {key:25s} = {value}")
    print("\n" + "-" * 60)
    print("Available Features:")
    print("-" * 60)
    print("  ✅ Signal preprocessing (§1)")
    print("  ✅ Precision system (§2)")
    print("  ✅ Signal accumulation (§3)")
    print("  ✅ Allostatic threshold dynamics (§4)")
    print("  ✅ Hard and soft ignition (§5)")
    print("  ✅ Post-ignition reset (§6)")
    print("  ✅ Continuous-time SDE (§7)")
    print("  ✅ Hierarchical multi-timescale (§8)")
    print("  ✅ Kuramoto oscillators (§9)")
    print("  ✅ Liquid state machine (§10)")
    print("  ✅ Thermodynamic constraints (§11)")
    print("  ✅ Observable mapping (§14)")
    print("  ✅ Fixed-point stability (§7)")
    print("\n" + "-" * 60)
    print("Examples:")
    print("-" * 60)
    print("  python main.py --demo")
    print("  python main.py --steps 1000 --output results.json")
    print("  python main.py --multiscale --levels 5")
    print("=" * 60 + "\n")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="APGI (Allostatic Precision-Gated Ignition) Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s info                      # Show system information
  %(prog)s --demo                    # Quick demonstration
  %(prog)s --steps 5000 --output out.json   # Long run with save
  %(prog)s --multiscale --levels 7   # Multi-scale hierarchy
  %(prog)s --ne-on-threshold         # Use NE on threshold (not precision)
        """,
    )
    parser.add_argument(
        "info",
        nargs="?",
        const=True,
        help="Show system information and configuration",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick demonstration (100 steps unless --steps is also given)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of simulation steps (default: 100 with --demo, else 1000)",
    )
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--multiscale", action="store_true", help="Enable multi-scale hierarchy")
    parser.add_argument(
        "--levels",
        type=int,
        default=5,
        help="Number of hierarchy levels (with --multiscale)",
    )
    parser.add_argument("--k", type=float, default=1.6, help="Timescale expansion factor (1.3-2.0)")
    parser.add_argument(
        "--ne-on-threshold",
        action="store_true",
        help="Apply NE to threshold instead of precision",
    )
    parser.add_argument(
        "--gamma-ne",
        type=float,
        default=None,
        help="NE modulation strength (default: 0.1, use <=0.01 with --ne-on-threshold)",
    )
    parser.add_argument("--beta", type=float, default=0.0, help="Dopaminergic bias value")
    parser.add_argument("--stochastic", action="store_true", help="Enable stochastic ignition")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Output logs in JSON format for production",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=None,
        help="Maximum history size to prevent unbounded memory growth",
    )
    parser.add_argument(
        "--strict-mode",
        action="store_true",
        default=True,
        help="Enable strict validation (no auto-adjustments)",
    )
    args = parser.parse_args()
    # --demo picks a small step count for a quick showcase run unless the
    # caller explicitly overrides --steps. Previously this flag was parsed
    # but never consulted (silent no-op — `--demo` ran the identical
    # 1000-step standard pipeline as no flags at all).
    if args.steps is None:
        args.steps = 100 if args.demo else 1000
    # Configure logging
    configure_logging(level=args.log_level, json_output=args.json_logs)
    # Seed every stochastic component from one place. np.random.seed() alone
    # was a no-op for ignition sampling (which used np.random.default_rng()),
    # so identical --seed values produced different ignition trains.
    seed_everything(args.seed)
    # Build custom config from defaults + CLI overrides
    config = CONFIG.copy()
    config["beta"] = args.beta
    config["stochastic_ignition"] = args.stochastic
    config["strict_mode"] = args.strict_mode
    config["strict"] = args.strict_mode
    config["seed"] = args.seed
    if args.ne_on_threshold:
        config["ne_on_precision"] = False
        config["ne_on_threshold"] = True
        # Auto-adjust gamma_ne to safe value for threshold mode if not explicitly set
        if args.gamma_ne is None:
            config["gamma_ne"] = 0.01
            logger.info(
                "configuration",
                ne_modulation="threshold",
                gamma_ne_adjusted=0.01,
                reason="threshold_mode_requires_gamma_ne<=0.01",
            )
        else:
            logger.info("configuration", ne_modulation="threshold", gamma_ne=args.gamma_ne)
    if args.gamma_ne is not None:
        config["gamma_ne"] = args.gamma_ne
    logger.info(
        "apgi_startup",
        version=APGI_VERSION,
        demo=args.demo,
        strict_mode=args.strict_mode,
        max_history=args.max_history,
    )
    try:
        if args.info:
            show_info()
            return 0
        if args.multiscale:
            results = run_multiscale_pipeline(
                n_steps=args.steps,
                n_levels=args.levels,
                timescale_k=args.k,
                config=config,
                max_history=args.max_history,
            )
            analyze_signal_statistics(results["history"]["S_multiscale"], "Multi-scale Signal")
        else:
            results = run_standard_pipeline(
                n_steps=args.steps,
                config=config,
                max_history=args.max_history,
            )
            analyze_signal_statistics(results["history"]["S"], "Standard Signal")
        if args.output:
            save_results(results, args.output)
        logger.info("simulation_complete", status="success")
        return 0
    except KeyboardInterrupt:
        logger.warning("simulation_interrupted", reason="user_interrupt")
        return 130
    except Exception as e:
        logger.error("simulation_failed", error=str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
