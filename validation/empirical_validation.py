"""Empirical dataset-driven validation for APGI.
Implements validation against real neural and behavioral datasets
to test APGI predictions beyond simulation/proxy metrics.
Spec §14: Observable mapping to neural and behavioral data
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.rng import RNGLike, resolve_rng

#: Extensions parsed by NumPy (no optional dependency needed).
NUMPY_EXTENSIONS: frozenset[str] = frozenset({".npy", ".npz"})
#: Extensions parsed by MNE (optional dependency).
MNE_EXTENSIONS: frozenset[str] = frozenset(
    {".fif", ".edf", ".bdf", ".gdf", ".set", ".vhdr", ".cnt", ".mff", ".nxe", ".egi"}
)


class DatasetLoadError(RuntimeError):
    """A dataset file exists in a supported format but could not be parsed."""


class OptionalDependencyError(ImportError):
    """The parser for a supported format is an optional dependency that is not installed."""


class UnsupportedFormatError(ValueError):
    """The file format is not one this loader supports."""


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and validation."""

    name: str
    data_type: str  # 'eeg', 'lfp', 'behavior', 'simulation'
    fs: float  # Sampling frequency
    duration: float | None = None
    n_trials: int | None = None
    condition_labels: list[str] | None = None


class EmpiricalDataLoader:
    """Load and preprocess empirical datasets for APGI validation.
    Supports multiple data formats:
    - EEG/MEG data (EDF, FIF, custom formats)
    - LFP data (NeuroExplorer, MATLAB)
    - Behavioral data (CSV, JSON, HDF5)
    - Simulation benchmarks (NPZ, HDF5)
    """

    def __init__(self, config: DatasetConfig):
        """Initialize data loader.
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.data: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self._behavioral_data: dict[str, Any] = {}

    def load_eeg_dataset(
        self,
        file_path: str,
        channel_names: list[str] | None = None,
        event_markers: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Load an EEG dataset, choosing the parser from the file format.

        Parser selection is explicit (AUD-DEF-12) rather than "try MNE, then
        anything":

        * ``.npy`` / ``.npz`` — NumPy, with ``allow_pickle=False`` (pickled
          object arrays can execute code and are refused as untrusted input).
        * MNE formats (:data:`MNE_EXTENSIONS`) — MNE. If MNE is not installed a
          :class:`OptionalDependencyError` is raised; there is no silent NumPy
          fallback, because NumPy cannot parse these formats and a "successful"
          fallback would only hide the real problem.
        * anything else — :class:`UnsupportedFormatError`.

        A parse failure raises :class:`DatasetLoadError` with the original
        exception chained, so a corrupt file is never confused with a missing
        optional dependency. The returned dict carries a ``provenance`` record
        (path, format, parser, SHA-256 of the file bytes).

        Args:
            file_path: Path to data file
            channel_names: List of channel names to load
            event_markers: Mapping of event names to marker codes
        Returns:
            Dictionary with signals, events, metadata and provenance
        """
        path = Path(file_path)
        ext = "".join(path.suffixes[-1:]).lower()
        if ext in NUMPY_EXTENSIONS:
            result = self._load_numpy_eeg(path)
            parser = "numpy"
        elif ext in MNE_EXTENSIONS:
            try:
                import mne  # type: ignore[import-not-found]
            except ImportError as exc:
                raise OptionalDependencyError(
                    f"{path.name}: format {ext!r} requires the optional dependency 'mne'"
                ) from exc
            try:
                result = self._load_mne_eeg(mne, path, channel_names, event_markers)
            except Exception as exc:
                raise DatasetLoadError(f"{path.name}: MNE could not parse the file: {exc}") from exc
            parser = "mne"
        else:
            raise UnsupportedFormatError(
                f"{path.name}: unsupported EEG format {ext!r}; supported: "
                f"{sorted(NUMPY_EXTENSIONS | MNE_EXTENSIONS)}"
            )
        result["provenance"] = {
            "path": str(path),
            "format": ext,
            "parser": parser,
            "sha256": _file_sha256(path) if path.is_file() else None,
        }
        self.data = result
        return result

    def _load_mne_eeg(
        self,
        mne: Any,
        path: Path,
        channel_names: list[str] | None,
        event_markers: dict[str, int] | None,
    ) -> dict[str, Any]:
        raw = mne.io.read_raw(str(path), preload=True)
        if channel_names:
            raw.pick_channels(channel_names)
        data = raw.get_data()
        sfreq = raw.info["sfreq"]
        result: dict[str, Any] = {
            "signals": data,
            "fs": sfreq,
            "channels": raw.ch_names,
            "duration": data.shape[1] / sfreq,
            "events": None,
        }
        if event_markers:
            events, event_dict = mne.events_from_annotations(raw)
            result["events"] = events
            result["event_dict"] = event_dict
        return result

    def _load_numpy_eeg(self, path: Path) -> dict[str, Any]:
        try:
            data = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            raise DatasetLoadError(f"{path.name}: NumPy could not parse the file: {exc}") from exc
        if isinstance(data, np.lib.npyio.NpzFile):
            sig_key = "signals" if "signals" in data.files else "arr_0"
            if sig_key not in data.files:
                raise DatasetLoadError(f"{path.name}: no 'signals' or 'arr_0' array")
            fs_val = float(data["fs"]) if "fs" in data.files else float(self.config.fs)
            ch_val = [str(c) for c in data["channels"]] if "channels" in data.files else []
            return {"signals": data[sig_key], "fs": fs_val, "channels": ch_val}
        return {
            "signals": data,
            "fs": self.config.fs,
            "channels": [f"ch_{i}" for i in range(data.shape[0])],
        }

    def load_behavioral_dataset(
        self,
        file_path: str,
        rt_column: str = "rt",
        accuracy_column: str = "accuracy",
        condition_column: str | None = None,
    ) -> dict[str, Any]:
        """Load behavioral dataset (reaction times, accuracy).
        Args:
            file_path: Path to CSV/JSON/HDF5 file
            rt_column: Column name for reaction times
            accuracy_column: Column name for accuracy
            condition_column: Column name for experimental condition
        Returns:
            Dictionary with behavioral metrics
        """
        import pandas as pd  # type: ignore[import-untyped]

        df: pd.DataFrame | pd.Series[Any]
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path)
        elif file_path.endswith(".h5") or file_path.endswith(".hdf5"):
            df = pd.read_hdf(file_path)
            if isinstance(df, pd.Series):
                df = df.to_frame()
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        result: dict[str, Any] = {
            "rt": df[rt_column].values,
            "accuracy": df[accuracy_column].values,
            "n_trials": len(df),
            "mean_rt": float(df[rt_column].mean()),
            "std_rt": float(df[rt_column].std()),
            "mean_accuracy": float(df[accuracy_column].mean()),
        }
        if condition_column and condition_column in df.columns:
            conditions: list[Any] = df[condition_column].unique().tolist()
            result["conditions"] = conditions
            by_condition: dict[str, dict[str, float | int]] = {}
            for cond in conditions:
                mask = df[condition_column] == cond
                by_condition[str(cond)] = {
                    "rt_mean": float(df.loc[mask, rt_column].mean()),
                    "rt_std": float(df.loc[mask, rt_column].std()),
                    "accuracy": float(df.loc[mask, accuracy_column].mean()),
                    "n": int(mask.sum()),
                }
            result["by_condition"] = by_condition
        self._behavioral_data = result
        return result

    def get_segment(
        self,
        start_time: float,
        end_time: float,
        channel_idx: int | None = None,
    ) -> np.ndarray:
        """Extract time segment from loaded data.
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            channel_idx: Channel index (None = all)
        Returns:
            Signal segment
        """
        if "signals" not in self.data:
            raise ValueError("No data loaded. Call load_* method first.")
        fs = self.data["fs"]
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        signals: np.ndarray = self.data["signals"]
        if channel_idx is not None:
            result: np.ndarray = signals[channel_idx, start_sample:end_sample]
            return result
        result_all: np.ndarray = signals[:, start_sample:end_sample]
        return result_all


class NeuralValidator:
    """Validate APGI against neural recordings (EEG, LFP, MEG).
    Key predictions from spec §14:
    - S(t) → Gamma-band power (30-100 Hz)
    - θ(t) → P300/N200 ERP amplitude
    - B(t) → Global ignition (gamma synchrony)
    """

    def __init__(self, fs: float = 100.0):
        """Initialize validator.
        Args:
            fs: Sampling frequency (Hz)
        """
        self.fs = fs

    def extract_gamma_power(
        self,
        signal: np.ndarray,
        freq_range: tuple[float, float] = (30, 100),
        method: str = "welch",
    ) -> float:
        """Extract gamma-band power from neural signal.
        Args:
            signal: Time series data
            freq_range: Gamma frequency range (Hz)
            method: Spectral estimation method
        Returns:
            Gamma-band power
        """
        from scipy import signal as scipy_signal  # type: ignore[import-untyped]

        if method == "welch":
            freqs, psd = scipy_signal.welch(signal, fs=self.fs, nperseg=min(256, len(signal) // 4))
        else:
            # FFT-based periodogram
            freqs = np.fft.rfftfreq(len(signal), 1 / self.fs)
            psd = np.abs(np.fft.rfft(signal)) ** 2
        # Integrate over gamma band
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if np.any(mask):
            return float(np.mean(psd[mask]))
        return 0.0

    def compute_erp(
        self,
        epochs: list[np.ndarray] | np.ndarray,
        baseline: tuple[float, float] = (-0.2, 0.0),
        time_window: tuple[float, float] = (-0.2, 0.6),
    ) -> dict[str, Any]:
        """Compute event-related potential (ERP) from epoched data.
        Args:
            epochs: List of epochs or array (n_epochs, n_samples)
            baseline: Baseline correction window (seconds)
            time_window: ERP time window (seconds)
        Returns:
            Dictionary with ERP statistics
        """
        if isinstance(epochs, list):
            epochs = np.array(epochs)
        n_epochs = epochs.shape[0]
        n_samples = epochs.shape[1]
        times = np.linspace(time_window[0], time_window[1], n_samples)
        # Baseline correction
        baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
        if np.any(baseline_mask):
            baseline_mean = np.mean(epochs[:, baseline_mask], axis=1, keepdims=True)
        else:
            # No baseline window overlap - use zero baseline
            baseline_mean = np.zeros((n_epochs, 1))
        epochs_corrected = epochs - baseline_mean
        # Compute ERP (average over trials)
        erp = np.mean(epochs_corrected, axis=0)
        # Find peaks
        p300_window = (times >= 0.25) & (times <= 0.5)
        n200_window = (times >= 0.15) & (times <= 0.25)
        p300_amp = float(np.max(np.abs(erp[p300_window]))) if np.any(p300_window) else 0.0
        # N200 is a negative deflection, hence min rather than max.
        n200_amp = float(np.min(erp[n200_window])) if np.any(n200_window) else 0.0
        return {
            "erp": erp,
            "times": times,
            "n_epochs": n_epochs,
            "p300_amplitude": p300_amp,
            "n200_amplitude": n200_amp,
            "peak_latency": times[np.argmax(np.abs(erp))],
        }

    def validate_against_apgi(
        self,
        apgi_result: dict[str, Any],
        neural_data: np.ndarray,
        epoch_times: list[tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        """Compare APGI predictions to neural data.
        Args:
            apgi_result: Output from APGIPipeline.step()
            neural_data: Neural recording (time series)
            epoch_times: List of (start, end) times for epochs
        Returns:
            Validation results with correlation metrics
        """
        # Extract APGI-predicted gamma from S(t)
        S_history = apgi_result.get("S_history", [apgi_result.get("S", 0.0)])
        # Compute neural gamma power
        neural_gamma = self.extract_gamma_power(neural_data)
        # Compare to APGI S(t) proxy
        # (In real validation, this would be time-aligned)
        apgi_s_proxy = np.mean(S_history) if S_history else 0.0
        return {
            "neural_gamma_power": neural_gamma,
            "apgi_s_proxy": apgi_s_proxy,
            "prediction_error": abs(neural_gamma - apgi_s_proxy),
            "validated": False,  # Would need time alignment for true validation
        }


class BehavioralValidator:
    """Validate APGI against behavioral data (RT, accuracy, decisions).
    Key predictions from spec §14:
    - S(t) → Perceptual sensitivity (d')
    - θ(t) → RT variability, response criterion
    - B(t) → Overt decision/button press
    """

    def __init__(self) -> None:
        """Initialize behavioral validator."""
        self.data: dict[str, Any] = {}

    def compute_signal_detection_metrics(
        self,
        hits: int,
        misses: int,
        false_alarms: int,
        correct_rejections: int,
    ) -> dict[str, float]:
        """Compute signal detection theory metrics.
        Args:
            hits: Number of hits
            misses: Number of misses
            false_alarms: Number of false alarms
            correct_rejections: Number of correct rejections
        Returns:
            Dictionary with d', criterion, bias metrics
        """
        from scipy import stats  # type: ignore[import-untyped]

        # Hit rate and false alarm rate (with correction for 0/1)
        n_signal = hits + misses
        n_noise = false_alarms + correct_rejections
        hit_rate = (hits + 0.5) / (n_signal + 1.0)
        fa_rate = (false_alarms + 0.5) / (n_noise + 1.0)
        # Convert to z-scores
        z_hit = stats.norm.ppf(hit_rate)
        z_fa = stats.norm.ppf(fa_rate)
        # d' (sensitivity)
        d_prime = z_hit - z_fa
        # Criterion (bias)
        criterion = -0.5 * (z_hit + z_fa)
        # Beta (likelihood ratio)
        beta = np.exp(criterion * d_prime)
        return {
            "d_prime": float(d_prime),
            "criterion": float(criterion),
            "beta": float(beta),
            "hit_rate": float(hit_rate),
            "fa_rate": float(fa_rate),
        }

    def analyze_rt_distribution(
        self,
        rts: np.ndarray,
        remove_outliers: bool = True,
        outlier_sd: float = 3.0,
    ) -> dict[str, Any]:
        """Analyze reaction time distribution.
        Args:
            rts: Reaction times in seconds
            remove_outliers: Whether to remove outliers
            outlier_sd: Outlier threshold in standard deviations
        Returns:
            Dictionary with RT statistics
        """
        rts = np.asarray(rts)
        if remove_outliers:
            mean_rt = np.mean(rts)
            std_rt = np.std(rts)
            mask = np.abs(rts - mean_rt) < outlier_sd * std_rt
            rts = rts[mask]
        return {
            "mean": float(np.mean(rts)),
            "median": float(np.median(rts)),
            "std": float(np.std(rts)),
            "cv": float(np.std(rts) / np.mean(rts)),  # Coefficient of variation
            "skewness": float(self._compute_skewness(rts)),
            "kurtosis": float(self._compute_kurtosis(rts)),
            "n": len(rts),
        }

    @staticmethod
    def _compute_skewness(x: np.ndarray) -> float:
        """Compute skewness."""
        x_arr: np.ndarray = np.asarray(x)
        m3: float = float(np.mean((x_arr - np.mean(x_arr)) ** 3))
        m2: float = float(np.mean((x_arr - np.mean(x_arr)) ** 2))
        return float(m3 / (m2**1.5 + 1e-10))

    @staticmethod
    def _compute_kurtosis(x: np.ndarray) -> float:
        """Compute excess kurtosis."""
        x_arr: np.ndarray = np.asarray(x)
        m4: float = float(np.mean((x_arr - np.mean(x_arr)) ** 4))
        m2: float = float(np.mean((x_arr - np.mean(x_arr)) ** 2))
        return float(m4 / (m2**2 + 1e-10) - 3.0)

    def validate_against_apgi(
        self,
        apgi_result: dict[str, Any],
        behavioral_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare APGI predictions to behavioral data.
        Args:
            apgi_result: Output from APGIPipeline.step()
            behavioral_data: Dictionary with RT, accuracy data
        Returns:
            Validation results
        """
        # Extract APGI threshold dynamics
        theta = apgi_result.get("theta", 0.0)
        # Extract behavioral metrics
        rt_std = behavioral_data.get("rt_std", 0.0)
        response_criterion = behavioral_data.get("criterion", 0.0)
        # Compare: θ(t) should correlate with RT variability
        # (This is a simplified proxy - real validation needs time series)
        theta_rt_correlation_proxy = 1.0 / (1.0 + abs(theta - rt_std))
        return {
            "apgi_theta": theta,
            "behavioral_rt_std": rt_std,
            "behavioral_criterion": response_criterion,
            "correlation_proxy": theta_rt_correlation_proxy,
            "prediction_match": theta_rt_correlation_proxy > 0.5,
        }


class CrossValidationRunner:
    """Run k-fold cross-validation for APGI parameter fitting.
        Fits APGI parameters to empirical data using cross-validation
    to prevent overfitting.
    """

    def __init__(self, n_folds: int = 5):
        """Initialize cross-validation runner.
        Args:
            n_folds: Number of folds for cross-validation
        """
        self.n_folds = n_folds
        self.results: list[dict[str, Any]] = []

    def split_data(
        self,
        data: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split data into train/test folds.
        Args:
            data: Data array (n_samples, ...)
            labels: Optional labels for stratified splitting
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(data)
        indices = np.arange(n_samples)
        # Fold assignment must be reproducible: a cross-validation split that
        # changes between runs makes reported scores unrepeatable.
        resolve_rng(getattr(self, "rng", None)).shuffle(indices)
        fold_size = n_samples // self.n_folds
        splits = []
        for i in range(self.n_folds):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_folds - 1 else n_samples
            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
            splits.append((train_indices, test_indices))
        return splits

    def run_validation(
        self,
        data: np.ndarray,
        apgi_pipeline: Any,
        metric_fn: Callable[..., float],
    ) -> dict[str, Any]:
        """Run cross-validation.
        Args:
            data: Dataset to validate on
            apgi_pipeline: APGIPipeline instance
            metric_fn: Function to compute validation metric
        Returns:
            Cross-validation results
        """
        splits = self.split_data(data)
        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            train_data = data[train_idx]
            test_data = data[test_idx]
            # Fit on training data (would adjust APGI parameters)
            # For now, just compute metric on test data
            _ = train_data  # Would use for training in full implementation
            score = metric_fn(test_data, apgi_pipeline)
            fold_scores.append(score)
            self.results.append(
                {
                    "fold": fold,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "score": score,
                }
            )
        return {
            "fold_scores": fold_scores,
            "mean_score": float(np.mean(fold_scores)),
            "std_score": float(np.std(fold_scores)),
            "n_folds": self.n_folds,
        }


def create_synthetic_validation_dataset(
    n_samples: int = 1000,
    duration: float = 10.0,
    fs: float = 100.0,
    with_ground_truth: bool = True,
    rng: RNGLike = None,
) -> dict[str, Any]:
    """Create synthetic dataset with known APGI parameters for validation testing.
    Args:
        n_samples: Number of trials/subjects
        duration: Duration of each recording (seconds)
        fs: Sampling frequency (Hz)
        with_ground_truth: Include ground truth parameters
        rng: Generator/seed for the synthetic noise. ``None`` uses the
            process-global APGI generator, so a seeded run regenerates the
            identical validation dataset.
    Returns:
        Synthetic dataset with known properties
    """
    generator = resolve_rng(rng)
    n_timepoints = int(duration * fs)
    # Generate signals with known 1/f properties
    from stats.spectral_model import generate_predicted_spectrum_from_hierarchy

    freqs = np.fft.rfftfreq(n_timepoints, 1 / fs)
    data = np.zeros((n_samples, n_timepoints))
    for i in range(n_samples):
        # Generate hierarchical signal
        psd, taus, sigma2s = generate_predicted_spectrum_from_hierarchy(
            freqs,
            n_levels=5,
            tau_min=0.01,
            tau_max=10.0,
        )
        # Generate time series from spectrum
        noise = generator.standard_normal(n_timepoints)
        fft_noise = np.fft.rfft(noise)
        scaling = np.sqrt(psd / (np.abs(fft_noise) ** 2 + 1e-10))
        data[i] = np.fft.irfft(fft_noise * scaling, n=n_timepoints)
    result = {
        "data": data,
        "fs": fs,
        "n_samples": n_samples,
        "duration": duration,
    }
    if with_ground_truth:
        result["ground_truth"] = {
            "taus": taus,
            "sigma2s": sigma2s,
            "beta": 1.0,  # Expected 1/f exponent
        }
    return result
