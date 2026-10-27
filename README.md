# APGI: Allostatic Precision-Gated Ignition

[![CI](https://github.com/apgiframework/apgi-implementation/actions/workflows/ci.yml/badge.svg)](https://github.com/apgiframework/apgi-implementation/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21632264-blue)](https://doi.org/10.5281/zenodo.21632264)
[![OSF](https://img.shields.io/badge/OSF-t5hcq-lightblue)](https://osf.io/t5hcq)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

APGI is a computational framework for modeling allostatic threshold dynamics in biological systems. It integrates signal processing, precision-weighted accumulation, threshold adaptation, and ignition mechanisms with optional advanced features including Kuramoto oscillators, reservoir computing, and thermodynamic constraints.

**Key Features:**
- ✅ Complete signal preprocessing pipeline
- ✅ Free Energy Principle-grounded precision system
- ✅ Precision-weighted signal accumulation
- ✅ Allostatic threshold dynamics
- ✅ Hard and soft ignition mechanisms
- ✅ Post-ignition reset and refractory periods
- ✅ Continuous-time SDE formulation
- ✅ Hierarchical multi-timescale architecture
- ✅ Kuramoto oscillators with phase coupling
- ✅ Liquid state machine reservoir layer
- ✅ Landauer's principle thermodynamic grounding
- ✅ Observable mapping to neural/behavioral data
- ✅ Fixed-point stability analysis
- ✅ Comprehensive parameter validation
---
## Quick Start
### Installation
**macOS / Linux (recommended):** one command, no sudo, nothing touches your system Python.
```bash
git clone https://github.com/apgiframework/apgi-implementation.git
cd apgi-implementation
./install.sh            # creates ./.venv and installs numpy, scipy, matplotlib, pydantic, structlog
source .venv/bin/activate
apgi --steps 100 --seed 42
```
`./install.sh --dev` adds the test and lint tools (`python -m pytest tests -q`); `./install.sh --pymc`
adds PyMC. You need Python 3.10 or newer (`brew install python` on a Mac without one).

**Manual install (any platform):**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis]"      # or  pip install -e .  for the bare core
```
Always use a virtual environment. If you previously ran `pip install -e . --break-system-packages`
and another package complains about numpy, repair it with `pip install -U numpy scipy`: APGI works with
numpy 1.24 and 2.x.
### Basic Usage
```python
from pipeline import APGIPipeline
from config import CONFIG

# Initialize pipeline. Always set a seed for anything you intend to report:
# every stochastic element draws from it, so the run is bit-for-bit reproducible.
pipeline = APGIPipeline({**CONFIG, "seed": 42})

# Run single step
result = pipeline.step(
    x_e=0.5,  # Exteroceptive signal
    x_hat_e=0.3,  # Exteroceptive prediction
    x_i=0.2,  # Interoceptive signal
    x_hat_i=0.1,  # Interoceptive prediction
)

# Access results
print(f"Signal: {result['S']:.4f}")
print(f"Threshold: {result['theta']:.4f}")
print(f"Ignition: {result['B']}")
```

### Reproducibility

The defaults are the canonical parameterisation from the Notation Appendix, and
configuration validation **fails closed**: a config that violates the
specification raises rather than running and returning plausible-looking
numbers. Pass `strict=False` to opt out explicitly; the departure is then
recorded in the run manifest.

```python
from core.config_io import load_config
from pipeline import APGIPipeline

config = load_config("configs/prod.toml")  # defaults < file < APGI_* env < args
pipeline = APGIPipeline(config)
for t in range(10_000):
    pipeline.step(x_e=..., x_i=...)

# Write provenance next to every figure or table you publish.
pipeline.write_manifest("outputs/figure_07.manifest.json", n_steps=10_000)
```

The manifest records the seed, the resolved config and its SHA-256, the git
commit and working-tree cleanliness, the interpreter, and every numerical
dependency version — everything needed to regenerate the output exactly.

Configuration resolves across four layers (lowest precedence first): built-in
defaults, a TOML/JSON file from `configs/`, `APGI_*` environment variables, then
explicit arguments. An unrecognised `APGI_*` variable raises rather than being
ignored.

```bash
APGI_CONFIG=configs/prod.toml APGI_SEED=42 python main.py
```
### Run Examples
```bash
# Basic usage
python examples/01_basic_usage.py
# Advanced features
python examples/02_advanced_features.py
# Observable mapping
python examples/03_observable_mapping.py
# Thermodynamic analysis
python examples/04_thermodynamics.py
# BOLD thermodynamics
python examples/05_bold_thermodynamics.py
# Hierarchical system
python examples/06_hierarchical_system.py
# Spectral validation (Lorentzian superposition + Hurst)
python examples/08_spectral_validation.py
# Kuramoto coupling
python examples/09_kuramoto_coupling.py
# Reservoir as threshold
python examples/10_reservoir_as_threshold.py
# Maturity assessment
python examples/11_maturity_assessment.py
# Maturity demo
python examples/12_maturity_demo.py
# Validation end-to-end
python examples/13_validation_e2e.py
# BOLD calibration
python examples/14_bold_calibration.py
# Hierarchical power spectrum
python examples/15_hierarchy_power_spectrum.py
```
---
## Documentation
### Getting Started
- **[API Reference](docs/API-REFERENCE.md)** - Complete API documentation
- **[Parameter Constraints](docs/PARAMETER-CONSTRAINTS.md)** - Parameter guide
### Understanding the System
- **[Design Choices](docs/DESIGN-CHOICES.md)** - Implementation rationale
- **[Specification](docs/APGI-Specs.md)** - Full mathematical specification
- **[APGI Overview](docs/APGI.md)** - Mathematical framework overview
- **[Thermodynamic Grounding](docs/APGI-Thermodynamic-Grounding.md)** - Landauer's principle analysis
### Troubleshooting & Advanced Topics
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common errors and solutions
### Spectral & Hierarchical
- **[Spectral Validation](docs/SPECTRAL-VALIDATION.md)** - 1/f spectral validation guide
- **[Hierarchical Guide](docs/HIERARCHICAL-GUIDE.md)** - Multi-timescale architecture guide
### Computation engine & desk science
- **[Computation Engine](docs/ENGINE.md)** - model variants, the qualified reference `apgi_reduced`, K7 identifiability/recovery, run manifests
- **[CI Gates](docs/CI-GATES.md)** - the required gates and how to run each locally
- **[Dependency Matrix](DEPENDENCY-MATRIX.md)** - ownership, dependency direction, deprecations
---
## Architecture
### Core Components
```text
APGIPipeline
├── Signal Preprocessing
│   ├── Prediction error computation
│   ├── Online variance estimation
│   └── Z-score normalization
├── Precision System
│   ├── Precision computation
│   ├── Neuromodulatory gains (ACh, NE, DA)
│   └── Precision ODE (optional)
├── Signal Accumulation
│   ├── Leaky integration
│   └── Log-compression (optional)
├── Threshold Dynamics
│   ├── Metabolic cost computation
│   ├── Information value computation
│   └── Allostatic update rule
├── Ignition Mechanism
│   ├── Hard threshold
│   └── Soft threshold (sigmoid)
├── Post-Ignition Reset
│   ├── Signal reset
│   └── Refractory period
├── Continuous-Time SDE
│   ├── Euler-Maruyama integration
│   └── Noise-precision duality
├── Hierarchical Architecture
│   ├── Multi-timescale leaky integrators
│   ├── Cross-level modulation
│   └── Weighted aggregation
├── Kuramoto Oscillators
│   ├── Coupled phase dynamics
│   ├── Ornstein-Uhlenbeck noise
│   └── Phase reset on ignition
├── Reservoir Layer 
│   ├── Fixed random weights
│   ├── Ridge regression readout
│   └── Suprathreshold amplification
├── Thermodynamics
│   ├── Landauer cost computation
│   ├── Information bit counting
│   └── Metabolic efficiency analysis
├── Observable Mapping
│   ├── Neural observable extraction
│   ├── Behavioral observable extraction
│   └── Key prediction validation
└── Stability Analysis 
    ├── Jacobian computation
    ├── Eigenvalue analysis
    └── Bifurcation detection
```
### Module Organization
```text
apgi-implementation/
├── core/                          # Core components
│   ├── allostatic.py              # Allostatic dynamics
│   ├── compliance.py              # Compliance validation
│   ├── config_schema.py           # Pydantic config schemas
│   ├── logging_config.py          # Logging configuration
│   ├── preprocessing.py           # Signal preprocessing
│   ├── precision.py               # Precision system
│   ├── accumulation.py            # Signal accumulation
│   ├── threshold.py               # Threshold dynamics
│   ├── ignition.py                # Ignition mechanism
│   ├── reset.py                   # Post-ignition reset
│   ├── sde.py                     # SDE integration
│   ├── thermodynamics.py          # Landauer's principle
│   └── validation.py              # Parameter validation
├── hierarchy/                     # Hierarchical architecture
│   ├── coupling.py                # Cross-level coupling
│   └── multiscale.py              # Multi-timescale system
├── oscillation/                   # Oscillatory coupling
│   ├── kuramoto.py                # Kuramoto oscillators
│   ├── phase.py                   # Phase dynamics
│   └── threshold_modulation.py    # Threshold modulation
├── reservoir/                     # Reservoir computing
│   ├── liquid_network.py          # Liquid network
│   └── liquid_state_machine.py    # LSM implementation
├── validation/                    # Observable mapping
│   ├── empirical_validation.py    # Empirical validation
│   └── observable_mapping.py      # Neural/behavioral observables
├── analysis/                      # Stability analysis
│   └── stability.py               # Fixed-point analysis
├── stats/                         # Statistical analysis
│   ├── maturity_assessment.py     # Maturity assessment
│   ├── spectral_model.py          # Lorentzian spectral fitting
│   ├── spectral_extraction.py     # 1/f signature extraction
│   ├── avalanche.py               # Neuronal avalanche analysis
│   └── hurst.py                   # Hurst exponent (DFA, Welch)
├── energy/                        # Thermodynamics
│   ├── bold_calibration.py        # BOLD calibration
│   ├── calibration_utils.py      # Calibration utilities
│   └── thermodynamics.py          # Thermodynamic constraints
├── models/                        # Seven model variants; apgi_reduced is the qualified reference
├── model_registry/                # Variant registry, parameter mapping
├── engine/                        # Prediction error, somatic gain, ignition sim, latent state, counterfactuals, provenance
├── estimation/                    # Likelihood, MCMC/PyMC, K7a identifiability, K7c recovery, gate table
├── adapters/                      # Canonical inputs/outputs, version adapter
├── runtime/                       # Session engine, checkpoints, batch runner, run manifest
├── qualification/                 # Executable qualification of apgi_reduced
├── tools/                         # check_dependencies.py, run_k7.py
├── gates/                         # K7 results and the per-parameter gate table
├── tests/                         # Test suite (100+ test files)
├── examples/                      # Example scripts (15 examples)
├── docs/                          # Documentation
│   ├── API-REFERENCE.md
│   ├── DESIGN-CHOICES.md
│   ├── PARAMETER-CONSTRAINTS.md
│   ├── SPECTRAL-VALIDATION.md
│   ├── HIERARCHICAL-GUIDE.md
│   ├── TROUBLESHOOTING.md
│   ├── APGI-Specs.md
│   └── APGI.md
├── pipeline.py                    # Main pipeline
├── config.py                      # Configuration
├── main.py                        # CLI interface
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```
---
## Configuration
### Minimal Configuration
```python
from config import CONFIG
from pipeline import APGIPipeline

# Use default configuration
pipeline = APGIPipeline(CONFIG)
```
### Custom Configuration
```python
config = {
    # Signal preprocessing
    "tau_s": 5.0,  # Signal timescale (ms)
    "tau_pi": 1000.0,  # Precision timescale (ms)
    "tau_theta": 1000.0,  # Threshold timescale (ms)
    # Signal accumulation
    "lam": 0.2,  # Integration rate
    # Threshold dynamics
    "eta": 0.1,  # Learning rate
    "c1": 0.2,  # Signal cost
    "c2": 0.1,  # Ignition cost
    # Optional features
    "use_kuramoto": True,
    "use_reservoir": True,
    "use_observable_mapping": True,
    "use_stability_analysis": True,
}
pipeline = APGIPipeline(config)
```
See [Parameter Constraints](docs/PARAMETER-CONSTRAINTS.md) for complete parameter guide.
---
## Testing
### Run All Tests
```bash
python -m pytest tests/ -v
```
### Run Specific Test File
```bash
python -m pytest tests/test_pipeline.py -v
```
### Run with Coverage
```bash
python -m pytest tests/ --cov=. --cov-report=html
```
### Test Results
- **Total Tests:** 799
- **Execution Time:** ~2.5 seconds
- **Coverage:** All major components
---
## Performance
| Metric | Value | Notes |
| :--- | :--- | :--- |
| Pipeline step | ~0.1 ms | Single step execution |
| Memory (base) | ~10 MB | Minimal configuration |
| Memory (per 1000 steps) | ~1 MB | History storage |
| Test suite | ~2.5 s | 799 tests |
| Examples | < 30 s | All 15 examples |
---
## Key Features
### Signal Processing
- Online variance estimation (EMA)
- Z-score normalization
- Generative model dynamics (optional)
### Precision System
- Free Energy Principle grounding
- Precision clamping [0.01, 100]
- Neuromodulatory gains (ACh, NE, DA)
- Precision ODE (optional)
### Threshold Dynamics
- Allostatic update rule
- Metabolic cost computation
- Information value computation
- Exponential decay
### Ignition Mechanism
- Hard threshold (binary)
- Soft threshold (sigmoid)
- Ignition margin computation
### Advanced Features
- **Kuramoto Oscillators:** Coupled phase dynamics with OU noise
- **Reservoir Layer:** Liquid state machine with ridge regression
- **Thermodynamics:** Landauer's principle grounding
- **Observable Mapping:** Neural and behavioral observables
- **Stability Analysis:** Fixed-point and bifurcation analysis
- **Hierarchical System:** Multi-timescale processing
---
## Neuromodulation
### Acetylcholine (ACh)
- Modulates exteroceptive precision
- Increases attention to external signals
- Typical range: 0.5-1.5
### Norepinephrine (NE)
- Modulates either precision or threshold (not both)
- Increases arousal/vigilance
- Typical range: 0-1.0
### Dopamine (DA)
- Additive bias to interoceptive signal
- Encodes motivation/reward
- Typical range: -1.0 to 1.0
---
## Observable Mapping
### Neural Observables
- **S(t) → Gamma-band power** (30-100 Hz)
- **θ(t) → P300/N200 ERP amplitude**
- **B(t) → Global ignition (gamma synchrony)**
### Behavioral Observables
- **S(t) → Perceptual sensitivity (d')**
- **θ(t) → RT variability, response criterion**
- **B(t) → Overt decision/button press**
### Key Testable Prediction
- Hit rate ∝ P_ign(t) = σ(Δ(t) / τ_σ)
- Margin Δ(t) = S(t) - θ(t) outperforms S(t) alone
---
## Thermodynamic Grounding
### Landauer's Principle
```text
E_min = κ_meta · N_erase · k_B · T_env · ln(2)
where N_erase ≈ log₂(S / ε_stab)
```
### Metabolic Constraint
```text
C(t) ≥ κ_meta · N_erase(t) · k_B · T_env · ln(2)
```
### Physical Constants
- Boltzmann constant: 1.38e-23 J/K
- Environment temperature: 310 K (37°C)
- Metabolic efficiency: 1.0 (configurable)
---
## Troubleshooting
### Common Errors
**Error:** `ValueError: lam must be in (0,1)`
- **Solution:** Set `config["lam"]` to value between 0 and 1
**Error:** `ValueError: NE cannot modulate both precision and threshold`
- **Solution:** Set either `ne_on_precision` or `ne_on_threshold` to False
**Error:** `ValueError: dt=X exceeds max Y`
- **Solution:** Reduce `dt` or increase timescales
See [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for more solutions.
---
## Contributing
### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd apgi-implementation
# Create virtual environment
python -m venv venv
source venv/bin/activate
# Install with development dependencies
pip install -e ".[dev]"
# Run tests
python -m pytest tests/ -v
```
### Code Style
- Follow PEP 8
- Use type hints
- Write comprehensive docstrings
- Include spec references (§X.Y format)
### Test Coverage
- Write tests for new features
- Ensure all tests pass
- Maintain > 90% coverage
---
## References
### Specification
- [APGI Specification](docs/APGI-Specs.md) - Full mathematical specification
- [APGI Overview](docs/APGI.md) - Mathematical framework overview
### Documentation Links
- [API Reference](docs/API-REFERENCE.md) - Complete API documentation
- [Design Choices](docs/DESIGN-CHOICES.md) - Implementation rationale
- [Parameter Constraints](docs/PARAMETER-CONSTRAINTS.md) - Parameter guide
- [APGI Overview](docs/APGI.md) - Mathematical framework overview
- [Thermodynamic Grounding](docs/APGI-Thermodynamic-Grounding.md) - Landauer's principle analysis
- [Spectral Validation](docs/SPECTRAL-VALIDATION.md) - 1/f spectral analysis guide
- [Hierarchical Guide](docs/HIERARCHICAL-GUIDE.md) - Multi-timescale architecture guide
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common errors and solutions
### Examples
- [Basic Usage](examples/01_basic_usage.py) - Simple introduction
- [Advanced Features](examples/02_advanced_features.py) - All features
- [Observable Mapping](examples/03_observable_mapping.py) - Neural/behavioral observables
- [Thermodynamics](examples/04_thermodynamics.py) - Landauer's principle
- [BOLD Thermodynamics](examples/05_bold_thermodynamics.py) - BOLD signal analysis
- [Hierarchical System](examples/06_hierarchical_system.py) - Multi-timescale processing
- [Spectral Validation](examples/08_spectral_validation.py) - Lorentzian superposition + Hurst
- [Kuramoto Coupling](examples/09_kuramoto_coupling.py) - Oscillatory coupling
- [Reservoir as Threshold](examples/10_reservoir_as_threshold.py) - Reservoir computing
- [Maturity Assessment](examples/11_maturity_assessment.py) - System maturity
- [Maturity Demo](examples/12_maturity_demo.py) - Maturity demonstration
- [Validation E2E](examples/13_validation_e2e.py) - End-to-end validation
- [BOLD Calibration](examples/14_bold_calibration.py) - BOLD calibration
- [Hierarchy Power Spectrum](examples/15_hierarchy_power_spectrum.py) - Power spectrum analysis
---
## License

This software is licensed under the MIT License. See the [LICENSE](LICENSE) file for the complete license text.

The APGI reference implementation is released as a DOI-archived replication artifact on [Zenodo](https://doi.org/10.5281/zenodo.21632264) for permanent archival and reproducibility.

---
## Citation

To cite the reference implementation, please use:

```bibtex
@software{pesochin_apgi_2026,
  title={APGI: Allostatic Precision-Gated Ignition — Reference Implementation},
  author={Pesochin, Dimitry},
  year={2026},
  version={1.1.0},
  doi={10.5281/zenodo.21632264},
  url={https://github.com/apgiframework/apgi-implementation},
  orcid={https://orcid.org/0009-0005-0328-2896}
}
```

To cite the theoretical framework and epistemic architecture, please refer to the companion manuscripts archived in the following locations:

- **Zenodo Manuscript Records:** [APGI Framework on Zenodo](https://doi.org/10.5281/zenodo.21632264)
- **OSF Preregistration:** [APGI OSF Project](https://osf.io/t5hcq)
- **OSF Registries:** [APGI Registry](https://osf.io/registries/your-registry-id/)

The tripartite open-science infrastructure comprises:
1. **GitHub:** Source code version control
2. **Zenodo:** Permanent DOI-archived release snapshots
3. **OSF:** Preregistration, supplementary materials, and registry links

---
