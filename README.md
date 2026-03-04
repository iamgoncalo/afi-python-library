# AFI — Architecture of Freedom Intelligence

[![Tests](https://github.com/iamgoncalo/afi-python-library/actions/workflows/tests.yml/badge.svg)](https://github.com/iamgoncalo/afi-python-library/actions/workflows/tests.yml)
[![Paper](https://github.com/iamgoncalo/afi-python-library/actions/workflows/draft-pdf.yml/badge.svg)](https://github.com/iamgoncalo/afi-python-library/actions/workflows/draft-pdf.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**F = P / D**

A Python library for computing Freedom — the structural availability of paths — across domains: swarm intelligence, building performance, network analysis, and complex adaptive systems.

`afi` is a **diagnostic toolkit**, not an optimiser. It sits *on top* of any swarm intelligence library (mealpy, NiaPy, PySwarms) and answers: **why** did this algorithm converge when it did?

## Installation

```bash
pip install afi
```

Or from source:

```bash
git clone https://github.com/iamgoncalo/afi-python-library.git
cd afi-python-library
pip install -e .
```

## Quick Start

```python
import numpy as np
from afi import FreedomField, ExplorationExploitation, ConvergenceBound
from afi.core.perception import Perception
from afi.core.distortion import Distortion, MultiplicativeDistortion, DistortionFactor
from afi.core.freedom import compute_freedom

# 1. Compute Freedom
p = Perception(n_states=16, temporal_depth=3)  # P = log2(16) × 3 = 12
d = Distortion(measure=30.0, baseline=10.0)     # D = 3.0
F = compute_freedom(p.value, d.value)            # F = 4.0

# 2. Multiplicative Distortion
md = MultiplicativeDistortion([
    DistortionFactor("redundancy", np.array([2.0, 1.5]), exponent=1.0),
    DistortionFactor("complexity", np.array([3.0, 2.0]), exponent=0.8),
])
print(md.value)  # Independent barriers compound

# 3. Exploration-Exploitation detection
P_history = np.random.uniform(1, 10, (200, 50))
D_history = np.random.uniform(1, 5, (200, 50))
ff = FreedomField(P_history, D_history)
ee = ExplorationExploitation(ff)
print(f"E(t) crossing at iteration: {ee.crossing_iteration('down')}")

# 4. Convergence bound
cb = ConvergenceBound(mean_distortion=5.0, state_space_size=1000, mean_perception=3.0)
print(f"Predicted bound: {cb.bound:.1f} iterations")

# 5. Model comparison (honest: tests 5 alternatives)
from afi.validation import compare_models
rng = np.random.default_rng(42)
p_data = rng.uniform(1, 10, 500)
d_data = rng.uniform(1, 5, 500)
nav = p_data / d_data + rng.normal(0, 0.1, 500)
results = compare_models(p_data, d_data, nav)
for name, r in results.items():
    print(f"  {name}: R² = {r['r_squared']:.4f}")
```

## Core Modules

| Module | What it computes |
|--------|-----------------|
| `afi.core.perception` | P = log₂(N) × T + domain-specific factories (ACO, PSO, sensors) |
| `afi.core.distortion` | D single-strategy + multiplicative/additive composition |
| `afi.core.freedom` | F = P/D, FreedomField, intelligent regime, anomaly decomposition |
| `afi.exploration` | E(t) = σ[F]/μ[F], crossing detection, phase analysis |
| `afi.gradient` | dx/dt = −P·∇D, agent simulation on distortion fields |
| `afi.convergence` | t_conv ≤ D̄·ln(|S|)/P_mean bound evaluation |
| `afi.validation` | 5-model comparison, distortion composition testing, noise robustness |

## How It Differs from Existing Libraries

`afi` does **not** implement any optimisation algorithm. It is a structural analysis layer.

| | mealpy | NiaPy | PySwarms | pymoo | **afi** |
|---|---|---|---|---|---|
| **Purpose** | Run algorithms | Run algorithms | Run PSO | Multi-objective | **Analyse convergence** |
| **Algorithms** | 233 | 70+ | PSO variants | NSGA-II/III | 0 |
| **P/D decomposition** | — | — | — | — | ✓ |
| **E(t) metric** | — | — | — | — | ✓ |
| **Anomaly attribution** | — | — | — | — | ✓ |
| **Model comparison** | — | — | — | — | ✓ |

Use mealpy to **run** ACO. Then use `afi` to understand **why** it converged.

## Key Design Decisions

- **F = P/D is the ratio law**, not F = P−D or F = P×D. Uniqueness proof in Melo (2026).
- **Distortion factors compose multiplicatively** (D = R^α · O^β · ...), not additively. Both forms are testable via `compare_distortion_composition`.
- **E(t) = σ/μ** is the coefficient of variation of the Freedom field — the first proposed algorithm-agnostic exploration-exploitation metric.
- All validation functions compare against alternative models by default. If an alternative wins, the library reports it honestly.

## Tests

```bash
python run_tests.py
```

40 tests covering all modules, mathematical properties (scale covariance, monotonicity, neutral element), and boundary conditions.

## Documentation

- [Complete Data Verification Appendix](docs/APPENDIX_DATA.md) — every numerical claim verified against manual calculation

## Citation

If you use `afi` in your research, please cite:

```bibtex
@article{melo2026afi,
  title={Freedom as Navigability: A Ratio Law Unifying Swarm Intelligence,
         Gradient Dynamics, and Complex Adaptive Systems},
  author={Melo, Gon{\c{c}}alo},
  journal={SSRN Electronic Journal},
  year={2026},
  doi={10.2139/ssrn.5137338}
}
```

## Acknowledgements

This work was supported by the Portuguese Foundation for Science and Technology (FCT) through Project 2025.00020.AIVLAB.DEUCALION, providing access to the Deucalion supercomputer at the National Advanced Computing Centre (MACC), Guimarães, Portugal.

## License

MIT
