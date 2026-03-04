# AFI — Architecture of Freedom Intelligence

**F = P / D**

A Python library for computing Freedom — the structural availability of paths — across domains: swarm intelligence, building performance, network analysis, and complex adaptive systems.

## Installation

```bash
pip install afi
```

## Quick Start

```python
import numpy as np
from afi import FreedomField, ExplorationExploitation, ConvergenceBound
from afi.core.perception import Perception
from afi.core.distortion import Distortion, MultiplicativeDistortion, DistortionFactor

# 1. Compute Freedom
p = Perception(n_states=16, temporal_depth=3)  # P = log2(16) × 3 = 12
d = Distortion(measure=30.0, baseline=10.0)     # D = 3.0
F = p.value / d.value                            # F = 4.0

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
```

## Core Modules

| Module | What it computes |
|--------|-----------------|
| `afi.core.perception` | P = log₂(N) × T + domain-specific factories (ACO, PSO) |
| `afi.core.distortion` | D single-strategy + multiplicative/additive composition |
| `afi.core.freedom` | F = P/D, FreedomField, intelligent regime, anomaly decomposition |
| `afi.exploration` | E(t) = σ[F]/μ[F], crossing detection, phase analysis |
| `afi.gradient` | dx/dt = −P·∇D, agent simulation on distortion fields |
| `afi.convergence` | t_conv ≤ D̄·ln(|S|)/P_mean bound evaluation |
| `afi.validation` | Model comparison, distortion composition testing, noise robustness |

## Key Design Decisions

- **F = P/D is the ratio law**, not F = P−D or F = P×D. Uniqueness proof in Melo (2026).
- **Distortion factors compose multiplicatively** (D = R^α · O^β · ...), not additively. Both forms are testable via `afi.validation.compare_distortion_composition`.
- **E(t) = σ/μ** is the coefficient of variation (standard deviation divided by mean), not the Fano factor (variance/mean).
- All validation functions compare against alternative models by default.

## Citation

If you use AFI in your research, please cite:

```bibtex
@article{melo2026afi,
  title={Architecture of Freedom Intelligence: Freedom as the structural
         origin of navigability, intelligence, and design},
  author={Melo, Gon{\c{c}}alo},
  year={2026},
  doi={10.2139/ssrn.5137338}
}
```

## License

MIT
