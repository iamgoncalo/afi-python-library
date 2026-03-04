"""
Distortion (D): Structural resistance to traversal.

Single-strategy:
    D = measure / baseline

Multiplicative composition:
    D = R^α · O^β · T^γ · C^δ · M^ε

where each factor >= 1 (neutral = 1, no contribution).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


class Distortion:
    """Single-strategy Distortion measurement.

    D = measure / baseline

    Parameters
    ----------
    measure : float or array-like
        Measured resistance (e.g., traversal time, energy, error rate).
    baseline : float or array-like
        Baseline resistance for normalization. Must be > 0.

    Examples
    --------
    >>> d = Distortion(measure=30.0, baseline=10.0)
    >>> d.value
    3.0
    """

    def __init__(self, measure: float | ArrayLike, baseline: float | ArrayLike = 1.0):
        self._measure = np.asarray(measure, dtype=float)
        self._baseline = np.asarray(baseline, dtype=float)
        self._validate()

    def _validate(self) -> None:
        if np.any(self._baseline <= 0):
            raise ValueError("baseline must be > 0")
        if np.any(self._measure < 0):
            raise ValueError("measure must be >= 0")

    @property
    def value(self) -> float | np.ndarray:
        """Compute D = measure / baseline."""
        result = self._measure / self._baseline
        # Clamp to minimum of 1e-10 to avoid D = 0
        result = np.maximum(result, 1e-10)
        return float(result) if result.ndim == 0 else result

    def __repr__(self) -> str:
        return f"Distortion(measure={self._measure}, baseline={self._baseline})"


@dataclass
class DistortionFactor:
    """A single factor in multiplicative distortion composition.

    Attributes
    ----------
    name : str
        Human-readable name (e.g., "redundancy", "obligation").
    values : np.ndarray
        Factor values (>= 1).
    exponent : float
        Domain-calibrated exponent (> 0).
    """

    name: str
    values: np.ndarray
    exponent: float = 1.0

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if np.any(self.values < 1.0):
            raise ValueError(
                f"Factor '{self.name}' values must be >= 1, got min {np.min(self.values)}"
            )
        if self.exponent <= 0:
            raise ValueError(f"Exponent for '{self.name}' must be > 0, got {self.exponent}")


class MultiplicativeDistortion:
    """Multiplicative distortion composition.

    D = R^α · O^β · T^γ · C^δ · M^ε

    Each factor >= 1 (neutral). Independent barriers compound.

    Parameters
    ----------
    factors : list of DistortionFactor
        The resistance factors with their exponents.

    Examples
    --------
    >>> md = MultiplicativeDistortion([
    ...     DistortionFactor("redundancy", np.array([1.5, 2.0]), exponent=1.0),
    ...     DistortionFactor("obligation", np.array([3.0, 1.2]), exponent=0.8),
    ... ])
    >>> md.value  # R^1.0 * O^0.8 per element
    array([3.78446649, 2.31516738])
    """

    def __init__(self, factors: list[DistortionFactor]):
        if not factors:
            raise ValueError("At least one factor is required")
        self._factors = factors
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        shapes = [f.values.shape for f in self._factors]
        if len(set(shapes)) > 1:
            raise ValueError(f"All factor arrays must have the same shape. Got: {shapes}")

    @property
    def value(self) -> np.ndarray:
        """Compute D = product of factor^exponent."""
        result = np.ones_like(self._factors[0].values)
        for f in self._factors:
            result *= np.power(f.values, f.exponent)
        return result

    @property
    def log_value(self) -> np.ndarray:
        """Compute ln(D) = sum of exponent * ln(factor).

        Useful for linear regression in log-space.
        """
        result = np.zeros_like(self._factors[0].values)
        for f in self._factors:
            result += f.exponent * np.log(f.values)
        return result

    @property
    def factors(self) -> list[DistortionFactor]:
        """List of distortion factors."""
        return list(self._factors)

    @property
    def factor_names(self) -> list[str]:
        """Names of all factors."""
        return [f.name for f in self._factors]

    def contribution(self, factor_name: str) -> np.ndarray:
        """Compute the contribution of a single factor to total D.

        Returns factor^exponent for the named factor.
        """
        for f in self._factors:
            if f.name == factor_name:
                return np.power(f.values, f.exponent)
        raise KeyError(f"Factor '{factor_name}' not found. Available: {self.factor_names}")

    def __repr__(self) -> str:
        parts = [f"{f.name}^{f.exponent}" for f in self._factors]
        return f"MultiplicativeDistortion({' · '.join(parts)})"


class AdditiveDistortion:
    """Additive distortion composition (for comparison testing).

    D = α·R + β·O + γ·T + δ·C + ε·M

    Parameters
    ----------
    factors : list of DistortionFactor
        Same factors, but exponents become linear coefficients.
    """

    def __init__(self, factors: list[DistortionFactor]):
        if not factors:
            raise ValueError("At least one factor is required")
        self._factors = factors

    @property
    def value(self) -> np.ndarray:
        """Compute D = sum of coefficient * factor."""
        result = np.zeros_like(self._factors[0].values)
        for f in self._factors:
            result += f.exponent * f.values
        return result

    def __repr__(self) -> str:
        parts = [f"{f.exponent}·{f.name}" for f in self._factors]
        return f"AdditiveDistortion({' + '.join(parts)})"


def distortion_from_graph(
    graph: "networkx.Graph",
    weight: str = "weight",
    baseline: float | None = None,
) -> dict[tuple, float]:
    """Compute per-edge Distortion from a weighted graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph with edge weights.
    weight : str
        Edge attribute name for weight/cost.
    baseline : float or None
        If None, uses the minimum edge weight as baseline.

    Returns
    -------
    dict
        Mapping (u, v) -> D value for each edge.
    """
    import networkx as nx

    weights = {(u, v): d.get(weight, 1.0) for u, v, d in graph.edges(data=True)}
    if not weights:
        return {}
    if baseline is None:
        baseline = min(weights.values())
    if baseline <= 0:
        raise ValueError(f"baseline must be > 0, got {baseline}")
    return {edge: w / baseline for edge, w in weights.items()}
