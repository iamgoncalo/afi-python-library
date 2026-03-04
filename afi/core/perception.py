"""
Perception (P): Differentiation capacity of a system navigating
a distortion field.

    P = log2(N) × T

where:
    N >= 2: number of distinguishable continuation classes
    T >= 1: predictive depth in temporal steps

Units: bit-cycles
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


class Perception:
    """Compute Perception from distinguishable states and temporal depth.

    Parameters
    ----------
    n_states : int or array-like
        Number of distinguishable states (N >= 2).
    temporal_depth : float or array-like
        Predictive horizon in temporal steps (T >= 1).

    Examples
    --------
    >>> p = Perception(n_states=8, temporal_depth=3)
    >>> p.value
    9.0
    >>> p = Perception(n_states=np.array([4, 8, 16]), temporal_depth=1)
    >>> p.value
    array([2., 3., 4.])
    """

    def __init__(self, n_states: int | ArrayLike, temporal_depth: float | ArrayLike = 1.0):
        self._n_states = np.asarray(n_states, dtype=float)
        self._temporal_depth = np.asarray(temporal_depth, dtype=float)
        self._validate()

    def _validate(self) -> None:
        if np.any(self._n_states < 2):
            raise ValueError(f"n_states must be >= 2, got min {np.min(self._n_states)}")
        if np.any(self._temporal_depth < 1):
            raise ValueError(
                f"temporal_depth must be >= 1, got min {np.min(self._temporal_depth)}"
            )

    @property
    def value(self) -> float | np.ndarray:
        """Compute P = log2(N) × T."""
        result = np.log2(self._n_states) * self._temporal_depth
        return float(result) if result.ndim == 0 else result

    @property
    def n_states(self) -> float | np.ndarray:
        """Number of distinguishable states."""
        return float(self._n_states) if self._n_states.ndim == 0 else self._n_states

    @property
    def temporal_depth(self) -> float | np.ndarray:
        """Predictive horizon."""
        return (
            float(self._temporal_depth)
            if self._temporal_depth.ndim == 0
            else self._temporal_depth
        )

    def __repr__(self) -> str:
        return f"Perception(n_states={self.n_states}, temporal_depth={self.temporal_depth})"


def perception_from_sensors(
    quantization_levels: list[int], temporal_depth: float = 1.0
) -> Perception:
    """Create Perception from a list of sensor quantization levels.

    N = product of all quantization levels (joint distinguishable states).

    Parameters
    ----------
    quantization_levels : list of int
        Quantization levels per sensor. E.g., [256, 128, 64] for three
        sensors with 8-bit, 7-bit, and 6-bit resolution.
    temporal_depth : float
        Predictive horizon in temporal steps.

    Returns
    -------
    Perception
        With N = prod(quantization_levels).

    Examples
    --------
    >>> p = perception_from_sensors([256, 128, 64], temporal_depth=2)
    >>> p.n_states
    2097152.0
    """
    n = int(np.prod(quantization_levels))
    if n < 2:
        raise ValueError(f"Product of quantization levels must be >= 2, got {n}")
    return Perception(n_states=n, temporal_depth=temporal_depth)


def perception_from_aco(
    pheromone: ArrayLike, visibility: ArrayLike, alpha: float = 1.0, beta: float = 2.0
) -> np.ndarray:
    """Compute per-edge Perception for ACO.

    P_ij = tau_ij^alpha * eta_ij^beta

    Parameters
    ----------
    pheromone : array-like
        Pheromone intensities (tau).
    visibility : array-like
        Visibility values (eta = 1/distance).
    alpha : float
        Pheromone exponent.
    beta : float
        Visibility exponent.

    Returns
    -------
    np.ndarray
        Perception values per edge.
    """
    tau = np.asarray(pheromone, dtype=float)
    eta = np.asarray(visibility, dtype=float)
    if np.any(tau < 0):
        raise ValueError("Pheromone values must be non-negative")
    if np.any(eta < 0):
        raise ValueError("Visibility values must be non-negative")
    return np.power(tau, alpha) * np.power(eta, beta)


def perception_from_pso(
    velocity: ArrayLike, pbest_diff: ArrayLike, gbest_diff: ArrayLike,
    w: float = 0.7, c1: float = 2.0, c2: float = 2.0,
) -> np.ndarray:
    """Compute per-particle Perception for PSO.

    P_i = |w * v_i + c1 * (pbest - x_i) + c2 * (gbest - x_i)|

    Parameters
    ----------
    velocity : array-like
        Current velocity vectors (n_particles, n_dims).
    pbest_diff : array-like
        Personal best - current position (n_particles, n_dims).
    gbest_diff : array-like
        Global best - current position (n_particles, n_dims).
    w : float
        Inertia weight.
    c1 : float
        Cognitive coefficient.
    c2 : float
        Social coefficient.

    Returns
    -------
    np.ndarray
        Perception magnitude per particle (n_particles,).
    """
    v = np.asarray(velocity, dtype=float)
    pb = np.asarray(pbest_diff, dtype=float)
    gb = np.asarray(gbest_diff, dtype=float)
    info_vector = w * v + c1 * pb + c2 * gb
    return np.linalg.norm(info_vector, axis=-1)
