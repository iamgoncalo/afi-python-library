"""
Exploration-Exploitation Quantifier.

    E(t) = σ[F(x,t)] / μ[F(x,t)]

The coefficient of variation of the Freedom field.

    E(t) > 1: Exploration (high variability in navigability)
    E(t) < 1: Exploitation (concentrated navigability)
    E(t) ≈ 1: Transition point
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from afi.core.freedom import FreedomField


class ExplorationExploitation:
    """Detect and analyze the exploration-exploitation transition.

    Parameters
    ----------
    freedom_field : FreedomField
        The Freedom field to analyze.

    Examples
    --------
    >>> P = np.random.uniform(1, 10, (100, 50))
    >>> D = np.random.uniform(1, 5, (100, 50))
    >>> ff = FreedomField(P, D)
    >>> ee = ExplorationExploitation(ff)
    >>> ee.e_series.shape
    (100,)
    """

    def __init__(self, freedom_field: FreedomField):
        self._ff = freedom_field
        self._e_series = freedom_field.cv_freedom()

    @property
    def e_series(self) -> np.ndarray:
        """E(t) time series. Shape: (n_timesteps,)."""
        return self._e_series.copy()

    @property
    def freedom_field(self) -> FreedomField:
        """Underlying Freedom field."""
        return self._ff

    def crossing_iteration(self, direction: str = "down") -> int | None:
        """Find the iteration where E(t) crosses 1.0.

        Parameters
        ----------
        direction : str
            'down': first crossing from E > 1 to E < 1 (exploration → exploitation).
            'up': first crossing from E < 1 to E > 1 (exploitation → exploration).

        Returns
        -------
        int or None
            Iteration index of crossing, or None if no crossing found.
        """
        e = self._e_series
        if len(e) < 2:
            return None

        for i in range(len(e) - 1):
            if direction == "down" and e[i] > 1.0 and e[i + 1] <= 1.0:
                return i + 1
            elif direction == "up" and e[i] < 1.0 and e[i + 1] >= 1.0:
                return i + 1
        return None

    def phase_at(self, t: int) -> str:
        """Return the phase at iteration t.

        Returns
        -------
        str
            'exploration' if E(t) > 1, 'exploitation' if E(t) < 1,
            'transition' if |E(t) - 1| < 0.05.
        """
        if t < 0 or t >= len(self._e_series):
            raise IndexError(f"t={t} out of range [0, {len(self._e_series)})")
        e = self._e_series[t]
        if abs(e - 1.0) < 0.05:
            return "transition"
        return "exploration" if e > 1.0 else "exploitation"

    def phase_durations(self) -> dict[str, int]:
        """Count iterations spent in each phase.

        Returns
        -------
        dict
            Keys: 'exploration', 'exploitation', 'transition'.
            Values: number of iterations in each phase.
        """
        phases = [self.phase_at(t) for t in range(len(self._e_series))]
        return {
            "exploration": phases.count("exploration"),
            "exploitation": phases.count("exploitation"),
            "transition": phases.count("transition"),
        }

    def summary(self) -> dict:
        """Summary statistics of the E(t) series.

        Returns
        -------
        dict with keys:
            'crossing_iteration': int or None
            'mean_e': float
            'initial_e': float
            'final_e': float
            'phase_durations': dict
        """
        return {
            "crossing_iteration": self.crossing_iteration("down"),
            "mean_e": float(np.mean(self._e_series)),
            "initial_e": float(self._e_series[0]),
            "final_e": float(self._e_series[-1]),
            "phase_durations": self.phase_durations(),
        }

    def __repr__(self) -> str:
        crossing = self.crossing_iteration("down")
        return (
            f"ExplorationExploitation(crossing={crossing}, "
            f"initial_E={self._e_series[0]:.3f}, final_E={self._e_series[-1]:.3f})"
        )


def compare_algorithms(
    freedom_fields: dict[str, FreedomField],
) -> dict[str, dict]:
    """Compare E(t) crossings across multiple algorithms.

    Parameters
    ----------
    freedom_fields : dict
        Mapping algorithm_name -> FreedomField.

    Returns
    -------
    dict
        Per-algorithm summary plus cross-algorithm statistics.
    """
    results = {}
    crossings = []

    for name, ff in freedom_fields.items():
        ee = ExplorationExploitation(ff)
        s = ee.summary()
        results[name] = s
        if s["crossing_iteration"] is not None:
            crossings.append(s["crossing_iteration"])

    if len(crossings) >= 2:
        results["_cross_algorithm"] = {
            "mean_crossing": float(np.mean(crossings)),
            "std_crossing": float(np.std(crossings, ddof=1)),
            "cv_crossing": float(np.std(crossings, ddof=1) / np.mean(crossings)),
            "n_algorithms": len(crossings),
        }
    return results
