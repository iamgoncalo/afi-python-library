"""
Convergence Bound.

    t_conv ≤ D̄ · ln(|S|) / P_mean

where:
    D̄ = mean distortion across traversed path
    |S| = state space cardinality
    P_mean = mean perception of the swarm
"""

from __future__ import annotations

import numpy as np


class ConvergenceBound:
    """Predict and evaluate convergence time bounds.

    Parameters
    ----------
    mean_distortion : float
        Mean distortion across search space.
    state_space_size : int
        Cardinality of the state space |S|.
    mean_perception : float
        Mean perception of the swarm.

    Examples
    --------
    >>> cb = ConvergenceBound(mean_distortion=5.0, state_space_size=1000, mean_perception=3.0)
    >>> cb.bound
    11.512925464970229
    """

    def __init__(
        self,
        mean_distortion: float,
        state_space_size: int,
        mean_perception: float,
    ):
        if mean_distortion <= 0:
            raise ValueError(f"mean_distortion must be > 0, got {mean_distortion}")
        if state_space_size < 1:
            raise ValueError(f"state_space_size must be >= 1, got {state_space_size}")
        if mean_perception <= 0:
            raise ValueError(f"mean_perception must be > 0, got {mean_perception}")

        self._d_bar = mean_distortion
        self._s = state_space_size
        self._p_mean = mean_perception

    @property
    def bound(self) -> float:
        """Compute t_conv ≤ D̄ · ln(|S|) / P_mean."""
        return self._d_bar * np.log(self._s) / self._p_mean

    @property
    def mean_distortion(self) -> float:
        return self._d_bar

    @property
    def state_space_size(self) -> int:
        return self._s

    @property
    def mean_perception(self) -> float:
        return self._p_mean

    def evaluate(self, observed_convergence: float) -> dict:
        """Compare predicted bound with observed convergence.

        Parameters
        ----------
        observed_convergence : float
            Observed convergence time (iterations).

        Returns
        -------
        dict with keys:
            'bound': predicted upper bound
            'observed': observed convergence time
            'bound_holds': whether observed <= bound
            'tightness': observed / bound (closer to 1 = tighter bound)
        """
        b = self.bound
        return {
            "bound": b,
            "observed": observed_convergence,
            "bound_holds": observed_convergence <= b,
            "tightness": observed_convergence / b if b > 0 else float("inf"),
        }

    def sensitivity(
        self,
        d_range: np.ndarray | None = None,
        p_range: np.ndarray | None = None,
    ) -> dict:
        """Compute convergence bound over parameter ranges.

        Parameters
        ----------
        d_range : np.ndarray or None
            Range of mean distortion values to sweep.
        p_range : np.ndarray or None
            Range of mean perception values to sweep.

        Returns
        -------
        dict with arrays of bound values.
        """
        result = {}
        if d_range is not None:
            d_range = np.asarray(d_range, dtype=float)
            result["d_sweep"] = d_range * np.log(self._s) / self._p_mean
            result["d_values"] = d_range
        if p_range is not None:
            p_range = np.asarray(p_range, dtype=float)
            result["p_sweep"] = self._d_bar * np.log(self._s) / p_range
            result["p_values"] = p_range
        return result

    def __repr__(self) -> str:
        return (
            f"ConvergenceBound(D̄={self._d_bar}, |S|={self._s}, "
            f"P_mean={self._p_mean}, bound={self.bound:.2f})"
        )
