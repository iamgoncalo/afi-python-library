"""
Freedom (F): The structural availability of paths.

    F = P / D

The ratio is the unique form satisfying scale-covariance,
monotonicity, and dimensional separability (Melo, 2026).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def compute_freedom(p: float | ArrayLike, d: float | ArrayLike) -> float | np.ndarray:
    """Compute Freedom F = P / D.

    Parameters
    ----------
    p : float or array-like
        Perception values (must be > 0).
    d : float or array-like
        Distortion values (must be > 0).

    Returns
    -------
    float or np.ndarray
        Freedom values.

    Raises
    ------
    ValueError
        If P <= 0 or D <= 0.

    Examples
    --------
    >>> compute_freedom(6.0, 2.0)
    3.0
    >>> compute_freedom(np.array([2, 4, 8]), np.array([1, 2, 4]))
    array([2., 2., 2.])
    """
    p_arr = np.asarray(p, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    if np.any(p_arr <= 0):
        raise ValueError(f"P must be > 0, got min {np.min(p_arr)}")
    if np.any(d_arr <= 0):
        raise ValueError(f"D must be > 0, got min {np.min(d_arr)}")
    result = p_arr / d_arr
    return float(result) if result.ndim == 0 else result


def compute_freedom_intelligent(
    p_ext: float | ArrayLike,
    p_rec: float | ArrayLike,
    d_ext: float | ArrayLike,
    d_int: float | ArrayLike,
) -> float | np.ndarray:
    """Compute Freedom in the intelligent regime.

    F = (P_ext × P_rec) / (D_ext × D_int)

    Parameters
    ----------
    p_ext : float or array-like
        External perception (sensor-derived).
    p_rec : float or array-like
        Recursive perception (internal modeling).
    d_ext : float or array-like
        External distortion (environmental resistance).
    d_int : float or array-like
        Internal distortion (computational overhead).

    Returns
    -------
    float or np.ndarray
        Freedom values.
    """
    pe = np.asarray(p_ext, dtype=float)
    pr = np.asarray(p_rec, dtype=float)
    de = np.asarray(d_ext, dtype=float)
    di = np.asarray(d_int, dtype=float)
    for name, arr in [("p_ext", pe), ("p_rec", pr), ("d_ext", de), ("d_int", di)]:
        if np.any(arr <= 0):
            raise ValueError(f"{name} must be > 0")
    return (pe * pr) / (de * di)


class FreedomField:
    """A spatiotemporal Freedom field F(x, t) = P(x, t) / D(x, t).

    Stores time-series of P and D values across agents or spatial
    positions and computes Freedom, statistics, and the E(t) metric.

    Parameters
    ----------
    p_history : np.ndarray
        Shape (n_timesteps, n_agents). Perception values over time.
    d_history : np.ndarray
        Shape (n_timesteps, n_agents). Distortion values over time.

    Raises
    ------
    ValueError
        If shapes don't match or values are non-positive.

    Examples
    --------
    >>> P = np.array([[4, 6, 8], [3, 5, 7]])
    >>> D = np.array([[2, 3, 4], [1, 1, 1]])
    >>> ff = FreedomField(P, D)
    >>> ff.freedom.shape
    (2, 3)
    >>> ff.mean_freedom()
    array([2., 5.])
    """

    def __init__(self, p_history: np.ndarray, d_history: np.ndarray):
        self._p = np.asarray(p_history, dtype=float)
        self._d = np.asarray(d_history, dtype=float)
        if self._p.shape != self._d.shape:
            raise ValueError(
                f"P and D must have same shape. Got P={self._p.shape}, D={self._d.shape}"
            )
        if self._p.ndim != 2:
            raise ValueError(f"Expected 2D arrays (timesteps, agents), got {self._p.ndim}D")
        if np.any(self._p <= 0):
            raise ValueError("All P values must be > 0")
        if np.any(self._d <= 0):
            raise ValueError("All D values must be > 0")

    @property
    def freedom(self) -> np.ndarray:
        """F(x, t) = P(x, t) / D(x, t). Shape: (n_timesteps, n_agents)."""
        return self._p / self._d

    @property
    def perception(self) -> np.ndarray:
        """P history. Shape: (n_timesteps, n_agents)."""
        return self._p.copy()

    @property
    def distortion(self) -> np.ndarray:
        """D history. Shape: (n_timesteps, n_agents)."""
        return self._d.copy()

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps."""
        return self._p.shape[0]

    @property
    def n_agents(self) -> int:
        """Number of agents / spatial positions."""
        return self._p.shape[1]

    def mean_freedom(self) -> np.ndarray:
        """Mean Freedom across agents at each timestep. Shape: (n_timesteps,)."""
        return np.mean(self.freedom, axis=1)

    def std_freedom(self) -> np.ndarray:
        """Std of Freedom across agents at each timestep. Shape: (n_timesteps,)."""
        return np.std(self.freedom, axis=1, ddof=0)

    def cv_freedom(self) -> np.ndarray:
        """Coefficient of variation of Freedom. Shape: (n_timesteps,).

        CV = std(F) / mean(F)

        This is the exploration-exploitation quantifier E(t).
        """
        mu = self.mean_freedom()
        sigma = self.std_freedom()
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(mu > 0, sigma / mu, np.inf)
        return cv

    def swarm_freedom(self) -> np.ndarray:
        """Swarm Freedom: mean P / mean D at each timestep."""
        mean_p = np.mean(self._p, axis=1)
        mean_d = np.mean(self._d, axis=1)
        return mean_p / mean_d

    def decompose_anomaly(
        self, baseline_end: int, threshold_sigma: float = 2.0
    ) -> dict:
        """Decompose Freedom anomalies into ΔP and ΔD causes.

        Parameters
        ----------
        baseline_end : int
            Timestep index marking end of baseline period.
        threshold_sigma : float
            Number of standard deviations for anomaly detection.

        Returns
        -------
        dict with keys:
            'f_baseline': mean Freedom during baseline
            'f_series': Freedom time series (mean across agents)
            'anomaly_mask': boolean array of anomalous timesteps
            'delta_p': fractional change in P from baseline
            'delta_d': fractional change in D from baseline
            'primary_cause': 'perception' or 'distortion' per anomaly
        """
        f_series = self.mean_freedom()
        p_series = np.mean(self._p, axis=1)
        d_series = np.mean(self._d, axis=1)

        f_baseline = np.mean(f_series[:baseline_end])
        f_std = np.std(f_series[:baseline_end], ddof=1) if baseline_end > 1 else 1.0
        p_baseline = np.mean(p_series[:baseline_end])
        d_baseline = np.mean(d_series[:baseline_end])

        anomaly_mask = np.abs(f_series - f_baseline) > threshold_sigma * f_std
        delta_p = (p_series - p_baseline) / p_baseline
        delta_d = (d_series - d_baseline) / d_baseline

        primary_cause = np.where(
            np.abs(delta_p) > np.abs(delta_d), "perception", "distortion"
        )

        return {
            "f_baseline": f_baseline,
            "f_series": f_series,
            "anomaly_mask": anomaly_mask,
            "delta_p": delta_p,
            "delta_d": delta_d,
            "primary_cause": primary_cause,
        }

    def __repr__(self) -> str:
        return (
            f"FreedomField(timesteps={self.n_timesteps}, agents={self.n_agents}, "
            f"mean_F={np.mean(self.freedom):.3f})"
        )
