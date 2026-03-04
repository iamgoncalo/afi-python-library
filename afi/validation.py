"""
Validation utilities for the AFI framework.

Compare F = P/D against alternative models.
Test multiplicative vs additive distortion.
Sensitivity analysis and noise robustness.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from numpy.typing import ArrayLike


def compare_models(
    p: ArrayLike, d: ArrayLike, navigability: ArrayLike
) -> dict[str, dict]:
    """Compare five Freedom models against measured navigability.

    Models:
        - ratio: F = P/D
        - additive: F = P - D
        - multiplicative: F = P * D
        - nonlinear: F = P²/D
        - random: uniform random

    Parameters
    ----------
    p : array-like
        Measured Perception values.
    d : array-like
        Measured Distortion values.
    navigability : array-like
        Measured navigability (ground truth).

    Returns
    -------
    dict
        Model name -> {'r_squared', 'pearson_r', 'p_value', 'predictions'}.
    """
    p_arr = np.asarray(p, dtype=float).ravel()
    d_arr = np.asarray(d, dtype=float).ravel()
    nav = np.asarray(navigability, dtype=float).ravel()

    if not (len(p_arr) == len(d_arr) == len(nav)):
        raise ValueError("p, d, and navigability must have the same length")

    models = {
        "ratio_PD": p_arr / np.maximum(d_arr, 1e-10),
        "additive_P-D": p_arr - d_arr,
        "multiplicative_PxD": p_arr * d_arr,
        "nonlinear_P2D": (p_arr ** 2) / np.maximum(d_arr, 1e-10),
        "random": np.random.default_rng(42).uniform(
            nav.min(), nav.max(), size=len(nav)
        ),
    }

    results = {}
    for name, preds in models.items():
        # Pearson correlation
        r, p_val = stats.pearsonr(preds, nav)
        # R² via explained variance
        ss_res = np.sum((nav - preds) ** 2)
        ss_tot = np.sum((nav - np.mean(nav)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # For comparison, use correlation-based R² which is always in [0,1]
        r_squared_corr = r ** 2

        results[name] = {
            "r_squared": r_squared_corr,
            "pearson_r": r,
            "p_value": p_val,
            "predictions": preds,
        }

    return results


def compare_distortion_composition(
    factors: dict[str, ArrayLike],
    exponents: dict[str, float],
    measured_d: ArrayLike,
) -> dict[str, dict]:
    """Compare multiplicative vs additive distortion composition.

    Parameters
    ----------
    factors : dict
        Factor name -> values array (each >= 1).
    exponents : dict
        Factor name -> exponent/coefficient.
    measured_d : array-like
        Measured distortion (ground truth).

    Returns
    -------
    dict with 'multiplicative' and 'additive' sub-dicts,
    each containing 'r_squared', 'predictions', 'residuals'.
    """
    measured = np.asarray(measured_d, dtype=float)
    factor_arrays = {k: np.asarray(v, dtype=float) for k, v in factors.items()}

    # Multiplicative: D = prod(factor^exponent)
    mult_pred = np.ones_like(measured)
    for name, vals in factor_arrays.items():
        mult_pred *= np.power(vals, exponents.get(name, 1.0))

    # Additive: D = sum(coeff * factor)
    add_pred = np.zeros_like(measured)
    for name, vals in factor_arrays.items():
        add_pred += exponents.get(name, 1.0) * vals

    results = {}
    for label, preds in [("multiplicative", mult_pred), ("additive", add_pred)]:
        r, _ = stats.pearsonr(preds, measured)
        results[label] = {
            "r_squared": r ** 2,
            "predictions": preds,
            "residuals": measured - preds,
            "mean_absolute_error": float(np.mean(np.abs(measured - preds))),
        }

    results["delta_r_squared"] = (
        results["multiplicative"]["r_squared"] - results["additive"]["r_squared"]
    )
    return results


def noise_robustness(
    p: ArrayLike,
    d: ArrayLike,
    navigability: ArrayLike,
    noise_levels: ArrayLike = (0.01, 0.05, 0.10, 0.20, 0.50),
    n_trials: int = 30,
    seed: int = 42,
) -> dict:
    """Test ratio law robustness under Gaussian noise on P.

    Parameters
    ----------
    p : array-like
        True Perception values.
    d : array-like
        True Distortion values.
    navigability : array-like
        Measured navigability.
    noise_levels : array-like
        Standard deviation levels as fraction of P range.
    n_trials : int
        Number of noisy repetitions per level.
    seed : int
        Random seed.

    Returns
    -------
    dict with 'noise_levels', 'mean_r_squared', 'std_r_squared'.
    """
    rng = np.random.default_rng(seed)
    p_arr = np.asarray(p, dtype=float)
    d_arr = np.asarray(d, dtype=float)
    nav = np.asarray(navigability, dtype=float)
    p_scale = np.std(p_arr)

    results_r2 = []
    for sigma in noise_levels:
        trial_r2 = []
        for _ in range(n_trials):
            p_noisy = p_arr + rng.normal(0, sigma * p_scale, size=p_arr.shape)
            p_noisy = np.maximum(p_noisy, 1e-10)  # keep positive
            f_noisy = p_noisy / np.maximum(d_arr, 1e-10)
            r, _ = stats.pearsonr(f_noisy.ravel(), nav.ravel())
            trial_r2.append(r ** 2)
        results_r2.append(trial_r2)

    results_r2 = np.array(results_r2)
    return {
        "noise_levels": np.array(noise_levels),
        "mean_r_squared": np.mean(results_r2, axis=1),
        "std_r_squared": np.std(results_r2, axis=1),
    }
