"""
Gradient Law: dx/dt = −P(x) · ∇D(x)

Systems move opposite the distortion gradient, scaled by perception.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


class GradientLaw:
    """Compute movement direction from the gradient law.

    dx/dt = -P(x) · ∇D(x)

    Parameters
    ----------
    distortion_field : np.ndarray
        2D array representing the distortion landscape.
    """

    def __init__(self, distortion_field: np.ndarray):
        self._d_field = np.asarray(distortion_field, dtype=float)
        if self._d_field.ndim != 2:
            raise ValueError(f"distortion_field must be 2D, got {self._d_field.ndim}D")
        # Compute gradient (∂D/∂y, ∂D/∂x)
        self._grad_y, self._grad_x = np.gradient(self._d_field)

    @property
    def distortion_field(self) -> np.ndarray:
        """The distortion landscape."""
        return self._d_field.copy()

    @property
    def gradient_x(self) -> np.ndarray:
        """∂D/∂x component of the distortion gradient."""
        return self._grad_x.copy()

    @property
    def gradient_y(self) -> np.ndarray:
        """∂D/∂y component of the distortion gradient."""
        return self._grad_y.copy()

    def gradient_magnitude(self) -> np.ndarray:
        """||∇D|| at each grid point."""
        return np.sqrt(self._grad_x**2 + self._grad_y**2)

    def movement_direction(
        self, positions: np.ndarray, perception: float | np.ndarray = 1.0
    ) -> np.ndarray:
        """Compute movement vectors from the gradient law.

        dx/dt = -P(x) · ∇D(x)

        Parameters
        ----------
        positions : np.ndarray
            Shape (n_agents, 2) with (row, col) positions.
        perception : float or np.ndarray
            Scalar or per-agent perception values.

        Returns
        -------
        np.ndarray
            Shape (n_agents, 2) movement vectors (dy, dx).
        """
        positions = np.asarray(positions, dtype=int)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        rows = np.clip(positions[:, 0], 0, self._d_field.shape[0] - 1)
        cols = np.clip(positions[:, 1], 0, self._d_field.shape[1] - 1)

        grad_at_pos = np.column_stack([
            self._grad_y[rows, cols],
            self._grad_x[rows, cols],
        ])

        p = np.asarray(perception, dtype=float)
        if p.ndim == 0:
            p = np.full(len(rows), float(p))

        # dx/dt = -P · ∇D
        movement = -p[:, np.newaxis] * grad_at_pos
        return movement

    def freedom_field(self, perception_field: np.ndarray) -> np.ndarray:
        """Compute F = P / D across the grid.

        Parameters
        ----------
        perception_field : np.ndarray
            Same shape as distortion_field.

        Returns
        -------
        np.ndarray
            Freedom field.
        """
        return perception_field / np.maximum(self._d_field, 1e-10)

    def simulate_agents(
        self,
        initial_positions: np.ndarray,
        perception: float | np.ndarray,
        n_steps: int = 100,
        step_size: float = 1.0,
    ) -> np.ndarray:
        """Simulate agent trajectories under the gradient law.

        Parameters
        ----------
        initial_positions : np.ndarray
            Shape (n_agents, 2).
        perception : float or np.ndarray
            Per-agent or scalar perception.
        n_steps : int
            Number of simulation steps.
        step_size : float
            Multiplier for movement vector.

        Returns
        -------
        np.ndarray
            Shape (n_steps + 1, n_agents, 2) trajectory.
        """
        positions = np.asarray(initial_positions, dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        trajectory = np.zeros((n_steps + 1, len(positions), 2))
        trajectory[0] = positions.copy()

        h, w = self._d_field.shape
        for t in range(n_steps):
            int_pos = np.clip(positions.astype(int), 0, [h - 1, w - 1])
            move = self.movement_direction(int_pos, perception)
            positions = positions + step_size * move
            positions[:, 0] = np.clip(positions[:, 0], 0, h - 1)
            positions[:, 1] = np.clip(positions[:, 1], 0, w - 1)
            trajectory[t + 1] = positions.copy()

        return trajectory

    def __repr__(self) -> str:
        return (
            f"GradientLaw(field_shape={self._d_field.shape}, "
            f"mean_D={np.mean(self._d_field):.3f})"
        )
