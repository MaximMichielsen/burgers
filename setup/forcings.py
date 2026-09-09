"""Forcing function callables for Burgers solver configurations."""

import numpy as np
from numpy.typing import NDArray


def none_forcing(mesh: NDArray) -> NDArray:
    """Return zero forcing."""
    return np.zeros_like(mesh)


def uniform_steady_forcing(mesh: NDArray) -> NDArray:
    """Return uniform steady forcing f = 1."""
    return np.ones_like(mesh)


def sin_steady_forcing(mesh: NDArray) -> NDArray:
    """Return sin(x) steady forcing."""
    return np.sin(mesh)


def sin_cos_forcing(mesh: NDArray, time: float, omega: float = 2 * np.pi) -> NDArray:
    """Return sin(x) * cos(omega * t) unsteady forcing."""
    return np.sin(mesh) * np.cos(omega * time)


def compute_sine_forcing(
    x_coords: NDArray,
    time: float,
    num_modes: int = 8,
) -> NDArray:
    """Harmonic forcing: sum_{k=1}^{N} (N-k+1) sin(2πkx) sin(2πt).

    Implements the diagnostic test case forcing from Rajampeta (2022), Table 4.1.
    """
    forcing_values = np.zeros_like(x_coords)
    for k_mode in range(1, num_modes + 1):
        forcing_values += (
            (num_modes - k_mode + 1)
            * np.sin(2 * np.pi * k_mode * x_coords)
            * np.sin(2 * np.pi * time)
        )
    return forcing_values
