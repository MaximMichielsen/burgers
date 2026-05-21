"""Set initial conditions for a solver run, all are Callables passed using the configuration file."""

import numpy as np
from numpy.typing import NDArray


def uniform_steady_forcing(mesh: NDArray) -> NDArray:
    """Return uniform steady forcing function to be used by x: mesh."""
    return np.ones_like(mesh)


def none_forcing(mesh: NDArray) -> NDArray:
    """Return a zero array, used when forcing is set to 0."""
    return np.zeros_like(mesh)


def sin_cos_forcing(mesh: NDArray, time: float) -> NDArray:
    """Return sin(x) * cos(t) unsteady forcing function."""
    return np.sin(mesh) * np.cos(time)


def sin_steady_forcing(mesh: NDArray) -> NDArray:
    """Return sin(x) steady forcing function."""
    return np.sin(mesh)


def compute_sine_forcing(
    x_coords: np.ndarray,
    time: float,
    num_modes: int = 8,
) -> np.ndarray:
    """Harmonic forcing for the Burgers diagnostic test case (Case 2).

    Implements f_sine = sum_{k=1}^{8} (8 - k + 1) * sin(2*pi*k*x) * sin(2*pi*t),
    as defined in Rajampeta (2022), Table 4.1.

    Args:
        x_coords: Spatial coordinates, shape (n_points,).
        time: Current simulation time.
        num_modes: Number of Fourier modes to sum (default 8).

    Returns:
        Forcing values at each spatial coordinate, shape (n_points,).
    """
    forcing_values = np.zeros_like(x_coords)

    for k_mode in range(1, num_modes + 1):
        amplitude = num_modes - k_mode + 1
        forcing_values += (
            amplitude * np.sin(2 * np.pi * k_mode * x_coords) * np.sin(2 * np.pi * time)
        )

    return forcing_values


def sin_cos_unsteady_forcing(
    mesh: NDArray, t: float, omega: float = 2 * np.pi
) -> NDArray:
    """Return sin(x) * cos(omega t) unsteady forcing function."""
    return np.sin(mesh) * np.cos(omega * t)


def sin_cos_unsteady_forcing_plus_uniform_steady(
    mesh: NDArray, t: float, omega: float = 2 * np.pi, uniform_factor: float = 0.1
) -> NDArray:
    """Return sin(x) * cos(omega t) unsteady forcing function."""
    return np.sin(mesh) * np.cos(omega * t) + uniform_factor * np.ones_like(mesh)
