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


def sin_cos_unsteady_forcing(
    mesh: NDArray, t: float, omega: float = 2 * np.pi
) -> NDArray:
    """Return sin(x) * cos(omega t) unsteady forcing function."""
    return np.sin(mesh) * np.cos(omega * t)
