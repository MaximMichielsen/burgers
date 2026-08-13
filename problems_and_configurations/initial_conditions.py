"""Initial condition callables for Burgers solver configurations."""

import numpy as np
from numpy.typing import NDArray


def uniform_initial_condition(mesh: NDArray, alpha: float = 1.0) -> NDArray:
    """Return a uniform field of value alpha."""
    return alpha * np.ones(len(mesh))


def zero_initial_condition(mesh: NDArray) -> NDArray:
    """Return a zero field."""
    return np.zeros_like(mesh)


def sin_initial_condition(mesh: NDArray) -> NDArray:
    """Return a sin field."""
    return np.sin(mesh)
