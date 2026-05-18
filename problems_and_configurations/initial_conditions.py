import numpy as np
from numpy.typing import NDArray


def uniform_initial_condition(mesh: NDArray, alpha: float = 1.0) -> NDArray:
    """Return a uniform initial condition."""
    return alpha * np.ones(len(mesh))


def zero_initial_condition(mesh: NDArray) -> NDArray:
    """Returns a uniform zero initial condition."""
    return np.zeros_like(mesh)
