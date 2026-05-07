"""Set initial conditions for a solver run, all are Callables passed using the configuration file."""

from typing import Callable

import numpy as np


def uniform_steady_forcing() -> Callable:
    """Return uniform steady forcing function to be used by x: mesh."""
    return np.ones_like


def none_forcing() -> Callable:
    """Return a zero array, used when forcing is set to 0."""
    return np.zeros_like
