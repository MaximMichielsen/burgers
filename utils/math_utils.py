"""Utility functions for math related aspects."""

import numpy as np
from numpy.typing import NDArray


def calc_required_grid_points(
    length: float,
    reynolds: float,
    factor_spatial: float,
    factor_points: float,
    round_to_power_of_2: bool = True,
) -> int:
    """Return DNS node count based on Kolmogorov-scale estimate."""
    spatial_step = factor_spatial * length / reynolds
    required_points = factor_points * length / spatial_step
    if not round_to_power_of_2:
        return int(required_points)
    return int(2 ** np.ceil(np.log2(required_points)))


def round_down(value: float, decimals: int) -> float:
    """Round value down to the given number of decimal places."""
    factor = 10**decimals
    return np.floor(value * factor) / factor


def compute_time_step(
    h: float,
    max_velocity: float,
    viscosity: float,
    do_round_down: bool = True,
) -> float:
    """CFL-based time step: minimum of convective and diffusive limits."""
    if max_velocity == 0:
        return h**2 / viscosity
    dt = min(h / max_velocity, h**2 / viscosity)
    return round_down(dt, 4) if do_round_down else dt


def implicit_euler_first_order(field: NDArray | float, h: float) -> NDArray:
    """Central-difference first derivative with periodic roll."""
    return (np.roll(field, -1) - np.roll(field, 1)) / (2 * h)
