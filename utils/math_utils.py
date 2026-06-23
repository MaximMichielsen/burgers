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


def implicit_euler_first_order(field: NDArray | float, h: float) -> NDArray:
    """Central-difference first derivative with periodic roll."""
    return (np.roll(field, -1) - np.roll(field, 1)) / (2 * h)


def compute_adjusted_dt(
    dt_nominal: float, time_end: float | int, time_start: float = 0.0
) -> tuple[float, int]:
    """Adjust dt so that time_end is hit exactly.

    Rounds to the nearest integer step count and recomputes dt.
    The relative change in dt is O(1/n_steps), negligible in practice.
    """
    time_span = time_end - time_start
    n_steps = max(1, round(time_span / dt_nominal))
    dt_adjusted = time_span / n_steps
    return dt_adjusted, n_steps
