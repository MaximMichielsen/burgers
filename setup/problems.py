"""Problem definitions for the 1D Burgers pipeline."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from setup.forcings import (
    compute_sine_forcing,
    uniform_steady_forcing,
)
from setup.initial_conditions import (
    uniform_initial_condition,
    zero_initial_condition,
)


@dataclass(frozen=True)
class Problem:
    """Immutable problem definition with derived viscosity."""

    name: str
    domain_length: float
    domain_timespan: float
    reynolds: float
    initial_condition: Callable | NDArray | None
    forcing_times: NDArray
    forcing: Callable | NDArray | None
    forcing_is_steady: bool
    boundary_condition_type: str
    boundary_condition_value: float | int | tuple[float | int, float | int] | None = (
        None
    )
    t_start: float = 0.0

    @property
    def forcing_start_step(self):
        index = int(np.argmin(np.abs(self.forcing_times + 308.0101 - self.t_start))) # hardcoded, live with it :(
        return index

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity derived from Re and domain length."""
        return self.domain_length / self.reynolds

    @property
    def t_end(self) -> float:
        return self.t_start + self.domain_timespan
