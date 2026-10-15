"""Problem definitions for the 1D Burgers pipeline."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Problem:
    """Immutable problem definition with derived viscosity."""

    name: str
    domain_length: float
    t_start: float
    t_end: float
    reynolds: float
    initial_condition: Callable | NDArray | None
    forcing_times: NDArray
    forcing: Callable | NDArray | None
    forcing_is_steady: bool
    boundary_condition_type: str
    boundary_condition_value: float | int | tuple[float | int, float | int] | None = (
        None
    )

    @property
    def domain_timespan(self) -> float:
        if self.t_end <= self.t_start:
            raise ValueError(
                f"t_end ({self.t_end}) must be greater than t_start ({self.t_start})"
            )
        return self.t_end - self.t_start

    @property
    def forcing_start_step(self):
        index = int(
            np.argmin(np.abs(self.forcing_times + 308.0101 - self.t_start))
        )  # hardcoded, live with it :(
        return index

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity derived from Re and domain length."""
        return self.domain_length / self.reynolds
