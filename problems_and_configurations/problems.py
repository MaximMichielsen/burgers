"""Problem definitions for the 1D Burgers pipeline."""

from dataclasses import dataclass
from typing import Callable

from problems_and_configurations.forcings import (
    compute_sine_forcing,
    uniform_steady_forcing,
)
from problems_and_configurations.initial_conditions import (
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
    initial_condition: Callable
    forcing: Callable | None
    forcing_is_steady: bool
    boundary_condition_type: str
    boundary_condition_value: float | int | tuple[float | int, float | int] | None = (
        None
    )

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity derived from Re and domain length."""
        return self.domain_length / self.reynolds


pipeline_test = Problem(
    name="pipeline_test",
    domain_length=1.0,
    domain_timespan=0.5,
    reynolds=20,
    initial_condition=uniform_initial_condition,
    forcing=uniform_steady_forcing,
    forcing_is_steady=True,
    boundary_condition_type="fixed",
    boundary_condition_value=1.0,
)

robijns_one = Problem(
    name="robijns_one",
    domain_length=1.0,
    domain_timespan=2.0,
    reynolds=100,
    initial_condition=uniform_initial_condition,
    forcing=uniform_steady_forcing,
    forcing_is_steady=True,
    boundary_condition_type="dirichlet",
    boundary_condition_value=1.0,
)

raj_one = Problem(
    name="raj_one",
    domain_length=1.0,
    domain_timespan=5.0,
    reynolds=100,
    initial_condition=zero_initial_condition,
    forcing=uniform_steady_forcing,
    forcing_is_steady=True,
    boundary_condition_type="dirichlet",
    boundary_condition_value=0.0,
)

raj_two = Problem(
    name="raj_two",
    domain_length=1.0,
    domain_timespan=4.0,
    reynolds=100,
    initial_condition=uniform_initial_condition,
    forcing=compute_sine_forcing,
    forcing_is_steady=False,
    boundary_condition_type="dirichlet",
    boundary_condition_value=1.0,
)

raj_three = Problem(
    name="raj_three",
    domain_length=1.0,
    domain_timespan=1.0,
    reynolds=100,
    initial_condition=uniform_initial_condition,
    forcing=uniform_steady_forcing,
    forcing_is_steady=True,
    boundary_condition_type="dirichlet",
    boundary_condition_value=1.0,
)


class Problems:
    """Namespace for all problem definitions."""

    pipeline_test = pipeline_test
    robijns_one = robijns_one
    raj_one = raj_one
    raj_two = raj_two
    raj_three = raj_three
