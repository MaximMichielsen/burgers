"""Problem definitions for the 1D Burgers pipeline."""

from dataclasses import dataclass
from typing import Callable

from numpy.typing import NDArray

from problems_and_configurations.forcing_types import (
    compute_sine_forcing,
    none_forcing,
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
    external_forcing: Callable | None
    forcing_steady: bool
    boundary_condition_type: str
    boundary_condition_value: float | NDArray | Callable | None = None

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity derived from Re and domain length."""
        return self.domain_length / self.reynolds


# ---------------------------------------------------------------------------
# Problem instances
# ---------------------------------------------------------------------------

pipeline_test = Problem(
    name="pipeline_test",
    domain_length=1.0,
    domain_timespan=0.1,
    reynolds=100,
    initial_condition=uniform_initial_condition,
    external_forcing=uniform_steady_forcing,
    forcing_steady=True,
    boundary_condition_type="fixed",
    boundary_condition_value=1.0,
)

placeholder_problem = Problem(
    name="placeholder",
    domain_length=1.0,
    domain_timespan=0.5,
    reynolds=50,
    initial_condition=zero_initial_condition,
    external_forcing=none_forcing,
    forcing_steady=True,
    boundary_condition_type="fixed",
    boundary_condition_value=0.0,
)

robijns_one = Problem(
    name="robijns_one",
    domain_length=1.0,
    domain_timespan=2.0,
    reynolds=100,
    initial_condition=uniform_initial_condition,
    external_forcing=uniform_steady_forcing,
    forcing_steady=True,
    boundary_condition_type="dirichlet",
    boundary_condition_value=1.0,
)

raj_one = Problem(
    name="raj_one",
    domain_length=1.0,
    domain_timespan=1.0,
    reynolds=100,
    initial_condition=zero_initial_condition,
    external_forcing=uniform_steady_forcing,
    forcing_steady=True,
    boundary_condition_type="dirichlet",
    boundary_condition_value=0.0,
)

raj_two = Problem(
    name="raj_two",
    domain_length=1.0,
    domain_timespan=4.0,
    reynolds=100,
    initial_condition=uniform_initial_condition,
    external_forcing=compute_sine_forcing,
    forcing_steady=False,
    boundary_condition_type="dirichlet",
    boundary_condition_value=1.0,
)

raj_three = Problem(
    name="raj_three",
    domain_length=1.0,
    domain_timespan=1.0,
    reynolds=100,
    initial_condition=uniform_initial_condition,
    external_forcing=uniform_steady_forcing,
    forcing_steady=True,
    boundary_condition_type="dirichlet",
    boundary_condition_value=1.0,
)

periodic_steady_forcing_uniform = Problem(
    name="psfu",
    domain_length=1.0,
    domain_timespan=2.0,
    reynolds=100,
    initial_condition=uniform_initial_condition,
    external_forcing=uniform_steady_forcing,
    forcing_steady=True,
    boundary_condition_type="periodic",
)


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------


class Problems:
    """Namespace for all problem definitions."""

    pipeline_test = pipeline_test
    placeholder = placeholder_problem
    robijns_one = robijns_one
    raj_one = raj_one
    raj_two = raj_two
    raj_three = raj_three
    psfu = periodic_steady_forcing_uniform
