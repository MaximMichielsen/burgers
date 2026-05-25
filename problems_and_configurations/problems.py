"""Problem definitions for the 1D Burgers pipeline."""

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


def create_problem_definition(
    name: str,
    external_forcing: str | Callable | None,
    forcing_steady: bool,
    reynolds: float,
    domain_length: float,
    domain_timespan: float,
    initial_condition: Callable,
    boundary_condition_type: str,
    boundary_condition_value: float | NDArray | Callable | None = None,
) -> dict:
    """Assemble a problem parameter dict with derived viscosity."""
    return {
        "name": name,
        "domain_timespan": domain_timespan,
        "domain_length": domain_length,
        "reynolds": reynolds,
        "viscosity": domain_length / reynolds,
        "external_forcing": external_forcing,
        "forcing_steady": forcing_steady,
        "boundary_condition_type": boundary_condition_type,
        "boundary_condition_value": boundary_condition_value,
        "initial_condition": initial_condition,
    }


pipeline_test = create_problem_definition(
    name="pipeline_test",
    external_forcing=uniform_steady_forcing,
    forcing_steady=True,
    domain_length=1.0,
    domain_timespan=0.5,
    reynolds=100,
    boundary_condition_type="fixed",
    boundary_condition_value=1.0,
    initial_condition=uniform_initial_condition,
)

placeholder_problem = create_problem_definition(
    name="placeholder",
    external_forcing=none_forcing,
    forcing_steady=True,
    domain_length=1,
    domain_timespan=0.5,
    reynolds=50,
    boundary_condition_type="fixed",
    boundary_condition_value=0,
    initial_condition=zero_initial_condition,
)

robijns_one = create_problem_definition(
    name="robijns_one",
    external_forcing=uniform_steady_forcing,
    forcing_steady=True,
    domain_length=1,
    domain_timespan=2,
    reynolds=100,
    boundary_condition_type="dirichlet",
    boundary_condition_value=1,
    initial_condition=uniform_initial_condition,
)

raj_one = create_problem_definition(
    name="raj_one",
    external_forcing=uniform_steady_forcing,
    forcing_steady=True,
    domain_length=1,
    domain_timespan=1,
    reynolds=100,
    boundary_condition_type="dirichlet",
    boundary_condition_value=0,
    initial_condition=zero_initial_condition,
)

raj_two = create_problem_definition(
    name="raj_two",
    external_forcing=compute_sine_forcing,
    forcing_steady=False,
    domain_length=1,
    domain_timespan=4,
    reynolds=100,
    boundary_condition_type="dirichlet",
    boundary_condition_value=1,
    initial_condition=uniform_initial_condition,
)

raj_three = create_problem_definition(
    name="raj_three",
    external_forcing=uniform_steady_forcing,
    forcing_steady=True,
    domain_length=1,
    domain_timespan=1,
    reynolds=100,
    boundary_condition_type="dirichlet",
    boundary_condition_value=1,
    initial_condition=uniform_initial_condition,
)

periodic_steady_forcing_uniform = create_problem_definition(
    name="psfu",
    external_forcing=uniform_steady_forcing,
    forcing_steady=True,
    domain_length=1,
    domain_timespan=2,
    reynolds=100,
    boundary_condition_type="periodic",
    initial_condition=uniform_initial_condition,
)
