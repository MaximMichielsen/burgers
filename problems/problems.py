from math import pi
from typing import Callable

from problems.forcing_types import uniform_steady_forcing, none_forcing, sin_steady_forcing
from problems.initial_conditions import uniform_initial_condition


def create_problem_definition(
    forcing: str | Callable | None,
    reynolds: float,
    domain_length: float,
    domain_timespan: float,
    boundary_conditions: str,
    initial_condition: Callable,
) -> dict:
    """Create dictionary containing problem parameters."""
    viscosity = 1 * domain_length / reynolds
    return {
        "domain_timespan": domain_timespan,
        "domain_length": domain_length,
        "reynolds": reynolds,
        "viscosity": viscosity,
        "forcing": forcing,
        "boundary_conditions": boundary_conditions,
        "initial_condition": initial_condition,
    }


problem_robijns_one = create_problem_definition(
    forcing=uniform_steady_forcing,
    domain_length=1,
    domain_timespan=2,
    reynolds=100,
    boundary_conditions="fixed_one",
    initial_condition=uniform_initial_condition,
)

periodic_no_forcing = create_problem_definition(
    forcing=none_forcing,
    domain_length=1,
    domain_timespan=2,
    reynolds=100,
    boundary_conditions="periodic",
    initial_condition=uniform_initial_condition,
)

periodic_steady_forcing = create_problem_definition(
    forcing=uniform_steady_forcing,
    domain_length=1,
    domain_timespan=2,
    reynolds=100,
    boundary_conditions="periodic",
    initial_condition=uniform_initial_condition,
)

periodic_sin_forcing_low_visc = create_problem_definition(
    forcing=sin_steady_forcing,
    domain_length=2 * pi,
    domain_timespan=2,
    reynolds=50,
    boundary_conditions="periodic",
    initial_condition=uniform_initial_condition,)

periodic_sin_forcing_high_visc = create_problem_definition(
    forcing=sin_steady_forcing,
    domain_length=2 * pi,
    domain_timespan=2,
    reynolds=180,
    boundary_conditions="periodic",
    initial_condition=uniform_initial_condition,)

periodic_sin_forcing_med_visc = create_problem_definition(
    forcing=sin_steady_forcing,
    domain_length=2 * pi,
    domain_timespan=2,
    reynolds=100,
    boundary_conditions="periodic",
    initial_condition=uniform_initial_condition,)