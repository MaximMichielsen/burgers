from typing import Callable

from problems.forcing_types import uniform_steady_forcing


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
    initial_condition=uniform_steady_forcing,
)
