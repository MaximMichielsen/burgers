from typing import Callable

from numpy.typing import NDArray

from problems_and_configurations.forcing_types import none_forcing, uniform_steady_forcing
from problems_and_configurations.initial_conditions import zero_initial_condition, uniform_initial_condition


def create_problem_definition(
    external_forcing: str | Callable | None,
    forcing_steady: bool,
    reynolds: float,
    domain_length: float,
    domain_timespan: float,
    initial_condition: Callable,
    boundary_condition_type: str,
    boundary_condition_value: float | NDArray | Callable | None = None,
    name: str | None = None,
) -> dict:
    """Create dictionary containing problem parameters."""
    viscosity = 1 * domain_length / reynolds
    return {
        "domain_timespan": domain_timespan,
        "domain_length": domain_length,
        "reynolds": reynolds,
        "viscosity": viscosity,
        "external_forcing": external_forcing,
        "forcing_steady": forcing_steady,
        "boundary_condition_type": boundary_condition_type,
        "boundary_condition_value": boundary_condition_value,
        "initial_condition": initial_condition,
        "name": name,
    }


placeholder_problem = create_problem_definition(
    external_forcing=none_forcing,
    forcing_steady=True,
    domain_length=1,
    domain_timespan=0.5,
    reynolds=50,
    boundary_condition_type="fixed",
    boundary_condition_value=0,
    initial_condition=zero_initial_condition,
    name="placeholder",
)


problem_robijns_one = create_problem_definition(
    forcing=uniform_steady_forcing,
    forcing_is_steady=True,
    domain_length=1,
    domain_timespan=2,
    reynolds=100,
    boundary_condition_type="dirichlet",
    boundary_condition_value=1,
    initial_condition=uniform_initial_condition,
    name="robijns_one",
)


periodic_steady_forcing_uniform = create_problem_definition(
    forcing=uniform_steady_forcing,
    forcing_is_steady=True,
    domain_length=1,
    domain_timespan=2,
    reynolds=100,
    boundary_condition_type="periodic",
    initial_condition=uniform_initial_condition,
    name="psfu",
)
