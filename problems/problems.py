from math import pi
from typing import Callable

from numpy.typing import NDArray

from problems.forcing_types import (
    uniform_steady_forcing,
    none_forcing,
    sin_steady_forcing,
    sin_cos_unsteady_forcing,
    sin_cos_unsteady_forcing_plus_uniform_steady,
)
from problems.initial_conditions import uniform_initial_condition


def create_problem_definition(
    forcing: str | Callable | None,
    forcing_is_steady: bool,
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
        "forcing": forcing,
        "forcing_is_steady": forcing_is_steady,
        "boundary_condition_type": boundary_condition_type,
        "boundary_condition_value": boundary_condition_value,
        "initial_condition": initial_condition,
        "name": name,
    }


problem_robijns_one = create_problem_definition(
    forcing=uniform_steady_forcing,
    forcing_is_steady=True,
    domain_length=1,
    domain_timespan=2,
    reynolds=100,
    boundary_condition_type="dirichlet",
    boundary_condition_value=1,
    initial_condition=uniform_initial_condition,
)

periodic_no_forcing = create_problem_definition(
    forcing=none_forcing,
    forcing_is_steady=True,
    domain_length=1,
    domain_timespan=2,
    reynolds=100,
    boundary_condition_type="periodic",
    initial_condition=uniform_initial_condition,
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

periodic_steady_forcing_sin_low_visc = create_problem_definition(
    forcing=sin_steady_forcing,
    forcing_is_steady=True,
    domain_length=2 * pi,
    domain_timespan=2,
    reynolds=50,
    boundary_condition_type="periodic",
    initial_condition=uniform_initial_condition,
    name="psfslv",
)

periodic_steady_forcing_sin_med_visc = create_problem_definition(
    forcing=sin_steady_forcing,
    forcing_is_steady=True,
    domain_length=2 * pi,
    domain_timespan=2,
    reynolds=100,
    boundary_condition_type="periodic",
    initial_condition=uniform_initial_condition,
    name="psfsmv",
)

periodic_steady_forcing_sin_high_visc = create_problem_definition(
    forcing=sin_steady_forcing,
    forcing_is_steady=True,
    domain_length=2 * pi,
    domain_timespan=2,
    reynolds=180,
    boundary_condition_type="periodic",
    initial_condition=uniform_initial_condition,
    name="psfshv",
)

periodic_steady_forcing_sin_high_visc_long_t = create_problem_definition(
    forcing=sin_steady_forcing,
    forcing_is_steady=True,
    domain_length=2 * pi,
    domain_timespan=10,
    reynolds=180,
    boundary_condition_type="periodic",
    initial_condition=uniform_initial_condition,
    name="psfshvlt",
)

periodic_steady_forcing_sin_low_visc_long_t = create_problem_definition(
    forcing=sin_steady_forcing,
    forcing_is_steady=True,
    domain_length=2 * pi,
    domain_timespan=10,
    reynolds=50,
    boundary_condition_type="periodic",
    initial_condition=uniform_initial_condition,
    name="psfslvlt",
)

periodic_unsteady_forcing_sin_med_visc_long_t = create_problem_definition(
    forcing=sin_cos_unsteady_forcing,
    forcing_is_steady=False,
    domain_length=2 * pi,
    domain_timespan=10,
    reynolds=100,
    boundary_condition_type="periodic",
    initial_condition=uniform_initial_condition,
    name="pufsmvlt",
)

periodic_unsteady_forcing_sin_plus_steady_uniform_med_visc_long_t = (
    create_problem_definition(
        forcing=sin_cos_unsteady_forcing_plus_uniform_steady,
        forcing_is_steady=False,
        domain_length=2 * pi,
        domain_timespan=10,
        reynolds=100,
        boundary_condition_type="periodic",
        initial_condition=uniform_initial_condition,
        name="pufspsumvlt",
    )
)
