"""DNS data generation for obtaining training data."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from burgers import Burgers
from constants import (
    DNS_SPATIAL_FACTOR,
    DNS_POINTS_FACTOR,
    DNS_SNAPSHOT_AMOUNT,
)
from functions import calc_required_grid_points, compute_time_step, set_extractions


def create_dns_config(problem_definition: dict) -> dict:
    """Create configuration for Burgers solver for a DNS simulation."""
    simulation_length: float = problem_definition["domain_length"]
    simulation_duration: float = problem_definition["domain_timespan"]
    reynolds: float = problem_definition["reynolds"]
    viscosity: float = problem_definition["viscosity"]
    initial_condition: NDArray | Callable = problem_definition["initial_condition"]

    # TODO: change handling of boundary conditions and forcing -> similar to initial condition
    boundary_conditions = problem_definition["boundary_conditions"]
    forcing = problem_definition["forcing"]

    required_grid_points_dns = calc_required_grid_points(
        length=simulation_length,
        reynolds=reynolds,
        factor_spatial=DNS_SPATIAL_FACTOR,
        factor_points=DNS_POINTS_FACTOR,
    )

    mesh, h = np.linspace(
        start=0, stop=simulation_length, num=required_grid_points_dns, retstep=True
    )

    initial_solution = initial_condition(mesh)

    time_step_dns = compute_time_step(
        mesh=mesh,
        max_velocity=max(initial_solution),
        viscosity=viscosity,
        do_round_down=True,
    )

    dns_extractions = set_extractions(
        duration=simulation_duration,
        extraction_amount=DNS_SNAPSHOT_AMOUNT,
        time_step=time_step_dns,
    )

    config_dns = Burgers.create_config(
        solution_initial=initial_solution,
        simulation_type="dns",
        run_objective="data generation",
        node_amount=required_grid_points_dns,
        boundary_conditions=boundary_conditions,
        forcing=forcing,
        time=simulation_duration,
        time_step=time_step_dns,
        length=simulation_length,
        convergence_tol_residual=1e-6,
        convergence_tol_update=1e-6,
        max_iterations=100,
        relaxation=None,
        viscosity=viscosity,
        extract_at_times=dns_extractions,
    )

    return config_dns


def create_code_test_config() -> dict:
    """Quick test run to check code behavior."""
    n_nodes = 512
    initial_solution = np.zeros(n_nodes)

    config_test = Burgers.create_config(
        solution_initial=initial_solution,
        simulation_type="dns",
        run_objective="code test",
        node_amount=n_nodes,
        boundary_conditions="fixed",
        forcing=None,
        time=0.5,
        time_step=0.05,
        length=1,
        max_iterations=20,
        relaxation=None,
        viscosity=1,
        extract_at_times=[0.2, 0.4],
    )
    return config_test
