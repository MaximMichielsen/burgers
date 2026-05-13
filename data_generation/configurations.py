"""DNS data generation for obtaining training data."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from burgers import Burgers
from constants import (
    DNS_SPATIAL_FACTOR,
    DNS_POINTS_FACTOR,
    DNS_SNAPSHOT_AMOUNT,
    DNS_TO_LES_RATIO,
)
from functions import calc_required_grid_points, compute_time_step, set_extractions
from problems.forcing_types import sin_cos_forcing
from problems.initial_conditions import uniform_initial_condition


def create_solver_configs(
    problem_definition: dict,
    master_dir: Path | str | None = None,
    with_dns: bool = True,
    with_les_analytical: bool = True,
    with_les_no_model: bool = True,
    create_predictor_config: bool = False,
) -> dict | tuple[dict, dict] | tuple[dict, dict, dict]:
    """Create configuration for Burgers solver for a DNS simulation."""
    simulation_length: float = problem_definition["domain_length"]
    simulation_duration: float = problem_definition["domain_timespan"]
    reynolds: float = problem_definition["reynolds"]
    viscosity: float = problem_definition["viscosity"]
    initial_condition: NDArray | Callable = problem_definition["initial_condition"]
    boundary_condition_type = problem_definition["boundary_condition_type"]
    boundary_condition_value = problem_definition["boundary_condition_value"]
    forcing = problem_definition["forcing"]
    forcing_is_steady = problem_definition["forcing_is_steady"]

    grid_points_dns = calc_required_grid_points(
        length=simulation_length,
        reynolds=reynolds,
        factor_spatial=DNS_SPATIAL_FACTOR,
        factor_points=DNS_POINTS_FACTOR,
    )

    grid_points_les = grid_points_dns // DNS_TO_LES_RATIO

    mesh_dns, h_dns = np.linspace(
        start=0, stop=simulation_length, num=grid_points_dns, retstep=True
    )

    mesh_les, h_les = np.linspace(
        start=0, stop=simulation_length, num=grid_points_les, retstep=True
    )

    initial_solution_dns = initial_condition(mesh_dns)
    initial_solution_les = initial_condition(mesh_les)

    max_velocity = max(max(initial_solution_dns), max(initial_solution_les))

    time_step_dns = compute_time_step(
        h=h_dns,
        max_velocity=max(initial_solution_dns),
        viscosity=viscosity,
        do_round_down=True,
    )

    time_step_les = compute_time_step(
        h=h_les, max_velocity=max_velocity, viscosity=viscosity, do_round_down=True
    )

    dns_extractions = set_extractions(
        duration=simulation_duration,
        extraction_amount=DNS_SNAPSHOT_AMOUNT,
        time_step=time_step_dns,
    )

    les_extractions = set_extractions(
        duration=simulation_duration,
        extraction_amount=DNS_SNAPSHOT_AMOUNT // DNS_TO_LES_RATIO,
        time_step=time_step_les,
    )

    config_dns = Burgers.create_config(
        initial_condition=initial_solution_dns,
        simulation_type="dns",
        run_objective="data generation",
        node_amount=grid_points_dns,
        boundary_condition_type=boundary_condition_type,
        boundary_condition_value=boundary_condition_value,
        forcing=forcing,
        forcing_is_steady=forcing_is_steady,
        domain_timespan=simulation_duration,
        time_step=time_step_dns,
        domain_length=simulation_length,
        convergence_tol_residual=1e-6,
        convergence_tol_update=1e-6,
        max_iterations=100,
        relaxation=None,
        viscosity=viscosity,
        time_extractions=dns_extractions,
        master_path=master_dir,
    )

    config_les_analytical = Burgers.create_config(
        initial_condition=initial_solution_les,
        simulation_type="les",
        run_objective="data_generation",
        node_amount=grid_points_les,
        boundary_condition_type=boundary_condition_type,
        boundary_condition_value=boundary_condition_value,
        forcing=forcing,
        forcing_is_steady=forcing_is_steady,
        domain_timespan=simulation_duration,
        time_step=time_step_les,
        domain_length=simulation_length,
        convergence_tol_residual=1e-4,
        convergence_tol_update=1e-4,
        max_iterations=20,
        relaxation=None,
        viscosity=viscosity,
        time_extractions=les_extractions,
        master_path=master_dir,
    )

    config_les_no_model = Burgers.create_config(
        initial_condition=initial_solution_les,
        simulation_type="dns",
        run_objective="data_generation",
        node_amount=grid_points_les,
        boundary_condition_type=boundary_condition_type,
        boundary_condition_value=boundary_condition_value,
        forcing=forcing,
        forcing_is_steady=forcing_is_steady,
        domain_timespan=simulation_duration,
        time_step=time_step_les,
        domain_length=simulation_length,
        convergence_tol_residual=1e-4,
        convergence_tol_update=1e-4,
        max_iterations=20,
        relaxation=None,
        viscosity=viscosity,
        time_extractions=les_extractions,
        master_path=master_dir,
    )

    if create_predictor_config:
        config_les_predictor = Burgers.create_config(
            initial_condition=initial_solution_les,
            simulation_type="les_ann",
            run_objective="data_generation",
            node_amount=grid_points_les,
            boundary_condition_type=boundary_condition_type,
            boundary_condition_value=boundary_condition_value,
            forcing=forcing,
            forcing_is_steady=forcing_is_steady,
            domain_timespan=simulation_duration,
            time_step=time_step_les,
            domain_length=simulation_length,
            convergence_tol_residual=1e-4,
            convergence_tol_update=1e-4,
            max_iterations=20,
            relaxation=0.5,
            viscosity=viscosity,
            time_extractions=les_extractions,
            master_path=master_dir,
        )
        return config_les_predictor

    if with_dns and with_les_analytical and with_les_no_model:
        return config_dns, config_les_analytical, config_les_no_model

    elif with_dns and with_les_analytical:
        return config_dns, config_les_analytical

    elif with_dns and with_les_no_model:
        return config_dns, config_les_no_model

    return config_dns


def create_code_test_config() -> dict:
    """Quick test run to check code behavior."""
    grid_points_dns = calc_required_grid_points(
        length=1,
        reynolds=100,
        factor_spatial=DNS_SPATIAL_FACTOR,
        factor_points=DNS_POINTS_FACTOR,
    )

    config_test = Burgers.create_config(
        initial_condition=uniform_initial_condition,
        simulation_type="dns",
        run_objective="code test",
        node_amount=grid_points_dns,
        boundary_condition_type="fixed",
        boundary_condition_value=0,
        forcing=sin_cos_forcing,
        forcing_is_steady=False,
        domain_timespan=0.5,
        time_step=0.05,
        domain_length=1,
        max_iterations=20,
        relaxation=None,
        viscosity=1,
        time_extractions=[0.1, 0.2, 0.3, 0.4, 0.5],
        master_path="runs/pipeline_test",
    )
    return config_test
