from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from constants import (
    DNS_SPATIAL_FACTOR,
    DNS_POINTS_FACTOR,
    DNS_SNAPSHOT_AMOUNT,
    DNS_TO_LES_RATIO,
)
from functions import calc_required_grid_points, compute_time_step, set_extractions
from problems_and_configurations.forcing_types import sin_cos_forcing
from problems_and_configurations.initial_conditions import uniform_initial_condition

from solvers.burgers_pure import BurgersPure as Burgers


def create_placeholder_config(
    problem_definition: dict, master_dir: Path | str | None
) -> dict:
    """Placeholder configuration for code development, to be used with placeholder_problem."""
    simulation_length: float = problem_definition["domain_length"]
    reynolds: float = problem_definition["reynolds"]
    viscosity: float = problem_definition["viscosity"]
    initial_condition: NDArray | Callable = problem_definition["initial_condition"]

    grid_points_dns = calc_required_grid_points(
        length=simulation_length,
        reynolds=reynolds,
        factor_spatial=DNS_SPATIAL_FACTOR,
        factor_points=DNS_POINTS_FACTOR,
    )
    print(simulation_length)
    print(grid_points_dns)
    mesh_dns, h_dns = np.linspace(
        start=0, stop=simulation_length, num=grid_points_dns, retstep=True
    )
    initial_solution_dns = initial_condition(mesh_dns)
    time_step_dns = compute_time_step(
        h=h_dns,
        max_velocity=max(initial_solution_dns),
        viscosity=viscosity,
        do_round_down=True,
    )
    master_path = master_dir if master_dir is not None else "runs"
    config = Burgers.create_config(
        initial_condition=initial_condition,
        simulation_mode="dns",
        run_objective="PLACEHOLDER",
        node_amount=grid_points_dns,
        boundary_condition_type=problem_definition["boundary_condition_type"],
        boundary_condition_value=problem_definition["boundary_condition_value"],
        external_forcing=problem_definition["external_forcing"],
        forcing_steady=problem_definition["forcing_steady"],
        domain_timespan=problem_definition["domain_timespan"],
        time_step=time_step_dns,
        domain_length=problem_definition["domain_length"],
        max_iterations=20,
        relaxation=None,
        viscosity=problem_definition["viscosity"],
        extract_at_times=[0.1, 0.2, 0.3, 0.4, 0.5],
        master_path=master_path,
    )
    return config


def create_dns_config(
    problem_definition: dict,
    master_dir: Path | str | None,
) -> dict:
    """DNS configuration file."""
    simulation_length: float = problem_definition["domain_length"]
    simulation_duration: float = problem_definition["domain_timespan"]
    reynolds: float = problem_definition["reynolds"]
    viscosity: float = problem_definition["viscosity"]
    initial_condition: NDArray | Callable = problem_definition["initial_condition"]

    grid_points_dns = calc_required_grid_points(
        length=simulation_length,
        reynolds=reynolds,
        factor_spatial=DNS_SPATIAL_FACTOR,
        factor_points=DNS_POINTS_FACTOR,
    )
    mesh_dns, h_dns = np.linspace(
        start=0, stop=simulation_length, num=grid_points_dns, retstep=True
    )
    initial_solution_dns = initial_condition(mesh_dns)
    time_step_dns = compute_time_step(
        h=h_dns,
        max_velocity=max(initial_solution_dns),
        viscosity=viscosity,
        do_round_down=True,
    )
    master_path = master_dir if master_dir is not None else "runs"
    extractions = set_extractions(
        simulation_duration, DNS_SNAPSHOT_AMOUNT, time_step_dns
    )
    config = Burgers.create_config(
        initial_condition=initial_condition,
        simulation_mode="dns",
        run_objective="data_generation",
        node_amount=grid_points_dns,
        boundary_condition_type=problem_definition["boundary_condition_type"],
        boundary_condition_value=problem_definition["boundary_condition_value"],
        external_forcing=problem_definition["external_forcing"],
        forcing_steady=problem_definition["forcing_steady"],
        domain_timespan=problem_definition["domain_timespan"],
        time_step=time_step_dns,
        domain_length=problem_definition["domain_length"],
        max_iterations=100,
        relaxation=None,
        viscosity=problem_definition["viscosity"],
        extract_at_times=extractions,
        master_path=master_path,
    )
    return config


def create_solver_configs(
    problem_definition: dict,
    dns_dir: Path | str | None = None,
    les_a_dir: Path | str | None = None,
    les_nm_dir: Path | str | None = None,
    dns_time_step: float | None = None,
    les_time_step: float | None = None,
        n_nodes_dns: int | None = None,
        n_nodes_les: int | None = None,
) -> tuple[dict, dict, dict]:
    """Create configuration for Burgers solver for a DNS simulation."""
    simulation_length: float = problem_definition["domain_length"]
    simulation_duration: float = problem_definition["domain_timespan"]
    reynolds: float = problem_definition["reynolds"]
    viscosity: float = problem_definition["viscosity"]
    initial_condition: NDArray | Callable = problem_definition["initial_condition"]
    boundary_condition_type = problem_definition["boundary_condition_type"]
    boundary_condition_value = problem_definition["boundary_condition_value"]
    forcing = problem_definition["external_forcing"]
    forcing_is_steady = problem_definition["forcing_steady"]

    if n_nodes_dns is not None and n_nodes_les is not None:
        grid_points_dns = n_nodes_dns
        grid_points_les = n_nodes_les
    else:
        reynolds: float = problem_definition["reynolds"]
        grid_points_dns = calc_required_grid_points(
            length=simulation_length,
            reynolds=reynolds,
            factor_spatial=DNS_SPATIAL_FACTOR,
            factor_points=DNS_POINTS_FACTOR,
        )
        grid_points_les = (grid_points_dns - 1) // DNS_TO_LES_RATIO + 1

    mesh_dns, h_dns = np.linspace(
        start=0, stop=simulation_length, num=grid_points_dns, retstep=True
    )

    mesh_les, h_les = np.linspace(
        start=0, stop=simulation_length, num=grid_points_les, retstep=True
    )

    initial_solution_dns = initial_condition(mesh_dns)
    initial_solution_les = initial_condition(mesh_les)

    max_velocity = max(max(initial_solution_dns), max(initial_solution_les))

    time_step_dns = (
        dns_time_step
        if dns_time_step is not None
        else compute_time_step(
            h=h_dns,
            max_velocity=max(initial_solution_dns),
            viscosity=viscosity,
            do_round_down=True,
        )
    )

    time_step_les = (
        les_time_step
        if les_time_step is not None
        else compute_time_step(
            h=h_les, max_velocity=max_velocity, viscosity=viscosity, do_round_down=True
        )
    )

    n_dns_steps = int(round(simulation_duration / time_step_dns))

    dns_extractions = set_extractions(
        duration=simulation_duration,
        extraction_amount=n_dns_steps,  # every step
        time_step=time_step_dns,
    )

    les_extractions = set_extractions(
        duration=simulation_duration,
        extraction_amount=int(simulation_duration / time_step_les),
        time_step=time_step_les,
    )

    config_dns = Burgers.create_config(
        initial_condition=initial_solution_dns,
        simulation_mode="dns",
        run_objective="data generation",
        node_amount=grid_points_dns,
        boundary_condition_type=boundary_condition_type,
        boundary_condition_value=boundary_condition_value,
        external_forcing=forcing,
        forcing_steady=forcing_is_steady,
        domain_timespan=simulation_duration,
        time_step=time_step_dns,
        domain_length=simulation_length,
        convergence_tol_residual=1e-6,
        convergence_tol_update=1e-6,
        max_iterations=100,
        relaxation=None,
        viscosity=viscosity,
        extract_at_times=dns_extractions,
        master_path=dns_dir,
    )

    config_les_analytical = Burgers.create_config(
        initial_condition=initial_solution_les,
        simulation_mode="les",
        run_objective="data_generation",
        node_amount=grid_points_les,
        boundary_condition_type=boundary_condition_type,
        boundary_condition_value=boundary_condition_value,
        external_forcing=forcing,
        forcing_steady=forcing_is_steady,
        domain_timespan=simulation_duration,
        time_step=time_step_les,
        domain_length=simulation_length,
        convergence_tol_residual=1e-4,
        convergence_tol_update=1e-4,
        max_iterations=20,
        relaxation=None,
        viscosity=viscosity,
        extract_at_times=les_extractions,
        master_path=les_a_dir,
    )

    config_les_no_model = Burgers.create_config(
        initial_condition=initial_solution_les,
        simulation_mode="no_model",
        run_objective="data_generation",
        node_amount=grid_points_les,
        boundary_condition_type=boundary_condition_type,
        boundary_condition_value=boundary_condition_value,
        external_forcing=forcing,
        forcing_steady=forcing_is_steady,
        domain_timespan=simulation_duration,
        time_step=time_step_les,
        domain_length=simulation_length,
        convergence_tol_residual=1e-4,
        convergence_tol_update=1e-4,
        max_iterations=20,
        relaxation=None,
        viscosity=viscosity,
        extract_at_times=les_extractions,
        master_path=les_nm_dir,
    )
    return config_dns, config_les_analytical, config_les_no_model


from solvers.burgers_coupled import BurgersCoupled


def create_ann_config(
    problem_definition: dict,
    ann_model_path: Path,
    normalisation_stats_path: Path,
    les_ann_dir: Path | str | None = None,
    time_step_override: float | None = None,
    n_nodes_les: int | None = None,
) -> dict:
    """ANN-coupled LES configuration, matching LES grid and time step."""
    simulation_length: float = problem_definition["domain_length"]
    simulation_duration: float = problem_definition["domain_timespan"]
    reynolds: float = problem_definition["reynolds"]
    viscosity: float = problem_definition["viscosity"]
    initial_condition: NDArray | Callable = problem_definition["initial_condition"]

    grid_points_dns = calc_required_grid_points(
        length=simulation_length,
        reynolds=reynolds,
        factor_spatial=DNS_SPATIAL_FACTOR,
        factor_points=DNS_POINTS_FACTOR,
    )
    grid_points_les = (
        n_nodes_les
        if n_nodes_les is not None
        else (grid_points_dns - 1) // DNS_TO_LES_RATIO + 1
    )
    mesh_les, h_les = np.linspace(
        start=0, stop=simulation_length, num=grid_points_les, retstep=True
    )
    initial_solution_les = initial_condition(mesh_les)

    time_step_les = (
        time_step_override
        if time_step_override is not None
        else compute_time_step(
            h=h_les,
            max_velocity=max(initial_solution_les),
            viscosity=viscosity,
            do_round_down=True,
        )
    )
    les_extractions = set_extractions(
        duration=simulation_duration,
        extraction_amount=DNS_SNAPSHOT_AMOUNT // DNS_TO_LES_RATIO,
        time_step=time_step_les,
    )

    return BurgersCoupled.create_coupled_config(
        ann_model_path=ann_model_path,
        normalisation_stats_path=normalisation_stats_path,
        initial_condition=initial_solution_les,
        simulation_mode="ann",
        run_objective="ann_coupled_les",
        node_amount=grid_points_les,
        boundary_condition_type=problem_definition["boundary_condition_type"],
        boundary_condition_value=problem_definition["boundary_condition_value"],
        external_forcing=problem_definition["external_forcing"],
        forcing_steady=problem_definition["forcing_steady"],
        domain_timespan=simulation_duration,
        time_step=time_step_les,
        domain_length=simulation_length,
        convergence_tol_residual=1e-4,
        convergence_tol_update=1e-4,
        max_iterations=20,
        viscosity=viscosity,
        extract_at_times=les_extractions,
        master_path=les_ann_dir,
    )


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
        simulation_mode="dns",
        run_objective="code test",
        node_amount=grid_points_dns,
        boundary_condition_type="fixed",
        boundary_condition_value=0,
        external_forcing=sin_cos_forcing,
        forcing_steady=False,
        domain_timespan=0.5,
        time_step=0.05,
        domain_length=1,
        max_iterations=20,
        relaxation=None,
        viscosity=1,
        extract_at_times=[0.1, 0.2, 0.3, 0.4, 0.5],
        master_path="runs/pipeline_test",
    )
    return config_test
