from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from constants import DNS_SPATIAL_FACTOR, DNS_POINTS_FACTOR
from old.functions import calc_required_grid_points, compute_time_step

from burgers_pure import BurgersPure as Burgers


def create_placeholder_config(problem_definition: dict,
                              master_dir: Path | str | None,
                              save_dir: Path | str | None) -> dict:
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
        save_path=save_dir
    )
    return config
