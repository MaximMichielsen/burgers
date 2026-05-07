"""DNS data generation for obtaining training data."""

import numpy as np

from fem.burgers import Burgers
from fem.constants import (
    DNS_MESH_FACTOR,
    REYNOLDS,
    SIMULATION_DURATION,
    SIMULATION_LENGTH,
    SPATIAL_SAFETY_FACTOR,
    VISCOSITY_UNIT, DNS_SNAPSHOT_AMOUNT,
)
from fem.functions import calc_required_grid_points, compute_time_step, set_extractions
from fem.validation.mms.manufactured import set_manufactured_solution_initial

required_grid_points = calc_required_grid_points(
    length=SIMULATION_LENGTH, reynolds=REYNOLDS, factor_spatial=DNS_MESH_FACTOR, factor_points=SPATIAL_SAFETY_FACTOR
)
mesh, h = np.linspace(start=0, stop=SIMULATION_LENGTH, num=required_grid_points, retstep=True)
initial_solution = set_manufactured_solution_initial(mesh)
time_step = compute_time_step(
    mesh=mesh, max_velocity=max(initial_solution), viscosity=VISCOSITY_UNIT, do_round_down=True
)

dns_extractions = set_extractions(
    duration=SIMULATION_DURATION, extraction_amount=DNS_SNAPSHOT_AMOUNT, time_step=time_step
)

config = Burgers.create_config(
    solution_initial=initial_solution,
    simulation_type="dns",
    run_objective="data generation",
    node_amount=required_grid_points,
    boundary_conditions="fixed",
    time=SIMULATION_DURATION,
    time_step=time_step,
    length=SIMULATION_LENGTH,
    convergence_tol_residual=1e-6,
    convergence_tol_update=1e-6,
    max_iterations=100,
    relaxation=None,
    viscosity=VISCOSITY_UNIT,
    extract_at_times=dns_extractions,
)
