"""LES data generation for obtaining training data."""

from old.burgers import Burgers
from constants import NODES_LIST, SIMULATION_DURATION, SIMULATION_LENGTH, VISCOSITY_UNIT
from functions import create_config_variables
from validation.mms.manufactured import set_manufactured_solution_initial

meshes, delta_xs, initial_solutions, time_steps = create_config_variables(
    NODES_LIST,
    SIMULATION_LENGTH,
    VISCOSITY_UNIT,
    initial_condition=set_manufactured_solution_initial,
)

configs = []
for idx, initial_sol in enumerate(initial_solutions):
    config = Burgers.create_config(
        initial_condition=initial_sol,
        simulation_type="les",
        run_objective="data generation",
        node_amount=NODES_LIST[idx],
        boundary_conditions="periodic",
        domain_timespan=SIMULATION_DURATION,
        time_step=time_steps[idx],
        domain_length=SIMULATION_LENGTH,
        convergence_tol_residual=1e-6,
        convergence_tol_update=1e-6,
        max_iterations=100,
        relaxation=None,
        viscosity=VISCOSITY_UNIT,
        time_extractions=None,
    )
    configs.append(config)
