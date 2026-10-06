from pathlib import Path

import numpy as np

from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SolverBase, SimulationMode, TauModel
from utils.forcing_pathing import get_forcing_path

PROJECT_ROOT = Path(__file__).resolve().parent
FILTERED_DIR = PROJECT_ROOT / "dns_data" / "projected"

n_nodes_les = 17
z_length = 2.0
dt = 1e-3
t_start = 308.0101
t_end = 309.0100

simulation_mode = SimulationMode.TAU_BASED
tau_model = TauModel.TWO_PARAMS


ic_les = np.load(FILTERED_DIR / f"ic_linear_{n_nodes_les}.npy")
forcing_path = get_forcing_path(
    project_root=PROJECT_ROOT, n_nodes_les=n_nodes_les, dt=dt
)

forcing_projected = np.load(forcing_path)

print(forcing_path)

n_timesteps, _ = np.shape(forcing_projected)
timespan = n_timesteps * dt
if t_end is not None:
    timespan = t_end - t_start

problem = Problem(
    name="tcf_1d",
    domain_length=z_length,
    domain_timespan=timespan,
    reynolds=100.0,
    initial_condition=ic_les,
    forcing=forcing_projected,
    forcing_is_steady=False,
    boundary_condition_type="fixed",
    boundary_condition_value=0.0,
    t_start=t_start,
)

disc_config = DiscretizationConfig(
    n_nodes_les=n_nodes_les,
    n_nodes_dns=513,
    domain_timespan=1,
    temporal_refinement=1,
    courant_les=1,
    domain_length=z_length,
    set_dt_les=dt,
)

solver_les = SolverBase(
    problem=problem,
    disc_config=disc_config,
    simulation_mode=simulation_mode,
    master_path=PROJECT_ROOT
    / "solver_data"
    / f"les_n{n_nodes_les}_p{tau_model.output_dimensions}_dt_{dt}_test",
    tau_model=tau_model,
    external_correction_coefficients=np.array([4.1, 4.1]),
)
solver_les.run_simulation()
solver_les.post_plotting()
