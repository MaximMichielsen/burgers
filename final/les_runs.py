from pathlib import Path

import numpy as np

from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SolverBase, SimulationMode, TauModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).parent.resolve()
FILTERED_DIR = PROJECT_ROOT / "dns_data" / "filtered"

n_nodes_les = 33
z_length = 2.0
dt = 1e-4
t_start = 308.0101
t_end = 309.0100

simulation_mode = SimulationMode.TAU_BASED
tau_model = TauModel.TWO_PARAMS

mesh_dns = np.linspace(0.0, z_length, 513)
mesh_les = np.linspace(0.0, z_length, n_nodes_les)

ic_les = np.load(FILTERED_DIR / f"ic_linear_{n_nodes_les}.npy")
forcing_projected = np.load(FILTERED_DIR / f"forcing_l2_{n_nodes_les}.npy")

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
)

disc_config = DiscretizationConfig(
    n_nodes_les=n_nodes_les,
    temporal_refinement=1,
    courant_les=1,
    domain_length=z_length,
)
disc_config.dt_les = dt
disc_config.n_nodes_dns = 513

solver_les = SolverBase(
    problem=problem,
    disc_config=disc_config,
    simulation_mode=simulation_mode,
    master_path=PROJECT_ROOT
    / "final"
    / "solver_data"
    / f"les_n{n_nodes_les}_p{tau_model.output_dimensions}",
    tau_model=tau_model,
    t_start=t_start,
)
solver_les.run_simulation()
solver_les.post_plotting()
