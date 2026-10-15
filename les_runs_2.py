from dataclasses import replace
from pathlib import Path

import numpy as np

from ml.reference_scheduler import ReferenceTrajectory
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SolverBase, SimulationMode, TauModel

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECTED_DIR = PROJECT_ROOT / "dns_data" / "projected"
TEMPORAL_PROJECTED_DIR = PROJECT_ROOT / "dns_data" / "temporal" / "projected"

# Parameters from bandit training
n_nodes_les = 33
n_nodes_dns = 513
z_length = 2.0

dt = 1e-3
t_start = 314.01
t_end = 315.0
timespan = t_end - t_start

external_coefficients = np.array([1, 1])

simulation_mode = SimulationMode.TAU_BASED
tau_model = TauModel.TWO_PARAMS

# Data loading logic from bandit training
forcing_projected = np.load(TEMPORAL_PROJECTED_DIR / f"forcing_l2_{n_nodes_les}.npy")
times_forcing = np.load(TEMPORAL_PROJECTED_DIR / "times.npy")

problem = Problem(
    name="tcf_1d",
    domain_length=z_length,
    reynolds=100.0,
    initial_condition=None,
    forcing=forcing_projected,
    forcing_times=times_forcing,
    forcing_is_steady=False,
    boundary_condition_type="fixed",
    boundary_condition_value=0.0,
    t_start=t_start,
    t_end=t_end,
)

disc_config = DiscretizationConfig(
    n_nodes_les=n_nodes_les,
    n_nodes_dns=n_nodes_dns,
    temporal_refinement=1,
    set_dt_les=dt,
    domain_length=z_length,
    t_start=t_start,
    t_end=t_end,
)

aaa = PROJECT_ROOT / "dns_data" / "intervals"

reference_trajectory = ReferenceTrajectory(
    target_profile_dir=aaa,
    projected_w_path=PROJECTED_DIR / f"w_linear_{n_nodes_les}.npy",
    times_path=PROJECT_ROOT / "dns_data" / "curated" / "dat" / "times.npy",
    n_les_nodes=n_nodes_les,
    t_start=t_start,
    t_end=t_end,
)

problem = replace(problem, initial_condition=reference_trajectory.initial_condition)

solver_les = SolverBase(
    problem=problem,
    disc_config=disc_config,
    simulation_mode=simulation_mode,
    master_path=PROJECT_ROOT
    / "solver_data"
    / "base_les_2"
    / f"t_{t_start}_{t_end}"
    / f"n{n_nodes_les}_p{tau_model.output_dimensions}_dt{dt}",
    tau_model=tau_model,
    external_correction_coefficients=external_coefficients,
)
solver_les.run_simulation()
solver_les.post_plotting()
