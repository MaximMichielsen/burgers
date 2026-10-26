from dataclasses import replace
from pathlib import Path

import numpy as np

from ml.original.reference_scheduler import ReferenceTrajectory
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SolverBase, SimulationMode, TauModel
from utils.forcing_pathing import (
    get_forcing_path,
    get_forcing_times_path,
    get_velocity_profiles_path,
)
from utils.profile_plotting import plot_profile_comparisons

PROJECT_ROOT = Path(__file__).resolve().parent
MEAN_PROFILE_DIR = PROJECT_ROOT / "dns_data" / "intervals"

n_nodes_les = 17
n_nodes_dns = 513
z_length = 2.0

dt = 1e-3
t_start = 312.01
t_end = 313.01

simulation_mode = SimulationMode.TAU_BASED
tau_model = TauModel.TWO_PARAMS

# Data loading logic from bandit training
forcing_projected = np.load(get_forcing_path(PROJECT_ROOT, n_nodes_les, dt))
times_forcing = np.load(get_forcing_times_path(PROJECT_ROOT, dt))

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
    problem=problem,
    n_nodes_les=n_nodes_les,
    n_nodes_dns=n_nodes_dns,
    dt_les=dt,
)

reference_trajectory = ReferenceTrajectory(
    mean_target_profile_dir=MEAN_PROFILE_DIR,
    projected_w_path=get_velocity_profiles_path(PROJECT_ROOT, n_nodes_les),
    times_path=PROJECT_ROOT / "dns_data" / "curated" / "dat" / "times.npy",
    n_nodes_les=n_nodes_les,
)

reference_trajectory.set_active_training_interval(t_start, t_end)

problem = replace(problem, initial_condition=reference_trajectory.initial_condition)

master_path = (
    PROJECT_ROOT
    / "solver_data"
    / "test"
    / "base_les"
    / f"t_{t_start}_{t_end}"
    / f"n{n_nodes_les}_p{tau_model.output_dimensions}_dt{dt}"
)

solver_les = SolverBase(
    problem=problem,
    disc_config=disc_config,
    simulation_mode=simulation_mode,
    master_path=master_path,
    tau_model=tau_model,
)
solver_les.run_simulation()
solver_les.post_plotting()

plot_profile_comparisons(
    intervals_dir=MEAN_PROFILE_DIR,
    les_base_dir=master_path,
    t_start=t_start,
    z_domain=(0.0, 2.0),
)
print(f"[SUCCESS] Plotting completed for {master_path}")
