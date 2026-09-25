from pathlib import Path

import numpy as np

from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SimulationMode, TauModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).parent.resolve()

FILTERED_DIR = PROJECT_ROOT / "dns_data" / "projected"
TRAINING_DIR = PROJECT_ROOT / "dns_data" / "training" / "training"
RUN_DIR = PROJECT_ROOT / "final" / "solver_data" / "training_demo"

n_nodes_les = 65
n_nodes_dns = 513
z_length = 2.0
dt = 1e-3

t_start = 308.0101
t_end = 309.0100

simulation_mode = SimulationMode.TAU_BASED
tau_model = TauModel.TWO_PARAMS

ic_les = np.load(FILTERED_DIR / f"ic_linear_{n_nodes_les}.npy")
forcing_projected = np.load(FILTERED_DIR / f"forcing_l2_{n_nodes_les}.npy")

n_timesteps, _ = np.shape(forcing_projected)
timespan = n_timesteps * dt
if t_end is not None:
    timespan = t_end - t_start

ann_path = RUN_DIR / "ann_model.pt"
target_profile_path = TRAINING_DIR / "stat_mean_w.npy"
w_field_path = TRAINING_DIR / f"w_field_{n_nodes_les}.npy"

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
    n_nodes_dns=n_nodes_dns,
    temporal_refinement=1,
    set_dt_les=dt,
    domain_length=z_length,
    domain_timespan=timespan,
)

ann_config = TauANNConfig(
    tau_model,
    disc_config,
    ann_path=ann_path,
    n_skip_steps=1,
    output_scope=Scope.GLOBAL,
    input_scope=Scope.GLOBAL,
    n_training_episodes=2,
)

reference_trajectory = ReferenceTrajectory(
    target_profile_path=target_profile_path,
    projected_w_path=w_field_path,
    n_les_nodes=n_nodes_les,
    target_time_index=0,
)

trainer = TD3Trainer(
    problem,
    disc_config,
    ann_config,
    master_path=RUN_DIR,
    reference_trajectory=reference_trajectory,
)

trainer.run_training()
