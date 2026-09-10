from dataclasses import replace
from pathlib import Path

from ml.projection_schedule import ProjectionReferenceSchedule
from ml.tau_ann import TauANNConfig
from ml.training.sac import SACHyperparameters, run_sac_tau_ann_training
from ml.training.td3 import TD3Hyperparameters, run_td3_tau_ann_training
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem, Problems
from solvers.solver_base import SimulationMode, TauModel, SolverBase
from solvers.solver_coupled import SolverCoupled
from utils.pipeline_utils import resolve_pathing, run_dns
from utils.plotting.configs import create_velocity_plot_configs
from utils.plotting.energy_evolution import plot_energy_comparison
from utils.plotting.velocity_comparison import plot_solution_comparison

CURRENT_DIR = Path(__file__).parent.resolve()
problem: Problem = Problems.raj_one
problem = replace(problem, domain_timespan=1.0, reynolds=100)

# general simulation parameters
n_nodes_les: int = 9
temporal_refinement: int = 1
courant_les: float = 1.0

simulation_mode = SimulationMode.TAU_BASED
tau_model = TauModel.FOUR_PARAMS

TOTAL_EPISODES: int = 200

hp_td3 = TD3Hyperparameters(total_episodes=TOTAL_EPISODES)
hp_sac = SACHyperparameters(
    total_episodes=TOTAL_EPISODES, action_dim=tau_model.output_dimensions
)

# discretization config
disc_cfg = DiscretizationConfig(
    n_nodes_les,
    temporal_refinement,
    courant_les,
    problem.domain_length,
)

# pathing
paths = resolve_pathing(problem.name, CURRENT_DIR)

DNS_CACHE_ROOT = CURRENT_DIR / "dns_cache"
run_dns(DNS_CACHE_ROOT, problem, disc_cfg, paths)

# ------------------------------------- General ------------------------------------ #

proj_ref_schedule = ProjectionReferenceSchedule.from_projection_directory(
    projection_dir=paths.projection,
    domain_length=problem.domain_length,
    n_wavenumber_bins=disc_cfg.n_wavenumber_bins,
)

ann_config = TauANNConfig(
    tau_model=tau_model,
    n_wavenumber_bins=disc_cfg.n_wavenumber_bins,
    n_coefficients=tau_model.output_dimensions,
    ann_path=None,
    n_skip_steps=1,
    reward_weight_energy=1.0,
    reward_spectral_exponent=5.0 / 3.0,
)

# ------------------------------------- TD3 Training ------------------------------------ #
td3_config = replace(ann_config, ann_path=paths.td3_model)

td3_model = run_td3_tau_ann_training(
    problem=problem,
    disc_config=disc_cfg,
    tau_ann_config=td3_config,
    master_path=paths.master,
    proj_ref_schedule=proj_ref_schedule,
    hp=hp_td3,
)

# ------------------------------------- SAC Training ------------------------------------ #
sac_config = replace(ann_config, ann_path=paths.sac_model)

sac_model = run_sac_tau_ann_training(
    problem=problem,
    disc_config=disc_cfg,
    tau_ann_config=sac_config,
    master_path=paths.master,
    proj_ref_schedule=proj_ref_schedule,
    hp=hp_sac,
)

# ----------------------------------------- LES solvers ------------------------------------------ #
solver_base = SolverBase(
    problem,
    disc_cfg,
    simulation_mode=simulation_mode,
    tau_model=tau_model,
    master_path=tau_model.get_path(paths),
)
solver_base.run_simulation()
solver_base.post_processing()

solver_td3 = SolverCoupled(
    problem,
    disc_cfg,
    tau_model=tau_model,
    master_path=paths.td3_data,
    ann_path=paths.td3_model,
)
solver_td3.run_simulation()
solver_td3.post_processing()

solver_sac = SolverCoupled(
    problem,
    disc_cfg,
    tau_model=tau_model,
    master_path=paths.sac_data,
    ann_path=paths.sac_model,
)
solver_sac.run_simulation()
solver_sac.post_processing()

# -------------------------------------- Plotting --------------------------------------- #
plot_solution_comparison(
    configs=create_velocity_plot_configs(paths, disc_cfg),
    output_path=paths.master,
)

plot_energy_comparison(
    paths=paths,
    output_path=paths.master,
    domain_length=problem.domain_length,
)
