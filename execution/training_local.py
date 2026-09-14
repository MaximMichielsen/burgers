from dataclasses import replace
from pathlib import Path

from ml.projection_schedule import ProjectionReferenceSchedule
from ml.tau_ann import TauANNConfig, OutputScope
from ml.training.td3 import TD3Hyperparameters, TD3Trainer
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem, Problems
from solvers.solver_base import SimulationMode, SolverBase, TauModel
from solvers.solver_coupled import SolverCoupled
from utils.pipeline_utils import resolve_pathing, run_dns
from utils.plotting.configs import create_velocity_plot_configs
from utils.plotting.energy_evolution import plot_energy_comparison
from utils.plotting.velocity_comparison import plot_solution_comparison

# -------------------- Problem and pipeline configuration ------------------------------ #
CURRENT_DIR = Path(__file__).parent.resolve()
problem: Problem = Problems.raj_one
problem = replace(problem, domain_timespan=2.0, reynolds=100)

# general simulation parameters
n_nodes_les: int = 9
temporal_refinement: int = 1
courant_les: float = 1.0

simulation_mode = SimulationMode.TAU_BASED
tau_model = TauModel.FOUR_PARAMS

TOTAL_EPISODES: int = 500
hp_td3 = TD3Hyperparameters(total_episodes=TOTAL_EPISODES)

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

# ------------------------------------- TD3 Training ------------------------------------ #
proj_ref_schedule = ProjectionReferenceSchedule.from_projection_directory(
    projection_dir=paths.projection,
    n_wavenumber_bins=disc_cfg.n_wavenumber_bins,
)

td3_config = TauANNConfig(
    tau_model=tau_model,
    n_wavenumber_bins=disc_cfg.n_wavenumber_bins,
    n_coefficients=tau_model.output_dimensions,
    ann_path=paths.td3_model,
    n_skip_steps=1,
    reward_weight_energy=1.0,
    reward_spectral_exponent=5.0 / 3.0,
    n_nodes_les=disc_cfg.n_nodes_les,
    output_scope=OutputScope.LOCAL,
)

td3_trainer = TD3Trainer(
    problem=problem,
    disc_config=disc_cfg,
    ann_config=td3_config,
    master_path=paths.master,
    proj_ref_schedule=proj_ref_schedule,
    hp=hp_td3,
)

td3_model = td3_trainer.run_td3_tau_ann_training()
td3_trainer.plot_reward_evolution()


# ----------------------------------------- LES solvers ------------------------------------------ #
solver_tau_base = SolverBase(
    problem,
    disc_cfg,
    simulation_mode=simulation_mode,
    tau_model=tau_model,
    master_path=tau_model.get_path(paths),
)
solver_tau_base.run_simulation()
solver_tau_base.post_processing()

# Run LES using trained RL model
solver_tau_ann = SolverCoupled(
    problem,
    disc_cfg,
    tau_model=tau_model,
    master_path=paths.td3_data,
    ann_path=paths.td3_model,
    ann_config=td3_config,
)
solver_tau_ann.run_simulation()
solver_tau_ann.post_processing()
solver_tau_ann.plot_correction_coefficients()

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
