from dataclasses import replace
from pathlib import Path

from ml.projection_schedule import ProjectionReferenceSchedule
from ml.tau_ann import TauANNConfig
from ml.training.td3 import run_td3_tau_ann_training, TD3Hyperparameters
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem, Problems
from solvers.solver_base import SimulationMode, SolverBase, TauModel
from solvers.solver_coupled import SolverCoupled
from utils.pipeline_utils import resolve_pathing, run_dns
from utils.plotting.configs import create_velocity_plot_configs
from utils.plotting.energy_evolution import plot_energy_comparison
from utils.plotting.velocity_comparison import plot_solution_comparison

CURRENT_DIR = Path(__file__).parent.resolve()

# -------------------- Problem and pipeline configuration ------------------------------ #
problem: Problem = Problems.raj_one
problem = replace(problem, domain_timespan=1.0)

# general simulation parameters
n_nodes_les: int = 9
temporal_refinement: int = 1
courant_les: float = 1.0

simulation_mode = SimulationMode.TAU_BASED
tau_model = TauModel.FOUR_PARAMS

custom_hp = TD3Hyperparameters(total_episodes=500)

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
    domain_length=problem.domain_length,
    n_wavenumber_bins=disc_cfg.n_wavenumber_bins,
)

tau_ann_config = TauANNConfig(
    tau_model=tau_model,
    n_wavenumber_bins=disc_cfg.n_wavenumber_bins,
    n_coefficients=4,
    ann_path=paths.ann_model,
    n_skip_steps=1,
    reward_weight_energy=1.0,
    reward_spectral_exponent=5.0 / 3.0,
)

trained_tau_ann = run_td3_tau_ann_training(
    problem=problem,
    disc_config=disc_cfg,
    tau_ann_config=tau_ann_config,
    master_path=paths.master,
    proj_ref_schedule=proj_ref_schedule,
)

# ----------------------------------------- LES solvers ------------------------------------------ #
solver_tau_base = SolverBase(
    problem,
    disc_cfg,
    simulation_mode=simulation_mode,
    tau_model=tau_model,
    master_path=paths.les_four,
)
solver_tau_base.run_simulation()
solver_tau_base.post_processing()

# Run LES using trained RL model
solver_tau_ann = SolverCoupled(
    problem,
    disc_cfg,
    tau_model=tau_model,
    master_path=paths.ann_data,
    ann_path=paths.ann_model,
)
solver_tau_ann.run_simulation()
solver_tau_ann.post_processing()

# -------------------------------------- Plotting --------------------------------------- #
plot_solution_comparison(
    configs=create_velocity_plot_configs(paths, disc_cfg),
    output_path=paths.master,
    filename="comparison_dns_sgsp.png",
)

plot_energy_comparison(
    paths=paths,
    output_path=paths.master,
    domain_length=problem.domain_length,
)
