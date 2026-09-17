from dataclasses import replace
from pathlib import Path

from ml.projection_schedule import ProjectionReferenceSchedule
from ml.tau_ann import TauANNConfig, Scope
from ml.training.td3 import TD3Hyperparameters, TD3Trainer
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem, Problems
from solvers.solver_base import SimulationMode, SolverBase, TauModel
from solvers.solver_coupled import SolverCoupled
from utils.pipeline_utils import resolve_pathing, run_dns
from utils.plotting.configs import create_velocity_plot_configs
from utils.plotting.energy_evolution import plot_energy_comparison
from utils.plotting.velocity_comparison import plot_solution_comparison

# TODO: Advanced Hybrid Grouped-Local Closure & Reward Pipeline
# -----------------------------------------------------------------------------
# 1. Expand Local Input Stencil (n_points = 8): done
#    - Update `compute_local_energy_spectrum` stencil size from 4 to 8 nodes.
#    - Allows `wavenumbers > 0` to yield 3 positive spectral bins per node instead of 1 scalar.
#    - Normalize each local spectrum by its local energy sum before concatenating into state vector.

# 2. Implement Grouped Spatial Mapping: done
#    - Divide grid domain into M local groups/patches (e.g., M = 4 or 8 groups across N nodes).
#    - Add `map_grouped_actions_to_nodes()` helper to map group residual actions to finite element nodes.

# 3. Hybrid ANN Architecture (Global Base + Grouped Local Residuals): done
#    - Update `TauANN` output layer to split into two heads:
#        a) Global Head: 1 scalar `c_global` (base macro dissipation level).
#        b) Grouped-Local Head: M residual outputs `\delta c_m` (local patch corrections).
#    - Compute final coefficient profile: c(x) = clamp(c_global + \delta c_group(x), min_action, max_action).

# 4. Multi-Fidelity Reward Function (Spectral + Shock/Gradient Penalty):
#    - Add local spatial gradient penalty (du/dx)^2 / enstrophy term to `compute_reward`.
#    - Penalize unphysical Gibbs oscillations near shocks to force velocity profile alignment.
#    - Add `reward_weight_shock` to `TauANNConfig` for hyperparameter tuning.

# 5. Diagnostic Benchmarking (Deterministic 4-Way Comparison):
#    - Evaluate and plot:
#        1. Baseline (Uncontrolled / Default Tau)
#        2. Final Global Policy
#        3. Final Grouped-Local Hybrid Policy
#        4. Best Raw Prescribed Trajectory (Episode Replay)
#    - Save side-by-side plots for Velocity Profiles, Kinetic Energy, and Energy Spectrum.

# 6. When the ANN is run with 3 or 4 parameters, it should learn to shut of the extra parameters
#    in order to approach performance of the 2 parameter model, if this outclasses both.

# 7. Add different types of randomized exploration phases, such as close to base model, more wild, single parameter variations or removed (extra) parameter mode

# 8. change input stencil to use velocity values and u_x instead of spectral components of local stencils

# 9. improve post visualization of local corrections applied over time and space

# 10. add local input stencil to visualization process: DIY

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

TOTAL_EPISODES: int = 150
max_action = 1.0
input_scope = Scope.LOCAL
output_scope = Scope.HYBRID
hp_td3 = TD3Hyperparameters(total_episodes=TOTAL_EPISODES, max_action=max_action)

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
    input_scope=input_scope,
    output_scope=output_scope,
    max_action=max_action,
    input_scope_mode="spatial",
    n_local_stencil_points=4,
    n_local_action_groups=4,
)

td3_trainer = TD3Trainer(
    problem=problem,
    disc_config=disc_cfg,
    ann_config=td3_config,
    master_path=paths.master,
    proj_ref_schedule=proj_ref_schedule,
    hp=hp_td3,
)

td3_model, best_action_sequence = td3_trainer.run_td3_tau_ann_training()
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

# Run LES using best action sequence
solver_prescribed = SolverCoupled(
    problem,
    disc_cfg,
    tau_model=tau_model,
    master_path=paths.prescribed_action,
    ann_path=paths.td3_model,
    ann_config=td3_config,
    prescribed_action_trajectory=best_action_sequence,
)
solver_prescribed.run_simulation()
solver_prescribed.post_processing()


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
