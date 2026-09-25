from dataclasses import replace
from pathlib import Path

from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem, Problems
from solvers.solver_base import SolverBase, SimulationMode, TauModel
from utils.pipeline_utils import run_dns, resolve_pathing
from utils.plotting.configs import (
    create_velocity_plot_configs,
)
from utils.plotting.energy_evolution import plot_energy_comparison
from utils.plotting.velocity_comparison import plot_solution_comparison

# -------------------- Problem and pipeline configuration ------------------------------ #
CURRENT_DIR = Path(__file__).parent.resolve()
problem: Problem = Problems.raj_one
problem = replace(problem, domain_timespan=1.0, reynolds=100)

# simulation parameters
n_nodes_les: int = 9
temporal_refinement: int = 1
courant_les: float = 1.0

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

# ----------------------------------- DNS run / caching check ------------------------------------ #
run_dns(DNS_CACHE_ROOT, problem, disc_cfg, paths)

# ----------------------------------------- LES solvers ------------------------------------------ #
solver_no_model = SolverBase(
    problem,
    disc_cfg,
    simulation_mode=SimulationMode.NO_MODEL,
    master_path=paths.les_nm,
)
solver_no_model.run_simulation()
solver_no_model.post_processing()

solver_tau_2 = SolverBase(
    problem,
    disc_cfg,
    simulation_mode=SimulationMode.TAU_BASED,
    tau_model=TauModel.TWO_PARAMS,
    master_path=paths.les_two,
)
solver_tau_2.run_simulation()
solver_tau_2.post_processing()

solver_tau_3 = SolverBase(
    problem,
    disc_cfg,
    simulation_mode=SimulationMode.TAU_BASED,
    tau_model=TauModel.THREE_PARAMS,
    master_path=paths.les_three,
)
solver_tau_3.run_simulation()
solver_tau_3.post_processing()

solver_tau_4 = SolverBase(
    problem,
    disc_cfg,
    simulation_mode=SimulationMode.TAU_BASED,
    tau_model=TauModel.FOUR_PARAMS,
    master_path=paths.les_four,
)
solver_tau_4.run_simulation()
solver_tau_4.post_processing()

# -------------------------------------- Plotting --------------------------------------- #
plot_solution_comparison(
    configs=create_velocity_plot_configs(paths, disc_cfg),
    output_path=paths.master,
    filename="velocity_profile_comparison.png",
)

plot_energy_comparison(
    paths=paths,
    output_path=paths.master,
    domain_length=problem.domain_length,
)
