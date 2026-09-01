from dataclasses import replace
from pathlib import Path

from dissipation_evolution import plot_dissipation_comparison
from energy_evolution import plot_energy_comparison
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem, Problems
from solvers.solver_base import SolverBase
from utils.pipeline_utils import run_dns, resolve_pathing
from velocity_profiles import plot_solution_comparison, create_velocity_plot_configs

CURRENT_DIR = Path(__file__).parent.resolve()

# -------------------- Problem and pipeline configuration ------------------------------ #
problem: Problem = Problems.raj_one
problem = replace(problem, domain_timespan=1.0)

# general simulation parameters
n_nodes_les: int = 9
temporal_refinement: int = 1
courant_les: float = 0.1

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

# ----------------------------------------- LES solvers ------------------------------------------ #
solver_shakib_one = SolverBase(
    problem,
    disc_cfg,
    simulation_mode="tau_model",
    tau_model="2",
    master_path=paths.les_shakib_one_data,
)
solver_shakib_one.run_simulation()
solver_shakib_one.post_processing()

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

plot_dissipation_comparison(
    paths, paths.master, problem.viscosity, problem.domain_length
)
