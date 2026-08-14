"""Main execution file, runs all important blocks."""

from dataclasses import replace
from pathlib import Path

import matplotlib

from problems_and_configurations.disc_config import DiscretizationConfig
from problems_and_configurations.problems import Problems, Problem

from solvers.explicit.base_solver_rk2 import BaseRK2
from utils.plotting.dissipation_evolution import plot_dissipation_comparison
from utils.plotting.energy_evolution import plot_energy_comparison
from utils.pipeline_utils import (
    run_dns,
    resolve_pathing,
)
from utils.plotting.velocity_profiles import (
    plot_solution_comparison,
    create_velocity_plot_configs,
)


CURRENT_DIR = Path(__file__).parent.resolve()
matplotlib.use("Agg")  # needed when running on M12

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

if __name__ == "__main__":
    # ----------------------------------------- DNS data --------------------------------------------- #
    DNS_CACHE_ROOT = CURRENT_DIR / "dns_cache"
    run_dns(DNS_CACHE_ROOT, problem, disc_cfg, paths)

    # ----------------------------------------- LES solvers ------------------------------------------ #
    solver_shakib_one = BaseRK2(problem, disc_cfg, "shakib_one", paths.les_shakib_one_data)
    solver_shakib_one.run_simulation()
    solver_shakib_one.post_processing()

    solver_no_model = BaseRK2(problem, disc_cfg, "no_model", paths.les_nm_data)
    solver_no_model.run_simulation()
    solver_no_model.post_processing()

    # -------------------------------------- Plotting --------------------------------------- #
    plot_solution_comparison(
        configs=create_velocity_plot_configs(paths, disc_cfg),
        output_path=paths.master,
    )

    plot_energy_comparison(
        paths=paths,
        output_path=paths.master,
        viscosity=problem.viscosity,
        domain_length=problem.domain_length,
    )

    plot_dissipation_comparison(
        paths, paths.master, problem.viscosity, problem.domain_length
    )
