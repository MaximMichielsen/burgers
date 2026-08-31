from dataclasses import replace
from pathlib import Path

from new.setup.config_discretization import DiscretizationConfig
from new.setup.problems import Problem, Problems
from new.solvers.solver_base import SolverBase
from new.utils.pipeline_utils import run_dns, resolve_pathing

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
