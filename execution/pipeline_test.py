from dataclasses import replace
from pathlib import Path

from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem, Problems
from solvers.solver_base import SimulationMode, TauModel, SolverBase
from utils.pipeline_utils import resolve_pathing, run_dns

CURRENT_DIR = Path(__file__).parent.resolve()

# -------------------- Problem and pipeline configuration ------------------------------ #
problem: Problem = Problems.raj_two
problem = replace(problem, domain_timespan=4.0)

# general simulation parameters
n_nodes_les: int = 9
temporal_refinement: int = 1
courant_les: float = 1.0

simulation_mode = SimulationMode.TAU_BASED
tau_model = TauModel.FOUR_PARAMS

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
solver_tau_base = SolverBase(
    problem,
    disc_cfg,
    simulation_mode=simulation_mode,
    tau_model=tau_model,
    master_path=paths.les_four,
)
solver_tau_base.print_configuration()
solver_tau_base.run_simulation()
solver_tau_base.post_processing()
