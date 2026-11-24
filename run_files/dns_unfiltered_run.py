from pathlib import Path

from setup.config_discretization import DiscretizationConfig
from setup.constants import Z_LENGTH_DNS, N_NODES_DNS, DT_DNS
from setup.problems import Problem
from solvers.solver_base_opt import SolverBaseOpt, DNSLoader, SimulationMode

root = Path(__file__).resolve().parent.parent

t_start = 308.0101
t_end = 309.01

# True DNS run ------------------------------------------------------

dt = DT_DNS
n_nodes = N_NODES_DNS

interval_dir = f"t0_{t_start:.2f}_tf_{t_end:.2f}"
master_path = root / "runs" / "DNS" / "true_dns" / interval_dir

velocity_path = root / "dns_data" / "partitioned" / "dat" / "w.npy"
forcing_path = root / "dns_data" / "partitioned" / "dat" / "f.npy"

dns_problem = Problem(
    name="dns_true", domain_length=Z_LENGTH_DNS, t_start=t_start, t_end=t_end
)
dns_disc_config = DiscretizationConfig(dns_problem, N_NODES_DNS, dt)
dns_loader = DNSLoader(
    path_to_velocity=velocity_path,
    path_to_forcing=forcing_path,
    t_start=t_start,
)

dns_solver = SolverBaseOpt(
    dns_problem,
    dns_disc_config,
    dns_loader,
    simulation_mode=SimulationMode.NO_MODEL,
    master_path=master_path,
)
dns_solver.run_simulation()
