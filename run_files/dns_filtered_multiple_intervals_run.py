import gc
from pathlib import Path

from matplotlib import pyplot as plt

from setup.config_discretization import DiscretizationConfig
from setup.constants import Z_LENGTH_DNS, N_NODES_DNS, DT_DNS, INTERVALS
from setup.problems import Problem
from solvers.solver_base_opt import SolverBaseOpt, DNSLoader, SimulationMode
from utils.pathing import get_velocity_profiles_path, get_forcing_path

root = Path(__file__).resolve().parent.parent
dt = DT_DNS
n_nodes = N_NODES_DNS

for t_start, t_end in INTERVALS[-2:]:
    interval_dir = f"t0_{t_start:.2f}_tf_{t_end:.2f}"
    master_path = root / "runs" / "DNS" / "filtered" / interval_dir

    velocity_path = get_velocity_profiles_path(n_nodes)
    forcing_path = get_forcing_path(n_nodes, is_filtered=True)

    dns_problem = Problem(
        name="dns_filtered", domain_length=Z_LENGTH_DNS, t_start=t_start, t_end=t_end
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

    plt.close("all")

    del dns_solver, dns_loader, dns_disc_config, dns_problem
    gc.collect()
