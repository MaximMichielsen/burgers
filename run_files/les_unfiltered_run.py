from pathlib import Path


from setup.config_discretization import DiscretizationConfig
from setup.constants import Z_LENGTH_DNS
from setup.problems import Problem
from solvers.solver_base_opt import SolverBaseOpt, DNSLoader, SimulationMode, TauModel
from utils.pathing import get_velocity_profiles_path

root = Path(__file__).resolve().parent.parent

dt = 1e-3
n_nodes = 49

include_nonlinear = False
simulation_mode = SimulationMode.VMS
tau_model = TauModel.TWO_PARAMS_BASE

t_start = 308.0101
t_end = 309.01

velocity_path, _ = get_velocity_profiles_path(root, n_nodes)
forcing_path = root / "dns_data" / "dns_timestep_forcing" / f"forcing_l2_{n_nodes}.npy"

interval_dir = f"t0_{t_start:.2f}_tf_{t_end:.2f}"
master_path = (
    root
    / "runs"
    / "LES"
    / "true_forcing"
    / f"n{n_nodes}_dt{dt}_{simulation_mode.value}_{tau_model.value}"
    / interval_dir
)

dns_problem = Problem(
    name="les_true_forcing", domain_length=Z_LENGTH_DNS, t_start=t_start, t_end=t_end
)
dns_disc_config = DiscretizationConfig(dns_problem, n_nodes, dt)
dns_loader = DNSLoader(
    path_to_velocity=velocity_path,
    path_to_forcing=forcing_path,
    t_start=t_start,
)

les_solver = SolverBaseOpt(
    dns_problem,
    dns_disc_config,
    dns_loader,
    simulation_mode=simulation_mode,
    tau_model=tau_model,
    master_path=master_path,
    include_nonlinear_vms=include_nonlinear,
)
les_solver.run_simulation()
