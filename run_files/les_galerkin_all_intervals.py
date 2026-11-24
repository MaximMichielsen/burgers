from pathlib import Path


from setup.config_discretization import DiscretizationConfig
from setup.constants import Z_LENGTH_DNS, INTERVALS, N_NODES_LES, DT_LES
from setup.problems import Problem
from solvers.solver_base_opt import SolverBaseOpt, DNSLoader, SimulationMode
from utils.pathing import get_velocity_profiles_path, get_forcing_path
from utils.profile_plotting import plot_profile_comparisons_les_vs_reference

root = Path(__file__).resolve().parent.parent

dt = DT_LES
n_nodes = N_NODES_LES
include_nonlinear = False
simulation_mode = SimulationMode.NO_MODEL
tau_model = None

velocity_path, velocity_times_path = get_velocity_profiles_path(root, n_nodes)
forcing_path, forcing_times_path = get_forcing_path(root, n_nodes, dt)


for t_start, t_end in INTERVALS:
    interval_dir = f"t0_{t_start:.2f}_tf_{t_end:.2f}"
    master_path = (
        root
        / "runs"
        / "LES_sweeps"
        / "galerkin"
        / interval_dir
        / f"n{n_nodes}_dt{dt}_{simulation_mode.value}"
    )

    dns_problem = Problem(
        name="dns_testing", domain_length=Z_LENGTH_DNS, t_start=t_start, t_end=t_end
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

    intervals_dir = root / "runs" / "DNS" / f"dns_intervals_dt_{dt}"

    plot_profile_comparisons_les_vs_reference(
        ref_intervals_dir=intervals_dir,
        les_dir=master_path,
        t_start=t_start,
    )
