import shutil
from pathlib import Path

import numpy as np

from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.dns_wrapper import DNSDataForcing
from solvers.solver_base import SolverBase, TauModel, SimulationMode
from utils.dns_file_adapter import DNSDataReader
from utils.pipeline_utils import resolve_pathing, run_dns
from utils.plotting.configs import create_velocity_plot_configs
from utils.plotting.velocity_comparison import plot_solution_comparison

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).parent.resolve()

CRD_PATH = PROJECT_ROOT / "dns_raw_data" / "burgers_1D.crd"
DAT_PATH = PROJECT_ROOT / "dns_raw_data" / "burgers_1D.dat"

N_NODES_LES: int = 128
COURANT_LES: float = 0.5

def main():
    # -------------------------------------------------------------------------
    # 1. Setup paths and directories
    # -------------------------------------------------------------------------

    dns_csv_folder = CURRENT_DIR / "dns_cache" / "dns_reference_csvs"

    # Wipe directory to prevent mixing old files with new run
    if dns_csv_folder.exists():
        shutil.rmtree(dns_csv_folder)
    dns_csv_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Instantiate forcing function and reader
    # -------------------------------------------------------------------------
    dns_reader = DNSDataReader(CRD_PATH, DAT_PATH)
    dns_forcing_fn = DNSDataForcing(
        crd_filepath=str(CRD_PATH), dat_filepath=str(DAT_PATH)
    )

    # -------------------------------------------------------------------------
    # 3. Domain & Timespan setup (Starting at actual t_0 ~ 308)
    # -------------------------------------------------------------------------
    # Index 0 is t ≈ 308.0101
    start_idx = 0
    t_start = dns_forcing_fn.t_arr[start_idx]
    t_end = dns_forcing_fn.t_arr[-1]

    spatial_domain_len = float(dns_forcing_fn.z_arr[-1] - dns_forcing_fn.z_arr[0])
    effective_timespan = float(t_end - t_start)

    # Export reference CSVs with stride to save disk space
    dns_reader.export_to_csv_directory(target_dir=dns_csv_folder, stride=500)

    # -------------------------------------------------------------------------
    # 4. Set up Problem Definition
    # -------------------------------------------------------------------------
    problem = Problem(
        name="tcf_1d_forcing_verification",
        domain_length=spatial_domain_len,
        domain_timespan=effective_timespan,
        reynolds=100.0,
        # Takes the actual initial snapshot at index 0 (t ≈ 308.01)
        initial_condition=lambda x: np.interp(
            x, dns_reader.x_mesh, dns_reader.w_solutions[0]
        ),
        forcing=dns_forcing_fn,
        forcing_is_steady=False,
        boundary_condition_type="fixed",
        boundary_condition_value=0.0,
    )

    paths = resolve_pathing(problem.name, CURRENT_DIR)
    paths.dns_data = dns_csv_folder

    # -------------------------------------------------------------------------
    # 5. Configure Discretization and Run Simulations
    # -------------------------------------------------------------------------
    disc_cfg = DiscretizationConfig(
        n_nodes_les=N_NODES_LES,
        temporal_refinement=1,
        courant_les=COURANT_LES,
        domain_length=problem.domain_length,
    )

    disc_cfg.n_nodes_dns = 512

    # Run DNS
    run_dns(CURRENT_DIR / "dns_cache", problem, disc_cfg, paths)

    # Run LES (Tau-based)
    solver = SolverBase(
        problem=problem,
        disc_config=disc_cfg,
        simulation_mode=SimulationMode.DNS,
        tau_model=TauModel.TWO_PARAMS,
        master_path=paths.dns_forcing,
    )

    solver.run_simulation()
    solver.post_processing()

    # solver = SolverBase(
    #     problem=problem,
    #     disc_config=disc_cfg,
    #     simulation_mode=SimulationMode.TAU_BASED,
    #     tau_model=TauModel.THREE_PARAMS,
    #     master_path=paths.les_three,
    # )
    #
    # solver.run_simulation()
    #
    # solver = SolverBase(
    #     problem=problem,
    #     disc_config=disc_cfg,
    #     simulation_mode=SimulationMode.TAU_BASED,
    #     tau_model=TauModel.FOUR_PARAMS,
    #     master_path=paths.les_four,
    # )
    # solver.run_simulation()

    # -------------------------------------------------------------------------
    # 6. Plot Results
    # -------------------------------------------------------------------------
    plot_solution_comparison(
        configs=create_velocity_plot_configs(paths, disc_cfg),
        output_path=paths.master,
    )


if __name__ == "__main__":
    main()
