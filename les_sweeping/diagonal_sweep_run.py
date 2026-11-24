import os

# Prevent BLAS thread oversubscription when multiprocessing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from setup.config_discretization import DiscretizationConfig
from setup.constants import (
    Z_LENGTH_DNS,
    INTERVALS,
    N_NODES_LES,
    DT_LES,
    PATH_TO_DNS_REF_PROFILES,
)
from setup.problems import Problem
from archived.solver_base_2_opt import SimulationMode, TauModel
from solvers.solver_base_opt import DNSLoader, SolverBaseOpt
from utils.pathing import get_forcing_path, get_velocity_profiles_path

root = Path(__file__).resolve().parent.parent
MEAN_PROFILE_DIR = PATH_TO_DNS_REF_PROFILES


def run_diagonal_sweep(
    external_coeff_val: float,
    n_nodes: int,
    t_start: float,
    t_end: float,
    dt: float,
    tau_model: TauModel,
    simulation_mode: SimulationMode = SimulationMode.VMS,
):
    external_coefficients = np.ones(tau_model.output_dimensions) * external_coeff_val

    forcing_path = get_forcing_path(n_nodes, True)
    velocity_path = get_velocity_profiles_path(n_nodes)

    interval_dir = f"t0_{t_start:.2f}_tf_{t_end:.2f}"
    master_path = (
        root
        / "runs"
        / "LES_sweeps"
        / "diagonal"
        / interval_dir
        / f"n{n_nodes}_dt{dt}_{simulation_mode.value}_{tau_model.value}"
        / f"c_{external_coeff_val}"
    )

    dns_problem = Problem(
        name="les_sweep", domain_length=Z_LENGTH_DNS, t_start=t_start, t_end=t_end
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
        external_correction_coefficients=external_coefficients,
        include_nonlinear_vms=False,
    )
    les_solver.run_simulation()
    print(
        f"[SUCCESS] Completed simulation and plotting for coefficient {external_coeff_val}"
    )


if __name__ == "__main__":
    coeffs = np.arange(1.0, 8.1, 0.2)
    n_nodes = N_NODES_LES
    dt = DT_LES
    tau_modes = (TauModel.TWO_PARAMS_BASE, TauModel.THREE_PARAMS_BASE)

    for tau_mode in tau_modes:
        for t_start, t_end in INTERVALS:
            pending_coeffs = [round(float(coeff), 2) for coeff in coeffs]

            total = len(pending_coeffs)
            max_workers = max(1, (os.cpu_count() or 2) - 1)
            print(
                f"Diagonal sweep ({tau_mode.value}, t0={t_start:.2f}, tf={t_end:.2f}): "
                f"{total} simulations -> running on {max_workers} workers"
            )

            with ProcessPoolExecutor(
                max_workers=max_workers,
                max_tasks_per_child=1,
            ) as executor:
                futures = {
                    executor.submit(
                        run_diagonal_sweep,
                        external_coeff_val=coeff,
                        n_nodes=n_nodes,
                        t_start=t_start,
                        t_end=t_end,
                        dt=dt,
                        tau_model=tau_mode,
                    ): coeff
                    for coeff in pending_coeffs
                }

                for idx, future in enumerate(as_completed(futures), start=1):
                    coeff = futures.pop(future)
                    try:
                        future.result()
                        print(f"[{idx}/{total}] [SUCCESS] coefficient={coeff}")
                    except Exception as exc:
                        print(f"[{idx}/{total}] [FAILED] coefficient={coeff}: {exc}")
