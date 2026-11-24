import os

# Prevent BLAS thread oversubscription when multiprocessing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np

from setup.config_discretization import DiscretizationConfig
from setup.constants import Z_LENGTH_DNS, INTERVALS, N_NODES_LES, DT_LES
from setup.problems import Problem
from archived.solver_base_2_opt import SimulationMode, TauModel
from solvers.solver_base_opt import DNSLoader, SolverBaseOpt
from utils.pathing import get_forcing_path, get_velocity_profiles_path
from utils.profile_plotting import plot_profile_comparisons_les_vs_reference


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEAN_PROFILE_DIR = PROJECT_ROOT / "runs" / "DNS" / "dns_intervals_dt_0.001"
GRID_ROOT = PROJECT_ROOT / "runs" / "LES_sweeps" / "grid"


def get_master_path(
    coeffs_array: np.ndarray,
    n_nodes_les: int,
    dt: float,
    t_start: float,
    t_end: float,
    tau_model: TauModel,
    simulation_mode: SimulationMode,
) -> Path:
    interval_dir = f"t0_{t_start:.2f}_tf_{t_end:.2f}"
    return (
        GRID_ROOT
        / interval_dir
        / (
            f"n{n_nodes_les}_dt{dt}_{simulation_mode.value}_{tau_model.value}"
            f"_c{'_'.join(f'{value:.2f}' for value in coeffs_array)}"
        )
    )


def run_grid_point(
    external_coefficients: tuple[float, ...] | list[float] | np.ndarray,
    n_nodes_les: int,
    dt: float,
    t_start: float,
    t_end: float,
    tau_model: TauModel = TauModel.TWO_PARAMS_BASE,
    z_length: float = Z_LENGTH_DNS,
    simulation_mode: SimulationMode = SimulationMode.VMS,
    make_plots: bool = False,
) -> tuple[float, ...]:
    """Run one grid-search point with the current LES solver stack."""
    coeffs_array = np.asarray(external_coefficients, dtype=float)

    if coeffs_array.ndim != 1:
        raise ValueError("external_coefficients must be a one-dimensional sequence.")
    if coeffs_array.size != tau_model.output_dimensions:
        raise ValueError(
            f"Expected {tau_model.output_dimensions} external correction coefficients "
            f"for {tau_model.value}, got {coeffs_array.size}."
        )

    master_path = get_master_path(
        coeffs_array, n_nodes_les, dt, t_start, t_end, tau_model, simulation_mode
    )

    forcing_path = get_forcing_path(n_nodes_les, True)
    velocity_path = get_velocity_profiles_path(n_nodes_les)

    dns_problem = Problem(
        name="grid_search",
        domain_length=Z_LENGTH_DNS,
        t_start=t_start,
        t_end=t_end,
    )
    dns_disc_config = DiscretizationConfig(dns_problem, n_nodes_les, dt)
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
        external_correction_coefficients=coeffs_array,
        include_nonlinear_vms=False,
    )
    les_solver.run_simulation()

    if make_plots:
        plot_profile_comparisons_les_vs_reference(
            ref_intervals_dir=MEAN_PROFILE_DIR,
            les_dir=master_path,
            t_start=t_start,
            z_domain=(0.0, z_length),
        )

    return tuple(float(v) for v in coeffs_array)


def is_completed(master_path: Path) -> bool:
    """Check if a run directory already exists and is non-empty."""
    return master_path.exists() and any(master_path.iterdir())


if __name__ == "__main__":
    n_nodes = N_NODES_LES
    dt = DT_LES
    tau_model = TauModel.TWO_PARAMS_BASE
    simulation_mode = SimulationMode.VMS
    intervals = INTERVALS[:]

    for t_start, t_end in intervals:
        # Full [1.0, 10.0] grid at 0.5 spacing (361 total points)
        grid_axis = np.round(np.arange(1.0, 8.1, 0.5), 2)
        all_points = [
            tuple(round(float(v), 2) for v in pt)
            for pt in product(grid_axis, repeat=tau_model.output_dimensions)
        ]

        # Keep only points that haven't been computed yet (the missing L-shaped region)
        pending_points = [
            pt
            for pt in all_points
            if not is_completed(
                get_master_path(
                    np.array(pt),
                    n_nodes,
                    dt,
                    t_start,
                    t_end,
                    tau_model,
                    simulation_mode,
                )
            )
        ]

        total = len(pending_points)
        max_workers = max(1, (os.cpu_count() or 2) - 1)
        print(
            f"Grid search: {len(all_points) - total} already done, "
            f"{total} remaining -> running on {max_workers} workers"
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_grid_point,
                    pt,
                    n_nodes,
                    dt,
                    t_start,
                    t_end,
                    tau_model,
                    Z_LENGTH_DNS,
                    simulation_mode,
                    False,  # skip plotting during sweep
                ): pt
                for pt in pending_points
            }

            for idx, future in enumerate(as_completed(futures), start=1):
                pt = futures[future]
                try:
                    future.result()
                    print(f"[{idx}/{total}] [SUCCESS] {pt}")
                except Exception as exc:
                    print(f"[{idx}/{total}] [FAILED] {pt}: {exc}")
