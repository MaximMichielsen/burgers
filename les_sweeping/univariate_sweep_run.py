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


def run_univariate_sweep(
    external_coefficients: tuple[float, ...] | list[float] | np.ndarray,
    sweep_tag: str,
    n_nodes_les: int,
    dt: float,
    t_start: float,
    t_end: float,
    tau_model: TauModel = TauModel.TWO_PARAMS_BASE,
    z_length: float = Z_LENGTH_DNS,
    simulation_mode: SimulationMode = SimulationMode.VMS,
):
    """Run one univariate coefficient case using the current LES solver stack."""
    coeffs_array = np.asarray(external_coefficients, dtype=float)

    if coeffs_array.ndim != 1:
        raise ValueError("external_coefficients must be a one-dimensional sequence.")
    if coeffs_array.size != tau_model.output_dimensions:
        raise ValueError(
            "Expected "
            f"{tau_model.output_dimensions} external correction coefficients for "
            f"{tau_model.value}, got {coeffs_array.size}."
        )

    forcing_path, _ = get_forcing_path(PROJECT_ROOT, n_nodes_les, dt)
    velocity_path, _ = get_velocity_profiles_path(PROJECT_ROOT, n_nodes_les)

    interval_dir = f"t0_{t_start:.2f}_tf_{t_end:.2f}"
    master_path = (
        PROJECT_ROOT
        / "runs"
        / "LES_sweeps"
        / "univariate"
        / interval_dir
        / sweep_tag
        / (
            f"n{n_nodes_les}_dt{dt}_{simulation_mode.value}_{tau_model.value}"
            f"_c{'_'.join(f'{value:.2f}' for value in coeffs_array)}"
        )
    )

    dns_problem = Problem(
        name="dns_testing",
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
    print(coeffs_array)
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

    plot_profile_comparisons_les_vs_reference(
        ref_intervals_dir=MEAN_PROFILE_DIR,
        les_dir=master_path,
        t_start=t_start,
        z_domain=(0.0, z_length),
    )

    coeff_text = ", ".join(f"{value:.2f}" for value in coeffs_array)
    print(f"[SUCCESS] Completed {sweep_tag} for external coefficients [{coeff_text}]")


if __name__ == "__main__":
    n_nodes = N_NODES_LES
    dt = DT_LES
    tau_model = TauModel.TWO_PARAMS_BASE
    intervals = INTERVALS[0], INTERVALS[4]
    baseline_vals = (1.0, 2.0, 5.0)
    sweep_range = np.round(np.arange(1.0, 8.1, 0.2), 2)

    # Run one sweep per external-correction coefficient supported by the model.
    for t_start, t_end in intervals:
        for baseline_val in baseline_vals:
            baseline_coefficients = np.full(
                tau_model.output_dimensions, baseline_val, dtype=float
            )

            coefficient_index = 1
            sweep_tag = f"vary_c{coefficient_index + 1}"
            print(
                f"\\n--- Varying c{coefficient_index + 1}; "
                f"all other coefficients = {baseline_val} ---"
            )

            for value in sweep_range:
                coefficients = baseline_coefficients.copy()
                coefficients[coefficient_index] = round(float(value), 2)

                print(
                    "Executing coefficients = "
                    + ", ".join(f"{coefficient:.2f}" for coefficient in coefficients)
                )
                run_univariate_sweep(
                    external_coefficients=coefficients,
                    sweep_tag=sweep_tag,
                    n_nodes_les=n_nodes,
                    dt=dt,
                    tau_model=tau_model,
                    t_start=t_start,
                    t_end=t_end,
                )
