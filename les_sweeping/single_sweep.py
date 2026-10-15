from dataclasses import replace
from pathlib import Path
import numpy as np

from ml.reference_scheduler import ReferenceTrajectory
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SolverBase, SimulationMode, TauModel
from utils.forcing_pathing import (
    get_forcing_path,
    get_times_path,
    get_velocity_profiles_path,
)
from utils.profile_plotting import plot_profile_comparisons

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEAN_PROFILE_DIR = PROJECT_ROOT / "dns_data" / "intervals"


def run_offset_les_simulation(
        external_coeff_val: float,
        n_nodes_les: int = 17,
        z_length: float = 2.0,
        dt: float = 1e-3,
        t_start: float = 308.0101,
        t_end: float = 309.01,
        simulation_mode: SimulationMode = SimulationMode.TAU_BASED,
        tau_model: TauModel = TauModel.TWO_PARAMS,
):
    external_coefficients = np.ones(tau_model.output_dimensions) * external_coeff_val

    # Data loading logic adapted from single_les_run_2.py[cite: 9]
    forcing_path = get_forcing_path(
        project_root=PROJECT_ROOT, n_nodes_les=n_nodes_les, dt=dt
    )
    forcing_projected = np.load(forcing_path)
    times_forcing = np.load(get_times_path(PROJECT_ROOT, dt))

    problem = Problem(
        name="tcf_1d",
        domain_length=z_length,
        reynolds=100.0,
        initial_condition=None,
        forcing=forcing_projected,
        forcing_times=times_forcing,
        forcing_is_steady=False,
        boundary_condition_type="fixed",
        boundary_condition_value=0.0,
        t_start=t_start,
        t_end=t_end,
    )

    disc_config = DiscretizationConfig(
        problem=problem,
        n_nodes_les=n_nodes_les,
        n_nodes_dns=513,
        dt_les=dt,
    )

    reference_trajectory = ReferenceTrajectory(
        mean_target_profile_dir=MEAN_PROFILE_DIR,
        projected_w_path=get_velocity_profiles_path(PROJECT_ROOT, n_nodes_les),
        times_path=PROJECT_ROOT / "dns_data" / "curated" / "dat" / "times.npy",
        n_nodes_les=n_nodes_les,
        t_start=t_start,
        t_end=t_end,
    )

    problem = replace(problem, initial_condition=reference_trajectory.initial_condition)

    master_path = (
            PROJECT_ROOT
            / "solver_data"
            / "les_sweeps"
            / "single_sweep"
            / f"t_{t_start}_{t_end}"
            / f"les_n{n_nodes_les}_p{tau_model.output_dimensions}_dt_{dt}_coeff_{external_coeff_val:.1f}"
    )

    solver_les = SolverBase(
        problem=problem,
        disc_config=disc_config,
        simulation_mode=simulation_mode,
        master_path=master_path,
        tau_model=tau_model,
        external_correction_coefficients=external_coefficients,
    )

    solver_les.run_simulation()
    solver_les.post_plotting()

    # Generate the publication-quality comparison plot for this run[cite: 7]
    plot_profile_comparisons(
        intervals_dir=MEAN_PROFILE_DIR,
        les_base_dir=master_path,
        t_start=t_start,
        z_domain=(0.0, z_length),
    )
    print(f"[SUCCESS] Completed simulation and plotting for coefficient {external_coeff_val}")


if __name__ == "__main__":
    coeffs = np.arange(0.9, 5.5, 0.1)
    print(f"\n--- Running Baseline offsets with coefficients: ---\n {coeffs}")

    for coeff in coeffs:
        coeff = round(coeff, 1)
        print(f"\n--- Running simulation with external coefficient = {coeff} ---")
        run_offset_les_simulation(external_coeff_val=coeff)