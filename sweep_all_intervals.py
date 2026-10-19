from dataclasses import replace
from pathlib import Path
import re
import shutil

import numpy as np

from ml.reference_scheduler import ReferenceTrajectory
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SolverBase, SimulationMode, TauModel
from utils.forcing_pathing import (
    get_forcing_path,
    get_forcing_times_path,
    get_velocity_profiles_path,
)
from utils.profile_plotting import plot_profile_comparisons

PROJECT_ROOT = Path(__file__).resolve().parent
MEAN_PROFILE_DIR = PROJECT_ROOT / "dns_data" / "intervals"

n_nodes_les = 17
n_nodes_dns = 513
z_length = 2.0
dt = 1e-3

simulation_mode = SimulationMode.TAU_BASED
tau_model = TauModel.TWO_PARAMS


def discover_intervals(intervals_dir: Path) -> list[tuple[float, float]]:
    """Dynamically discover t_start and t_end float pairs from interval files."""
    intervals = []
    for f in sorted(intervals_dir.glob("w_mean_*.npy")):
        match = re.search(r"w_mean_(.+?)_(.+?)\.npy$", f.name)
        if match:
            intervals.append((float(match.group(1)), float(match.group(2))))
    return intervals


if __name__ == "__main__":
    intervals = discover_intervals(MEAN_PROFILE_DIR)
    coeffs = np.arange(5.1, 7.05, 0.1)

    print(f"\n--- Running Coefficient Sweep across {len(intervals)} intervals ---")
    print(f"Coefficients: {[round(c, 1) for c in coeffs]}")

    # Load shared forcing data once
    forcing_path = get_forcing_path(
        project_root=PROJECT_ROOT, n_nodes_les=n_nodes_les, dt=dt
    )
    forcing_projected = np.load(forcing_path)
    times_forcing = np.load(get_forcing_times_path(PROJECT_ROOT, dt))

    # Target directory for centralizing profile comparison plots across all interval sweeps
    profiles_output_dir = (
        PROJECT_ROOT / "solver_data" / "les_sweeps" / "all_intervals_profiles"
    )
    profiles_output_dir.mkdir(parents=True, exist_ok=True)

    for t_start, t_end in intervals:
        timespan = t_end - t_start
        print("\n" + "=" * 60)
        print(f"Interval: t in [{t_start}, {t_end}] (duration: {timespan:.4f}s)")
        print("=" * 60)

        for coeff in coeffs:
            coeff = round(float(coeff), 1)
            print(
                f"\n--- Running simulation for interval [{t_start}, {t_end}] with coefficient = {coeff} ---"
            )

            external_coefficients = np.ones(tau_model.output_dimensions) * coeff

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
                n_nodes_dns=n_nodes_dns,
                dt_les=dt,
            )

            reference_trajectory = ReferenceTrajectory(
                mean_target_profile_dir=MEAN_PROFILE_DIR,
                projected_w_path=get_velocity_profiles_path(PROJECT_ROOT, n_nodes_les),
                times_path=PROJECT_ROOT / "dns_data" / "curated" / "dat" / "times.npy",
                n_nodes_les=n_nodes_les,
            )

            reference_trajectory.set_active_training_interval(
                t_start=t_start, t_end=t_end
            )

            problem = replace(
                problem, initial_condition=reference_trajectory.initial_condition
            )

            master_path = (
                PROJECT_ROOT
                / "solver_data"
                / "les_sweeps"
                / "all_intervals_sweep"
                / f"t_{t_start}_{t_end}"
                / f"les_n{n_nodes_les}_p{tau_model.output_dimensions}_dt_{dt}_coeff_{coeff:.1f}"
            )

            solver_les = SolverBase(
                problem=problem,
                disc_config=disc_config,
                simulation_mode=simulation_mode,
                master_path=master_path,
                tau_model=tau_model,
                external_correction_coefficients=external_coefficients,
            )

            # Execute simulation and post-processing
            solver_les.run_simulation()
            solver_les.post_plotting()

            plot_profile_comparisons(
                intervals_dir=MEAN_PROFILE_DIR,
                les_base_dir=master_path,
                t_start=t_start,
                z_domain=(0.0, z_length),
            )

            # Centralize a copy of the generated comparison plot
            source_plot = (
                master_path
                / "combined_comparisons"
                / f"combined_profile_comparison_{t_end:.2f}.png"
            )
            if source_plot.exists():
                dest_plot = (
                    profiles_output_dir
                    / f"combined_profile_comparison_t_{t_start}_{t_end}_coeff_{coeff:.1f}.png"
                )
                shutil.copy(source_plot, dest_plot)
                print(f"[Copied] Profile plot to: {dest_plot}")

            print(
                f"[SUCCESS] Completed interval [{t_start}, {t_end}] with coefficient {coeff}"
            )

    print("\n[ALL INTERVAL COEFFICIENT SWEEPS COMPLETED SUCCESSFULLY]")
