from dataclasses import replace
from pathlib import Path
import numpy as np

from ml.original.reference_scheduler import ReferenceTrajectory
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SolverBase, SimulationMode, TauModel
from utils.forcing_pathing import (
    get_forcing_path,
    get_forcing_times_path,
    get_velocity_profiles_path,
)
from utils.profile_plotting import plot_profile_comparisons

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEAN_PROFILE_DIR = PROJECT_ROOT / "dns_data" / "intervals"


def run_offset_les_simulation(
    external_coefficients: tuple[float, float] | list[float] | np.ndarray,
    sweep_tag: str,
    n_nodes_les: int = 17,
    z_length: float = 2.0,
    dt: float = 1e-3,
    t_start: float = 308.0101,
    t_end: float = 309.01,
    simulation_mode: SimulationMode = SimulationMode.TAU_BASED,
    tau_model: TauModel = TauModel.TWO_PARAMS,
):
    c1, c2 = external_coefficients
    coeffs_array = np.array([c1, c2], dtype=float)

    # Data loading logic adapted from single_sweep_3.py[cite: 16]
    forcing_path = get_forcing_path(
        project_root=PROJECT_ROOT, n_nodes_les=n_nodes_les, dt=dt
    )
    forcing_projected = np.load(forcing_path)
    times_forcing = np.load(get_forcing_times_path(PROJECT_ROOT, dt))

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

    # Isolated folder structure integrated with les_sweeps[cite: 15, 16]
    master_path = (
        PROJECT_ROOT
        / "solver_data"
        / "les_sweeps"
        / f"double_sweep_{sweep_tag}"
        / f"t_{t_start}_{t_end}"
        / f"les_n{n_nodes_les}_p{tau_model.output_dimensions}_dt_{dt}_c1_{c1:.2f}_c2_{c2:.2f}"
    )

    solver_les = SolverBase(
        problem=problem,
        disc_config=disc_config,
        simulation_mode=simulation_mode,
        master_path=master_path,
        tau_model=tau_model,
        external_correction_coefficients=coeffs_array,
    )

    solver_les.run_simulation()
    solver_les.post_plotting()

    # Generate publication-quality profile comparisons automatically[cite: 7, 16]
    plot_profile_comparisons(
        intervals_dir=MEAN_PROFILE_DIR,
        les_base_dir=master_path,
        t_start=t_start,
        z_domain=(0.0, z_length),
    )
    print(
        f"[SUCCESS] Completed simulation and plotting for c1 = {c1:.2f}, c2 = {c2:.2f}"
    )


if __name__ == "__main__":
    baseline_val = 4.9
    sweep_range = np.round(np.arange(3.0, 5.1, 0.2), 2)

    # ------------------------------------------------------------------ #
    # Sweep 1: Vary c1, keep c2 fixed at baseline_val[cite: 15]
    # ------------------------------------------------------------------ #
    print(f"\n--- Running Sweep 1: Varying c1, c2 = {baseline_val} ---")
    for c1 in sweep_range:
        c1 = round(float(c1), 2)
        print(f"Executing c1 = {c1:.2f}, c2 = {baseline_val:.2f}")
        run_offset_les_simulation(
            external_coefficients=(c1, baseline_val),
            sweep_tag="vary_c1",
        )

    # ------------------------------------------------------------------ #
    # Sweep 2: Vary c2, keep c1 fixed at baseline_val[cite: 15]
    # ------------------------------------------------------------------ #
    print(f"\n--- Running Sweep 2: Varying c2, c1 = {baseline_val} ---")
    for c2 in sweep_range:
        c2 = round(float(c2), 2)
        print(f"Executing c1 = {baseline_val:.2f}, c2 = {c2:.2f}")
        run_offset_les_simulation(
            external_coefficients=(baseline_val, c2),
            sweep_tag="vary_c2",
        )
