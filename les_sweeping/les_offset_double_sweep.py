from pathlib import Path
import numpy as np

from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SolverBase, SimulationMode, TauModel
from utils.forcing_pathing import get_forcing_path

PROJECT_ROOT = Path(__file__).resolve().parent
FILTERED_DIR = PROJECT_ROOT / "dns_data" / "projected"


def run_offset_les_simulation(
    external_coefficients: tuple[float, float] | list[float] | np.ndarray,
    sweep_tag: str,
    n_nodes_les: int = 17,
    z_length: float = 2.0,
    dt: float = 1e-3,
    t_start: float = 308.0101,
    t_end: float = 309.0100,
    simulation_mode: SimulationMode = SimulationMode.TAU_BASED,
    tau_model: TauModel = TauModel.TWO_PARAMS,
):
    c1, c2 = external_coefficients
    coeffs_array = np.array([c1, c2], dtype=float)

    ic_les = np.load(FILTERED_DIR / f"ic_linear_{n_nodes_les}.npy")
    forcing_path = get_forcing_path(
        project_root=PROJECT_ROOT, n_nodes_les=n_nodes_les, dt=dt
    )
    forcing_projected = np.load(forcing_path)

    n_timesteps, _ = np.shape(forcing_projected)
    timespan = n_timesteps * dt
    if t_end is not None:
        timespan = t_end - t_start

    problem = Problem(
        name="tcf_1d",
        domain_length=z_length,
        domain_timespan=timespan,
        reynolds=100.0,
        initial_condition=ic_les,
        forcing=forcing_projected,
        forcing_is_steady=False,
        boundary_condition_type="fixed",
        boundary_condition_value=0.0,
        t_start=t_start,
    )

    disc_config = DiscretizationConfig(
        n_nodes_les=n_nodes_les,
        n_nodes_dns=513,
        domain_timespan=1,
        temporal_refinement=1,
        courant_les=1,
        domain_length=z_length,
        set_dt_les=dt,
    )

    # Isolated folder structure for each sweep type
    output_dir = (
        PROJECT_ROOT
        / "solver_data"
        / f"les_offset_sweeps_{sweep_tag}"
        / f"les_n{n_nodes_les}_p{tau_model.output_dimensions}_dt_{dt}_c1_{c1:.2f}_c2_{c2:.2f}"
    )

    solver_les = SolverBase(
        problem=problem,
        disc_config=disc_config,
        simulation_mode=simulation_mode,
        master_path=output_dir,
        tau_model=tau_model,
        external_correction_coefficients=coeffs_array,
    )
    solver_les.run_simulation()


if __name__ == "__main__":
    baseline_val = 4.9
    sweep_range = np.round(np.arange(3.0, 5.1, 0.2), 2)

    # ------------------------------------------------------------------ #
    # Sweep 1: Vary c1, keep c2 fixed at 4.9
    # ------------------------------------------------------------------ #
    print(f"\n--- Running Sweep 1: Varying c1 ∈ [2.0, 8.0], c2 = {baseline_val} ---")
    for c1 in sweep_range:
        print(f"Executing c1 = {c1:.2f}, c2 = {baseline_val:.2f}")
        run_offset_les_simulation(
            external_coefficients=(c1, baseline_val),
            sweep_tag="vary_c1",
        )

    # ------------------------------------------------------------------ #
    # Sweep 2: Vary c2, keep c1 fixed at 4.9
    # ------------------------------------------------------------------ #
    print(f"\n--- Running Sweep 2: Varying c2 ∈ [2.0, 8.0], c1 = {baseline_val} ---")
    for c2 in sweep_range:
        print(f"Executing c1 = {baseline_val:.2f}, c2 = {c2:.2f}")
        run_offset_les_simulation(
            external_coefficients=(baseline_val, c2),
            sweep_tag="vary_c2",
        )
