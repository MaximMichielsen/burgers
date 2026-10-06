import re
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SimulationMode, SolverBase, TauModel
from utils.forcing_pathing import get_forcing_path

# Path Configurations
PROJECT_ROOT = Path(__file__).resolve().parent
FILTERED_DIR = PROJECT_ROOT / "dns_data" / "projected"
PARSED_STAT_DIR = PROJECT_ROOT / "dns_data" / "curated" / "stat"

BASE_TEST_DIR = PROJECT_ROOT / "solver_data" / "les_offset_tests"
SINGLE_SWEEP_DIR = BASE_TEST_DIR / "single_sweep"
VARY_C1_DIR = BASE_TEST_DIR / "vary_c1"
VARY_C2_DIR = BASE_TEST_DIR / "vary_c2"
PLOTS_DIR = BASE_TEST_DIR / "comparison_plots"


# ------------------------------------------------------------------ #
# Simulation Runner Helper
# ------------------------------------------------------------------ #
def run_offset_les_simulation(
    external_coefficients: tuple[float, float] | list[float] | np.ndarray | float,
    master_path: Path,
    n_nodes_les: int = 33,
    z_length: float = 2.0,
    dt: float = 1e-3,
    t_start: float = 308.0101,
    t_end: float = 309.0100,
    simulation_mode: SimulationMode = SimulationMode.TAU_BASED,
    tau_model: TauModel = TauModel.TWO_PARAMS,
):
    """Executes a single 1D LES simulation given correction coefficients."""
    if np.isscalar(external_coefficients):
        coeffs_array = np.ones(tau_model.output_dimensions) * external_coefficients
    else:
        coeffs_array = np.array(external_coefficients, dtype=float)

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

    solver_les = SolverBase(
        problem=problem,
        disc_config=disc_config,
        simulation_mode=simulation_mode,
        master_path=master_path,
        tau_model=tau_model,
        external_correction_coefficients=coeffs_array,
    )
    solver_les.run_simulation()


# ------------------------------------------------------------------ #
# Single Sweep Execution & Plotting
# ------------------------------------------------------------------ #
def execute_single_sweep(coeffs: np.ndarray, z_domain: tuple[float, float] = (0.0, 2.0)):
    """Runs single coefficient simulations, plots results, and returns optimal baseline coeff."""
    print(f"\n=======================================================")
    print(f"  STEP 1: Single Parameter Sensitivity Sweep")
    print(f"=======================================================")

    for coeff in coeffs:
        coeff = round(float(coeff), 2)
        sim_dir = (
            SINGLE_SWEEP_DIR / f"les_n33_p2_dt_0.001_coeff_{coeff:.2f}"
        )
        print(f"Running simulation with external coefficient = {coeff:.2f}...")
        run_offset_les_simulation(
            external_coefficients=coeff,
            master_path=sim_dir,
        )

    # Process and evaluate profiles against DNS frame 0
    mean_w_profiles = np.load(PARSED_STAT_DIR / "mean_w.npy")
    dns_profile = mean_w_profiles[0]
    mesh_dns = np.linspace(z_domain[0], z_domain[1], len(dns_profile))

    l2_errors = []
    parsed_runs = []

    for coeff in coeffs:
        coeff = round(float(coeff), 2)
        profiles_dir = (
            SINGLE_SWEEP_DIR / f"les_n33_p2_dt_0.001_coeff_{coeff:.2f}" / "mean_profiles"
        )
        if profiles_dir.exists():
            npy_files = sorted(profiles_dir.glob("mean_w_*.npy"))
            if npy_files:
                les_mean = np.load(npy_files[-1])
                mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_mean))
                dns_interp = (
                    np.interp(mesh_les, mesh_dns, dns_profile)
                    if len(dns_profile) != len(les_mean)
                    else dns_profile
                )
                rel_err = (
                    np.linalg.norm(les_mean - dns_interp)
                    / (np.linalg.norm(dns_interp) + 1e-12)
                ) * 100.0
                l2_errors.append((coeff, rel_err))
                parsed_runs.append((coeff, les_mean, mesh_les))

    # Single Sweep Plotting
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), dpi=200, gridspec_kw={"width_ratios": [1.3, 1]}
    )

    norm = plt.Normalize(vmin=coeffs.min(), vmax=coeffs.max())
    cmap = cm.plasma

    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.plot(
        mesh_dns,
        dns_profile,
        color="black",
        linestyle="-",
        linewidth=2.5,
        label="DNS Reference",
        zorder=5,
    )

    for coeff, les_mean, mesh_les in parsed_runs:
        ax1.plot(
            mesh_les,
            les_mean,
            color=cmap(norm(coeff)),
            alpha=0.6,
            linewidth=1.1,
            zorder=3,
        )

    ax1.set_title(r"Mean Velocity Profiles $\langle w \rangle$", fontsize=12)
    ax1.set_xlabel(r"Domain Coordinate $z$", fontsize=11)
    ax1.set_ylabel(r"Mean Velocity $\langle w \rangle$", fontsize=11)
    ax1.set_xlim(z_domain)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, orientation="vertical", pad=0.02)
    cbar.set_label("External Coefficient", fontsize=10)

    # Error Curve
    coeffs_arr, errs_arr = zip(*l2_errors)
    ax2.plot(
        coeffs_arr, errs_arr, color="tab:blue", marker="o", markersize=4, linestyle="-"
    )

    min_idx = np.argmin(errs_arr)
    opt_coeff = coeffs_arr[min_idx]
    opt_err = errs_arr[min_idx]

    ax2.plot(
        opt_coeff,
        opt_err,
        color="tab:red",
        marker="d",
        markersize=12,
        label=f"Optimal: {opt_coeff:.2f} ({opt_err:.1f}%)",
    )

    ax2.set_title(r"Relative $L_2$ Error vs. Coefficient", fontsize=12)
    ax2.set_xlabel("External Coefficient", fontsize=11)
    ax2.set_ylabel("Relative $L_2$ Error (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    plt.suptitle("Single Parameter Sensitivity Sweep", fontsize=14, y=0.98)
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / "single_sweep_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"\nSingle sweep comparison saved to: {save_path}")
    print(f"Optimal baseline coefficient found: {opt_coeff:.2f} (Error: {opt_err:.1f}%)")
    return opt_coeff


# ------------------------------------------------------------------ #
# Double Sweep Plotting Helper
# ------------------------------------------------------------------ #
def create_heatmap_and_error_plot(
    sweep_dir: Path,
    sweep_tag: str,
    varied_coeff_name: str,
    fixed_coeff_val: float,
    z_domain: tuple[float, float] = (0.0, 2.0),
):
    """Generates 2D Spatial Heatmap and Error Curve for decoupled sweeps."""
    if not sweep_dir.exists():
        print(f"Directory {sweep_dir} does not exist.")
        return

    mean_w_profiles = np.load(PARSED_STAT_DIR / "mean_w.npy")
    dns_profile = mean_w_profiles[0]
    mesh_dns = np.linspace(z_domain[0], z_domain[1], len(dns_profile))

    coeff_folders = sorted(sweep_dir.glob("les_n33_p2_dt_0.001_c1_*_c2_*"))

    parsed_runs = []
    for d in coeff_folders:
        match = re.search(r"c1_(\d+\.\d+)_c2_(\d+\.\d+)$", d.name)
        if match:
            c1, c2 = float(match.group(1)), float(match.group(2))
            profiles_dir = d / "mean_profiles"
            if profiles_dir.exists():
                npy_files = sorted(profiles_dir.glob("mean_w_*.npy"))
                if npy_files:
                    target_file = npy_files[-1]
                    var_val = c1 if varied_coeff_name == "c1" else c2
                    parsed_runs.append((var_val, target_file))

    parsed_runs.sort(key=lambda x: x[0])
    varied_coeffs = np.array([run[0] for run in parsed_runs])

    velocity_matrix = []
    l2_errors = []

    for var_val, file_path in parsed_runs:
        les_mean = np.load(file_path)
        velocity_matrix.append(les_mean)

        mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_mean))
        dns_interp = (
            np.interp(mesh_les, mesh_dns, dns_profile)
            if len(dns_profile) != len(les_mean)
            else dns_profile
        )

        rel_err = (
            np.linalg.norm(les_mean - dns_interp)
            / (np.linalg.norm(dns_interp) + 1e-12)
        ) * 100.0
        l2_errors.append(rel_err)

    velocity_matrix = np.array(velocity_matrix)
    mesh_les = np.linspace(z_domain[0], z_domain[1], velocity_matrix.shape[1])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), dpi=200, gridspec_kw={"width_ratios": [1.4, 1]}
    )

    Z, C = np.meshgrid(mesh_les, varied_coeffs)
    contour = ax1.contourf(Z, C, velocity_matrix, levels=40, cmap="viridis")
    cbar = fig.colorbar(contour, ax=ax1, pad=0.02)
    cbar.set_label(r"Mean Velocity $\langle w \rangle$", fontsize=11)

    ax1.set_xlabel(r"Domain Coordinate $z$", fontsize=11)
    ax1.set_ylabel(f"External Coefficient ${varied_coeff_name}$", fontsize=11)
    ax1.set_title(
        f"Spatial Velocity Heatmap $\\langle w \\rangle(z, {varied_coeff_name})$",
        fontsize=12,
    )

    ax2.plot(
        varied_coeffs,
        l2_errors,
        color="tab:blue",
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=1.5,
    )

    min_idx = np.argmin(l2_errors)
    opt_val = varied_coeffs[min_idx]
    opt_err = l2_errors[min_idx]

    ax2.plot(
        opt_val,
        opt_err,
        color="tab:red",
        marker="d",
        markersize=10,
        label=f"Optimal {varied_coeff_name}: {opt_val:.2f} ({opt_err:.1f}%)",
    )

    other_coeff_name = "c_2" if varied_coeff_name == "c1" else "c_1"
    ax2.set_xlabel(f"External Coefficient ${varied_coeff_name}$", fontsize=11)
    ax2.set_ylabel(r"Relative $L_2$ Error (%)", fontsize=11)
    ax2.set_title(
        f"Relative $L_2$ Error vs. ${varied_coeff_name}$ (with ${other_coeff_name}={fixed_coeff_val:.2f}$)",
        fontsize=12,
    )
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    plt.suptitle(
        f"LES Parameter Sensitivity Sweep ({varied_coeff_name.upper()} decoupled)",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout()

    save_file = PLOTS_DIR / f"heatmap_sweep_{sweep_tag}.png"
    plt.savefig(save_file, dpi=300)
    print(f"Double sweep plot saved to: {save_file}")
    plt.close(fig)


# ------------------------------------------------------------------ #
# Main Orchestration Pipeline
# ------------------------------------------------------------------ #
def main():
    # 1. Define Reasonable Coefficient Ranges
    single_coeffs = np.round(np.arange(0.5, 5.5, 0.2), 2)

    # 2. Execute Single Sweep & Get Optimal Baseline Coefficient
    opt_baseline = execute_single_sweep(coeffs=single_coeffs)

    # 3. Define Double Sweep Range Centered around Optimal Baseline
    double_sweep_range = np.round(
        np.arange(max(0.2, opt_baseline - 1.5), opt_baseline + 1.6, 0.3), 2
    )

    # 4. Sweep 1: Vary c1, keep c2 fixed at opt_baseline
    print(f"\n=======================================================")
    print(f"  STEP 2: Sweep 1 (Varying c1, c2 fixed at {opt_baseline:.2f})")
    print(f"=======================================================")
    for c1 in double_sweep_range:
        print(f"Executing c1 = {c1:.2f}, c2 = {opt_baseline:.2f}")
        sim_dir = (
            VARY_C1_DIR
            / f"les_n33_p2_dt_0.001_c1_{c1:.2f}_c2_{opt_baseline:.2f}"
        )
        run_offset_les_simulation(
            external_coefficients=(c1, opt_baseline),
            master_path=sim_dir,
        )

    create_heatmap_and_error_plot(
        sweep_dir=VARY_C1_DIR,
        sweep_tag="vary_c1",
        varied_coeff_name="c1",
        fixed_coeff_val=opt_baseline,
    )

    # 5. Sweep 2: Vary c2, keep c1 fixed at opt_baseline
    print(f"\n=======================================================")
    print(f"  STEP 3: Sweep 2 (Varying c2, c1 fixed at {opt_baseline:.2f})")
    print(f"=======================================================")
    for c2 in double_sweep_range:
        print(f"Executing c1 = {opt_baseline:.2f}, c2 = {c2:.2f}")
        sim_dir = (
            VARY_C2_DIR
            / f"les_n33_p2_dt_0.001_c1_{opt_baseline:.2f}_c2_{c2:.2f}"
        )
        run_offset_les_simulation(
            external_coefficients=(opt_baseline, c2),
            master_path=sim_dir,
        )

    create_heatmap_and_error_plot(
        sweep_dir=VARY_C2_DIR,
        sweep_tag="vary_c2",
        varied_coeff_name="c2",
        fixed_coeff_val=opt_baseline,
    )

    print("\nFull parameter sensitivity sweep pipeline completed successfully!")


if __name__ == "__main__":
    main()