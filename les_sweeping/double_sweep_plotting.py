import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
PARSED_STAT_DIR = PROJECT_ROOT / "dns_data" / "curated" / "stat"


def create_heatmap_and_error_plot(
    sweep_tag: str,
    varied_coeff_name: str,
    fixed_coeff_val: float,
    z_domain: tuple[float, float] = (0.0, 2.0),
):
    solver_data_dir = PROJECT_ROOT / "solver_data" / f"les_offset_sweeps_{sweep_tag}"
    if not solver_data_dir.exists():
        print(f"Directory {solver_data_dir} does not exist. Run simulations first.")
        return

    # Load DNS Reference profile (Target index 0)
    mean_w_profiles = np.load(PARSED_STAT_DIR / "mean_w.npy")
    dns_profile = mean_w_profiles[0]
    mesh_dns = np.linspace(z_domain[0], z_domain[1], len(dns_profile))

    # Discover and parse output folders
    coeff_folders = sorted(solver_data_dir.glob("les_n17_p2_dt_0.001_c1_*_c2_*"))

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

    # Build 2D velocity matrix: shape (n_coeffs, n_les_nodes)
    velocity_matrix = []
    l2_errors = []

    for var_val, file_path in parsed_runs:
        les_mean = np.load(file_path)
        velocity_matrix.append(les_mean)

        mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_mean))
        dns_interp = np.interp(mesh_les, mesh_dns, dns_profile)

        rel_err = (
            np.linalg.norm(les_mean - dns_interp) / (np.linalg.norm(dns_interp) + 1e-12)
        ) * 100.0
        l2_errors.append(rel_err)

    velocity_matrix = np.array(velocity_matrix)
    mesh_les = np.linspace(z_domain[0], z_domain[1], velocity_matrix.shape[1])

    # ------------------------------------------------------------------ #
    # Plotting Layout
    # ------------------------------------------------------------------ #
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), dpi=200, gridspec_kw={"width_ratios": [1.4, 1]}
    )

    # Panel 1: 2D Spatial-Coefficient Contour/Heatmap
    Z, C = np.meshgrid(mesh_les, varied_coeffs)
    contour = ax1.contourf(Z, C, velocity_matrix, levels=40, cmap="viridis")
    cbar = fig.colorbar(contour, ax=ax1, pad=0.02)
    cbar.set_label(r"Mean Velocity $\langle w \rangle$", fontsize=11)

    # Overlay DNS contours/reference marker line
    ax1.set_xlabel(r"Domain Coordinate $z$", fontsize=11)
    ax1.set_ylabel(f"External Coefficient ${varied_coeff_name}$", fontsize=11)
    ax1.set_title(
        f"Spatial Velocity Heatmap $\\langle w \\rangle(z, {varied_coeff_name})$",
        fontsize=12,
    )

    # Panel 2: L2 Error Curve vs Varied Coefficient
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
        f"Relative $L_2$ Error vs. ${varied_coeff_name}$ (with ${other_coeff_name}={fixed_coeff_val}$)",
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

    out_dir = PROJECT_ROOT / "solver_data" / "coefficient_comparison_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file = out_dir / f"heatmap_sweep_{sweep_tag}.png"
    plt.savefig(save_file, dpi=300)
    print(f"Plot saved successfully to: {save_file}")
    plt.close(fig)


if __name__ == "__main__":
    # Plot Sweep 1 (Varying c1, c2 fixed at 4.9)
    create_heatmap_and_error_plot(
        sweep_tag="vary_c1", varied_coeff_name="c1", fixed_coeff_val=4.9
    )

    # Plot Sweep 2 (Varying c2, c1 fixed at 4.9)
    create_heatmap_and_error_plot(
        sweep_tag="vary_c2", varied_coeff_name="c2", fixed_coeff_val=4.9
    )
