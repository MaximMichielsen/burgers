import re
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# --- Publication-Quality Global Settings ---
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
    }
)

# Directory Setup updated to target all interval sweeps
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERVALS_DIR = PROJECT_ROOT / "dns_data" / "intervals"
base_sweep_dir = PROJECT_ROOT / "solver_data" / "les_sweeps" / "all_intervals_sweep"

# Dynamically locate all time directories (e.g., t_308.0101_309.01, etc.)
time_dirs = sorted(base_sweep_dir.glob("t_*"))
if not time_dirs:
    raise FileNotFoundError(
        f"No time interval directories found under {base_sweep_dir}. Make sure to run the interval sweep first."
    )

n_nodes_les = 17
tau_params = 2
dt = 0.001
z_domain = (0.0, 2.0)

output_dir = base_sweep_dir / "plots"
output_dir.mkdir(parents=True, exist_ok=True)

# Loop through every time interval directory
for solver_data_dir in time_dirs:
    # Parse t_start and t_end from directory name safely (supporting custom precision strings like 308.0101)
    dir_match = re.search(r"t_(.+?)_(.+?)$", solver_data_dir.name)
    if dir_match:
        folder_t_start_str = dir_match.group(1)
        folder_t_end_str = dir_match.group(2)
        folder_t_start = float(folder_t_start_str)
        folder_t_end = float(folder_t_end_str)
    else:
        folder_t_start_str = "308.0101"
        folder_t_end_str = "309.01"
        folder_t_start = 308.0101
        folder_t_end = 309.01

    # Discover all available coefficient runs dynamically inside the time directory
    coeff_dirs = sorted(solver_data_dir.glob(f"les_n{n_nodes_les}_p2_dt_0.001_coeff_*"))
    coeffs = []
    for d in coeff_dirs:
        match = re.search(r"coeff_(\d+\.\d+)$", d.name)
        if match:
            coeffs.append(float(match.group(1)))

    coeffs = np.array(sorted(coeffs))

    if len(coeffs) == 0:
        print(
            f"[Warning] No coefficient directories found under {solver_data_dir.name}. Skipping."
        )
        continue

    # Map frames dynamically from output directories
    frame_files = {}
    for coeff in coeffs:
        profiles_dir = (
            solver_data_dir
            / f"les_n{n_nodes_les}_p{tau_params}_dt_{dt}_coeff_{coeff:.1f}"
            / "mean_profiles"
        )
        if profiles_dir.exists():
            for les_path in profiles_dir.glob("mean_w_*.npy"):
                frame_files.setdefault(les_path.name, {})[coeff] = les_path

    # Setup colormap normalized to the coefficient range
    norm = plt.Normalize(vmin=coeffs.min(), vmax=coeffs.max())
    cmap = cm.plasma

    for filename in sorted(frame_files.keys()):
        match = re.search(r"mean_w_(\d+\.\d+)\.npy$", filename)
        if not match:
            continue

        t_start = folder_t_start
        t_end = float(match.group(1))

        # Directly target the exact interval DNS reference file using string tokens or float fallbacks
        dns_mean_path = INTERVALS_DIR / f"w_mean_{folder_t_start_str}_{t_end:.2f}.npy"
        if not dns_mean_path.exists():
            dns_mean_path = INTERVALS_DIR / f"w_mean_{t_start}_{t_end}.npy"
        if not dns_mean_path.exists():
            dns_mean_path = INTERVALS_DIR / f"w_mean_{t_start:.2f}_{t_end:.2f}.npy"

        if not dns_mean_path.exists():
            print(
                f"[Warning] Missing interval DNS reference for [{folder_t_start_str}, {t_end}] at {dns_mean_path.name}"
            )
            continue

        mean_profile_dns = np.load(dns_mean_path)
        mesh_dns = np.linspace(z_domain[0], z_domain[1], len(mean_profile_dns))

        # Create figure matching publication layout (no suptitle, clean dual panels)
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(13, 5),
            dpi=300,
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.3, 1]},
        )

        for ax in (ax1, ax2):
            ax.axhline(
                0, color="black", linestyle="--", linewidth=0.5, alpha=0.4, zorder=1
            )
            ax.minorticks_on()
            ax.tick_params(
                which="both", direction="in", top=True, right=True, length=4, width=0.8
            )
            ax.tick_params(which="minor", length=2, width=0.5)

        ax1.set_xlim(z_domain)

        # --- LEFT SUBPLOT: Profile Sweep with Colorbar ---
        ax1.plot(
            mesh_dns,
            mean_profile_dns,
            color="black",
            linestyle="-",
            linewidth=1.8,
            label=r"DNS Reference ($\langle w \rangle$)",
            zorder=5,
        )

        l2_errors = []

        for coeff in coeffs:
            if coeff not in frame_files[filename]:
                continue
            les_mean = np.load(frame_files[filename][coeff])
            mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_mean))

            dns_interp = np.interp(mesh_les, mesh_dns, mean_profile_dns)
            rel_err = (
                np.linalg.norm(les_mean - dns_interp)
                / (np.linalg.norm(dns_interp) + 1e-12)
            ) * 100.0
            l2_errors.append((coeff, rel_err))

            ax1.plot(
                mesh_les,
                les_mean,
                color=cmap(norm(coeff)),
                alpha=0.65,
                linewidth=1.1,
                zorder=3,
            )

        ax1.set_title(
            r"(a) Mean Velocity Profile Sweep $\langle w \rangle$",
            fontsize=11,
            pad=8,
            loc="left",
        )
        ax1.set_xlabel(r"Wall-normal coordinate $z$", fontsize=11)
        ax1.set_ylabel(r"$\langle w \rangle$", fontsize=11)

        y_min, y_max = ax1.get_ylim()
        ax1.set_ylim(y_min, y_max * 1.15)
        ax1.legend(
            loc="upper right",
            frameon=True,
            edgecolor="none",
            facecolor="white",
            framealpha=0.9,
        )

        # Attach Colorbar to Left Plot cleanly
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, orientation="vertical", pad=0.03, shrink=0.85)
        cbar.set_label("External Coefficient", fontsize=10)

        # --- RIGHT SUBPLOT: Error vs. Correction Coefficient ---
        if l2_errors:
            coeffs_arr, errs_arr = zip(*l2_errors)
            ax2.plot(
                coeffs_arr,
                errs_arr,
                color="#1b4965",
                marker="o",
                markersize=3,
                linewidth=1.4,
                linestyle="-",
            )

            # Highlight Optimal Point
            min_idx = np.argmin(errs_arr)
            ax2.plot(
                coeffs_arr[min_idx],
                errs_arr[min_idx],
                color="#b7094c",
                marker="d",
                markersize=7,
                linestyle="",
                label=f"Optimal ({coeffs_arr[min_idx]:.1f}): {errs_arr[min_idx]:.1f}%",
                zorder=4,
            )

            # Highlight Baseline (Coefficient = 1.0)
            if 1.0 in coeffs_arr:
                base_idx = coeffs_arr.index(1.0)
                base_err = errs_arr[base_idx]

                ax2.axhline(
                    base_err,
                    color="gray",
                    linestyle=":",
                    linewidth=1.1,
                    alpha=0.8,
                    zorder=2,
                )

                ax2.plot(
                    1.0,
                    base_err,
                    color="#2b9348",
                    marker="s",
                    markersize=6,
                    zorder=4,
                    linestyle="",
                    label=f"Baseline (1.0): {base_err:.1f}%",
                )

            ax2.set_xlim(coeffs.min() - 0.2, coeffs.max() + 0.2)

            # --- Tighter Y-Axis Scaling ---
            err_min, err_max = min(errs_arr), max(errs_arr)
            padding = (err_max - err_min) * 0.15 if err_max != err_min else 2.0
            ax2.set_ylim(max(0, err_min - padding), err_max + padding)

        ax2.set_title(
            r"(b) Relative Error vs. Coefficient", fontsize=11, pad=8, loc="left"
        )
        ax2.set_xlabel("External Coefficient", fontsize=11)
        ax2.set_ylabel(r"Relative $L_2$ Error (%)", fontsize=11)

        ax2.legend(
            loc="upper right",
            frameon=True,
            edgecolor="none",
            facecolor="white",
            framealpha=0.9,
        )

        # Save figure cleanly with exact string token naming
        output_plot_path = (
            output_dir
            / f"profile_sweep_n{n_nodes_les}_t_{folder_t_start_str}_{t_end:.2f}.png"
        )
        plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {output_plot_path.name}")

print(
    f"\n[SUCCESS] All sweep comparison plots for all intervals saved to: {output_dir}"
)
