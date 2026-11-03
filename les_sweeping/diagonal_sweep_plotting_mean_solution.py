import re
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# --- Publication-Quality Global Settings (Direct Times New Roman + STIX Math) ---
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",  # Times-compatible math font set
        "font.size": 10,  # Base font size
        "axes.labelsize": 10,  # Axis label size
        "axes.titlesize": 10,  # Title size
        "xtick.labelsize": 9,  # Tick label size
        "ytick.labelsize": 9,  # Tick label size
        "legend.fontsize": 8.5,  # Legend font size
        "xtick.direction": "in",  # Ticks pointing inward
        "ytick.direction": "in",
        "xtick.top": True,  # Ticks on all four sides for a clean box frame
        "ytick.right": True,
        "axes.linewidth": 0.8,  # Frame line width
    }
)

# Directory Setup matching single_sweep_2.py structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERVALS_DIR = PROJECT_ROOT / "dns_data" / "intervals"
base_sweep_dir = PROJECT_ROOT / "solver_data" / "les_sweeps" / "diagonal_sweep"

# Dynamically locate all time directories (e.g., t_308.0101_309.01, etc.)
time_dirs = sorted(base_sweep_dir.glob("t_*"))
if not time_dirs:
    raise FileNotFoundError(
        f"No time interval directories found under {base_sweep_dir}"
    )

n_nodes_les = 17
tau_params = 2
dt = 0.001
z_domain = (0.0, 2.0)

output_dir = base_sweep_dir / "plots"
output_dir.mkdir(parents=True, exist_ok=True)

# Loop through every time interval directory
for solver_data_dir in time_dirs:
    # Parse t_start and t_end from directory name safely
    dir_match = re.search(r"t_(\d+\.\d+)_(\d+\.\d+)", solver_data_dir.name)
    if dir_match:
        folder_t_start = float(dir_match.group(1))
        folder_t_end = float(dir_match.group(2))
    else:
        folder_t_start = 308.0101  # fallback default
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

    output_dir = base_sweep_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in sorted(frame_files.keys()):
        match = re.search(r"mean_w_(\d+\.\d+)\.npy$", filename)
        if not match:
            continue

        t_start = folder_t_start
        t_end = float(match.group(1))

        # Directly target the exact interval DNS reference file using t_start and t_end
        dns_mean_path = INTERVALS_DIR / f"w_mean_{t_start}_{t_end}.npy"
        if not dns_mean_path.exists():
            dns_mean_path = INTERVALS_DIR / f"w_mean_{t_start:.2f}_{t_end:.2f}.npy"

        if not dns_mean_path.exists():
            print(f"[Warning] Missing interval DNS reference for [{t_start}, {t_end}]")
            continue

        mean_profile_dns = np.load(dns_mean_path)
        mesh_dns = np.linspace(z_domain[0], z_domain[1], len(mean_profile_dns))

        # Create figure matching publication layout (dual panels) with room at bottom for legends
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(10, 3.5),
            dpi=300,
            gridspec_kw={"width_ratios": [1.3, 1]},
        )

        for ax in (ax1, ax2):
            ax.axhline(
                0, color="black", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0
            )
            ax.minorticks_on()
            ax.tick_params(
                which="both", direction="in", top=True, right=True, length=4, width=0.8
            )
            ax.tick_params(which="minor", length=2, width=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

        ax1.set_xlim(z_domain)

        # --- LEFT SUBPLOT: Profile Sweep with Colorbar ---
        (dns_line,) = ax1.plot(
            mesh_dns,
            mean_profile_dns,
            color="black",
            linestyle="-",
            linewidth=1.8,
            label=r"DNS Ref.",
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
                alpha=0.60,
                linewidth=1.1,
                zorder=3,
            )

        time_str = f"$t \\in [{t_start:.2f}, {t_end:.2f}]$"
        ax1.set_title(
            rf"(a) Mean Velocity Profile $\langle w \rangle$ | {time_str}",
            fontsize=10,
            pad=6,
            loc="left",
        )
        ax1.set_xlabel(r"Wall-normal coordinate $z$", fontsize=9)
        ax1.set_ylabel(r"$\langle w \rangle$", fontsize=10)

        y_min, y_max = ax1.get_ylim()
        ax1.set_ylim(y_min, y_max * 1.15)

        # Left Subplot Legend (placed outside below ax1)
        legend_handles_ax1 = [dns_line]
        ax1.legend(
            handles=legend_handles_ax1,
            labels=[h.get_label() for h in legend_handles_ax1],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=len(legend_handles_ax1),
            frameon=False,
            fontsize=8.5,
        )

        # Attach Colorbar to Left Plot cleanly with math-style notation
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, orientation="vertical", pad=0.03, shrink=0.85)
        cbar.set_label("Coefficient", fontsize=9)

        # --- RIGHT SUBPLOT: Error vs. Correction Coefficient ---
        opt_info = None
        base_info = None

        if l2_errors:
            coeffs_arr, errs_arr = zip(*l2_errors)
            ax2.plot(
                coeffs_arr,
                errs_arr,
                color="#1b4965",
                marker="o",
                markersize=3.5,
                markerfacecolor="white",
                markeredgewidth=0.9,
                linestyle="-",
                label="Error",
            )

            # Highlight Optimal Point
            min_idx = np.argmin(errs_arr)
            opt_coeff = coeffs_arr[min_idx]
            opt_err = errs_arr[min_idx]
            opt_info = (opt_coeff, opt_err)

            ax2.plot(
                opt_coeff,
                opt_err,
                color="#b7094c",
                marker="d",
                markersize=6,
                linestyle="",
                label=f"Opt. ({opt_coeff:.1f}): {opt_err:.1f}%",
                zorder=4,
            )

            # Highlight Baseline (Coefficient = 1.0)
            if 1.0 in coeffs_arr:
                base_idx = coeffs_arr.index(1.0)
                base_err = errs_arr[base_idx]
                base_info = base_err

                # Draw subtle gray horizontal reference line across the right plot at baseline error level
                ax2.axhline(
                    base_err,
                    color="gray",
                    linestyle=":",
                    linewidth=1.1,
                    alpha=0.8,
                    zorder=2,
                )

                # Highlight the point at coeff = 1.0 on the curve with a green square marker
                ax2.plot(
                    1.0,
                    base_err,
                    color="#e76f51",
                    marker="d",
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
            r"(b) Relative Error vs. Coefficient", fontsize=10, pad=6, loc="left"
        )
        ax2.set_xlabel("External Coefficient", fontsize=9)
        ax2.set_ylabel(r"Relative $L_2$ Error (%)", fontsize=10)

        # Right Subplot Legend (placed outside below ax2)
        legend_handles_ax2 = []
        curve_proxy = mlines.Line2D(
            [],
            [],
            color="#1b4965",
            linestyle="-",
            marker="o",
            markersize=3,
            linewidth=1.4,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label="Error curve",
        )
        legend_handles_ax2.append(curve_proxy)

        if opt_info is not None:
            opt_proxy = mlines.Line2D(
                [],
                [],
                color="#b7094c",
                marker="d",
                markersize=6,
                linestyle="",
                label=f"Opt. ({opt_info[0]:.1f}): {opt_info[1]:.1f}%",
            )
            legend_handles_ax2.append(opt_proxy)

        if base_info is not None:
            base_proxy = mlines.Line2D(
                [],
                [],
                color="#e76f51",
                marker="d",
                markersize=6,
                linestyle="",
                label=f"Baseline (1.0): {base_info:.1f}%",
            )
            legend_handles_ax2.append(base_proxy)

        ax2.legend(
            handles=legend_handles_ax2,
            labels=[h.get_label() for h in legend_handles_ax2],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=len(legend_handles_ax2),
            frameon=False,
            fontsize=8.5,
        )

        # Save figure cleanly
        output_plot_path = (
            output_dir / f"profile_sweep_n{n_nodes_les}_t_{t_start:.2f}_{t_end:.2f}.png"
        )
        plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
        output_plot_path = (
            output_dir / f"profile_sweep_n{n_nodes_les}_t_{t_start:.2f}_{t_end:.2f}.pdf"
        )
        plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {output_plot_path.name}")

print(f"\n[SUCCESS] All sweep comparison plots saved to: {output_dir}")
