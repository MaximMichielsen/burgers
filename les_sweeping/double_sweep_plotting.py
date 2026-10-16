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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERVALS_DIR = PROJECT_ROOT / "dns_data" / "intervals"


def create_profile_sweep_plot(
    sweep_tag: str,
    varied_coeff_name: str,
    fixed_coeff_val: float,
    z_domain: tuple[float, float] = (0.0, 2.0),
):
    base_sweep_dir = (
        PROJECT_ROOT / "solver_data" / "les_sweeps" / f"double_sweep_{sweep_tag}"
    )
    if not base_sweep_dir.exists():
        print(
            f"[Warning] Directory {base_sweep_dir} does not exist. Run simulations first."
        )
        return

    # Dynamically locate the time directory (e.g., t_308.0101_309.01)
    time_dirs = sorted(base_sweep_dir.glob("t_*"))
    if not time_dirs:
        print(f"[Warning] No time interval directories found under {base_sweep_dir}")
        return
    solver_data_dir = time_dirs[0]

    # Parse t_start from directory name safely
    dir_match = re.search(r"t_(\d+\.\d+)_(\d+\.\d+)", solver_data_dir.name)
    folder_t_start = float(dir_match.group(1)) if dir_match else 308.0101

    # Discover all available coefficient folder runs dynamically
    coeff_folders = sorted(solver_data_dir.glob("les_n17_p2_dt_0.001_c1_*_c2_*"))

    parsed_runs = []
    for d in coeff_folders:
        match = re.search(r"c1_(\d+\.\d+)_c2_(\d+\.\d+)$", d.name)
        if match:
            c1, c2 = float(match.group(1)), float(match.group(2))
            var_val = c1 if varied_coeff_name == "c1" else c2
            profiles_dir = d / "mean_profiles"
            if profiles_dir.exists():
                for les_path in profiles_dir.glob("mean_w_*.npy"):
                    parsed_runs.append((var_val, les_path))

    if not parsed_runs:
        print(f"[Warning] No profile data found for sweep tag: {sweep_tag}")
        return

    # Group by filename (timestep frame)
    frame_files = {}
    for var_val, les_path in parsed_runs:
        frame_files.setdefault(les_path.name, {})[var_val] = les_path

    varied_coeffs = np.array(sorted(list(set(r[0] for r in parsed_runs))))
    norm = plt.Normalize(vmin=varied_coeffs.min(), vmax=varied_coeffs.max())
    cmap = cm.plasma

    output_dir = base_sweep_dir / "coefficient_comparison_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in sorted(frame_files.keys()):
        match = re.search(r"mean_w_(\d+\.\d+)\.npy$", filename)
        if not match:
            continue

        t_start = folder_t_start
        t_end = float(match.group(1))

        # Locate matching interval DNS reference file
        matching_dns = [
            f
            for f in INTERVALS_DIR.glob("w_mean_*.npy")
            if f.name.endswith(f"_{t_end:.2f}.npy") or f.name.endswith(f"_{t_end}.npy")
        ]
        if not matching_dns:
            continue

        mean_profile_dns = np.load(matching_dns[0])
        mesh_dns = np.linspace(z_domain[0], z_domain[1], len(mean_profile_dns))

        # Create figure matching publication layout (dual panels)
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

        for var_val in varied_coeffs:
            if var_val not in frame_files[filename]:
                continue
            les_mean = np.load(frame_files[filename][var_val])
            mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_mean))

            dns_interp = np.interp(mesh_les, mesh_dns, mean_profile_dns)
            rel_err = (
                np.linalg.norm(les_mean - dns_interp)
                / (np.linalg.norm(dns_interp) + 1e-12)
            ) * 100.0
            l2_errors.append((var_val, rel_err))

            ax1.plot(
                mesh_les,
                les_mean,
                color=cmap(norm(var_val)),
                alpha=0.65,
                linewidth=1.1,
                zorder=3,
            )

        other_coeff_name = "c_2" if varied_coeff_name == "c1" else "c_1"
        ax1.set_title(
            r"(a) Mean Velocity Profile Sweep $\langle w \rangle$ "
            + f"(${other_coeff_name}={fixed_coeff_val}$)",
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
        cbar.set_label(f"Coefficient ${varied_coeff_name}$", fontsize=10)

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
                label=f"Optimal ({coeffs_arr[min_idx]:.2f}): {errs_arr[min_idx]:.1f}%",
                zorder=4,
            )

            # Highlight Baseline (Coefficient = 1.0) if present in sweep range
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
                    linestyle="",
                    zorder=4,
                    label=f"Baseline (1.0): {base_err:.1f}%",
                )

            ax2.set_xlim(varied_coeffs.min() - 0.2, varied_coeffs.max() + 0.2)

            # Tighter Y-Axis Scaling
            err_min, err_max = min(errs_arr), max(errs_arr)
            padding = (err_max - err_min) * 0.15 if err_max != err_min else 2.0
            ax2.set_ylim(max(0, err_min - padding), err_max + padding)

        ax2.set_title(
            r"(b) Relative Error vs. Coefficient", fontsize=11, pad=8, loc="left"
        )
        ax2.set_xlabel(f"Coefficient ${varied_coeff_name}$", fontsize=11)
        ax2.set_ylabel(r"Relative $L_2$ Error (%)", fontsize=11)

        ax2.legend(
            loc="upper right",
            frameon=True,
            edgecolor="none",
            facecolor="white",
            framealpha=0.9,
        )

        # Save figure cleanly
        output_plot_path = output_dir / f"profile_sweep_{t_start}_{t_end:.2f}.png"
        plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {output_plot_path.name}")

    print(f"\n[SUCCESS] All double sweep comparison plots saved to: {output_dir}")


if __name__ == "__main__":
    # Plot Sweep 1 (Varying c1, c2 fixed at 4.9)
    create_profile_sweep_plot(
        sweep_tag="vary_c1", varied_coeff_name="c1", fixed_coeff_val=4.9
    )

    # Plot Sweep 2 (Varying c2, c1 fixed at 4.9)
    create_profile_sweep_plot(
        sweep_tag="vary_c2", varied_coeff_name="c2", fixed_coeff_val=4.9
    )
