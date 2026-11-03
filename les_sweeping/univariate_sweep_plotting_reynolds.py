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
        "mathtext.fontset": "stix",  # Times-compatible math font set
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERVALS_DIR = PROJECT_ROOT / "dns_data" / "intervals"


def create_univariate_reynolds_sweep_plot(
    sweep_tag: str,
    varied_coeff_name: str,
    fixed_coeff_vals: tuple[float, float] = (4.9, 1.0),
    z_domain: tuple[float, float] = (0.0, 2.0),
    max_allowable_error: float = 50.0,  # Upper Y-limit cap for right plot
):
    base_sweep_dir = (
        PROJECT_ROOT / "solver_data" / "les_sweeps" / f"univariate_sweep_{sweep_tag}"
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
            fix_val = c2 if varied_coeff_name == "c1" else c1

            # Only include if fix_val matches one of our target fixed values
            if any(np.isclose(fix_val, f_val) for f_val in fixed_coeff_vals):
                reynolds_profiles_dir = d / "reynolds_profiles"
                if reynolds_profiles_dir.exists():
                    # Look for Reynolds stress profile .npy files
                    for les_path in reynolds_profiles_dir.glob(
                        "reynolds_stress_profile_*.npy"
                    ):
                        parsed_runs.append((var_val, fix_val, les_path))

    if not parsed_runs:
        print(
            f"[Warning] No Reynolds stress profile data found for sweep tag: {sweep_tag}"
        )
        return

    # Group by filename (timestep frame) and fixed coefficient value
    frame_files = {}
    for var_val, fix_val, les_path in parsed_runs:
        frame_files.setdefault(les_path.name, {}).setdefault(fix_val, {})[var_val] = (
            les_path
        )

    varied_coeffs = np.array(sorted(list(set(r[0] for r in parsed_runs))))
    norm = plt.Normalize(vmin=varied_coeffs.min(), vmax=varied_coeffs.max())
    cmap = cm.plasma

    output_dir = base_sweep_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    other_coeff_label = r"$c_2$" if varied_coeff_name == "c1" else r"$c_1$"
    var_coeff_label = r"$c_1$" if varied_coeff_name == "c1" else r"$c_2$"

    for filename in sorted(frame_files.keys()):
        match = re.search(r"reynolds_stress_profile_(\d+\.\d+)\.npy$", filename)
        if not match:
            match_alt = re.search(r"reynolds_stress_profile_(\d+)\.npy$", filename)
            if not match_alt:
                continue
            t_end = float(match_alt.group(1))
        else:
            t_end = float(match.group(1))

        t_start = folder_t_start

        # Locate matching interval DNS reference file for Reynolds stress (e.g., reynolds_mean_*.npy or similar naming convention)
        matching_dns = [
            f
            for f in INTERVALS_DIR.glob("ww_*.npy")
            if f.name.endswith(f"_{t_end:.2f}.npy") or f.name.endswith(f"_{t_end}.npy")
        ]
        # Fallback search pattern if prefix differs slightly
        if not matching_dns:
            matching_dns = [
                f
                for f in INTERVALS_DIR.glob("*reynolds*.npy")
                if f.name.endswith(f"_{t_end:.2f}.npy")
                or f.name.endswith(f"_{t_end}.npy")
            ]

        if not matching_dns:
            print(
                f"[Warning] Matching DNS Reynolds profile not found for t_end = {t_end}"
            )
            continue

        reynolds_profile_dns = np.load(matching_dns[0])
        mesh_dns = np.linspace(z_domain[0], z_domain[1], len(reynolds_profile_dns))

        # Calculate errors for all runs
        all_frame_data = {}
        stable_computed_errors = []

        for fix_val, var_dict in frame_files[filename].items():
            for var_val, les_path in var_dict.items():
                les_reynolds = np.load(les_path)
                mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_reynolds))
                dns_interp = np.interp(mesh_les, mesh_dns, reynolds_profile_dns)

                rel_err = (
                    np.linalg.norm(les_reynolds - dns_interp)
                    / (np.linalg.norm(dns_interp) + 1e-12)
                ) * 100.0

                # Store for right plot (all runs)
                all_frame_data.setdefault(fix_val, {})[var_val] = (les_path, rel_err)

                # Track stable errors for scaling/left plot filtering
                if rel_err <= max_allowable_error:
                    stable_computed_errors.append(rel_err)

        if not all_frame_data:
            continue

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

        # --- LEFT SUBPLOT: Reynolds Stress Profile Sweep (Excluding unstable profiles > max_allowable_error) ---
        (dns_line,) = ax1.plot(
            mesh_dns,
            reynolds_profile_dns,
            color="black",
            linestyle="-",
            linewidth=1.8,
            label=r"DNS Ref.",
            zorder=5,
        )

        profile_linestyles = ["-", "--", "-."]
        for idx, fix_val in enumerate(fixed_coeff_vals):
            if fix_val not in all_frame_data:
                continue
            ls = profile_linestyles[idx % len(profile_linestyles)]

            for var_val, (les_path, rel_err) in all_frame_data[fix_val].items():
                if rel_err <= max_allowable_error:
                    les_reynolds = np.load(les_path)
                    mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_reynolds))

                    ax1.plot(
                        mesh_les,
                        les_reynolds,
                        color=cmap(norm(var_val)),
                        linestyle=ls,
                        alpha=0.60,
                        linewidth=1.1,
                        zorder=3,
                    )

        # Time interval integration label matching reference design
        time_str = f"$t \\in [{t_start:.2f}, {t_end:.2f}]$"
        ax1.set_title(
            rf"(a) Reynolds Stress Profile $\langle w'w' \rangle$ | {time_str}",
            fontsize=10,
            pad=6,
            loc="left",
        )
        ax1.set_xlabel(r"Wall-normal coordinate $z$", fontsize=9)
        ax1.set_ylabel(r"$\langle w'w' \rangle$", fontsize=10)

        y_min, y_max = ax1.get_ylim()
        ax1.set_ylim(y_min, y_max * 1.15)

        # Left Subplot Legend (placed outside below ax1)
        legend_handles_ax1 = [dns_line]
        for idx, fix_val in enumerate(fixed_coeff_vals):
            if fix_val not in all_frame_data:
                continue
            ls = profile_linestyles[idx % len(profile_linestyles)]
            proxy_handle = mlines.Line2D(
                [],
                [],
                color="gray",
                linestyle=ls,
                linewidth=1.2,
                label=f"{other_coeff_label} = {fix_val}",
            )
            legend_handles_ax1.append(proxy_handle)

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
        cbar.set_label(f"Coefficient {var_coeff_label}", fontsize=9)

        # --- RIGHT SUBPLOT: Error vs. Coefficient (Includes all runs, clipped by Y-limits) ---
        styles = [
            {"color": "#1b4965", "linestyle": "-"},
            {"color": "#e07a5f", "linestyle": "--"},
            {"color": "#81b29a", "linestyle": "-."},
        ]

        global_min_err = float("inf")
        primary_opt_info = None

        for idx, fix_val in enumerate(fixed_coeff_vals):
            if fix_val not in all_frame_data:
                continue

            sorted_vars = sorted(all_frame_data[fix_val].keys())
            coeffs_arr = []
            errs_arr = []

            for var_val in sorted_vars:
                _, rel_err = all_frame_data[fix_val][var_val]
                coeffs_arr.append(var_val)
                errs_arr.append(rel_err)

            if errs_arr:
                style = styles[idx % len(styles)]
                ax2.plot(
                    coeffs_arr,
                    errs_arr,
                    color=style["color"],
                    marker="o",
                    markersize=3.5,
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    linestyle=style["linestyle"],
                    label=f"{other_coeff_label} = {fix_val}",
                )

                # Track global minimum for optimal annotation (only from stable runs <= max_allowable_error)
                for v_val, e_val in zip(coeffs_arr, errs_arr):
                    if e_val <= max_allowable_error and e_val < global_min_err:
                        global_min_err = e_val
                        if varied_coeff_name == "c1":
                            primary_opt_info = (v_val, fix_val, e_val)
                        else:
                            primary_opt_info = (fix_val, v_val, e_val)

        # Plot global optimal marker
        opt_err_val = 0.0
        if primary_opt_info is not None:
            opt_c1, opt_c2, opt_err = primary_opt_info
            opt_err_val = opt_err
            opt_x = opt_c1 if varied_coeff_name == "c1" else opt_c2
            ax2.plot(
                opt_x,
                opt_err,
                color="#b7094c",
                marker="d",
                markersize=7,
                linestyle="",
                label=f"Opt. {opt_err:.1f}%",
                zorder=4,
            )

        if varied_coeffs.size > 0:
            ax2.set_xlim(varied_coeffs.min() - 0.2, varied_coeffs.max() + 0.2)

        # Set Y-limits cleanly with lower bound slightly below the minimum stable error, upper bound at max_allowable_error
        if stable_computed_errors:
            min_err_val = min(stable_computed_errors)
            err_span = max(stable_computed_errors) - min_err_val
            padding = err_span * 0.10 if err_span > 0 else 2.0
            ax2.set_ylim(max(0.0, min_err_val - padding), max_allowable_error)

        ax2.set_title(
            r"(b) Relative Error vs. Coefficient", fontsize=10, pad=6, loc="left"
        )
        ax2.set_xlabel(f"Coefficient {var_coeff_label}", fontsize=9)
        ax2.set_ylabel(r"Relative $L_2$ Error (%)", fontsize=10)

        # Right Subplot Legend (placed outside below ax2)
        legend_handles_ax2 = []
        for idx, fix_val in enumerate(fixed_coeff_vals):
            if fix_val not in all_frame_data:
                continue
            st = styles[idx % len(styles)]
            proxy_handle = mlines.Line2D(
                [],
                [],
                color=st["color"],
                linestyle=st["linestyle"],
                marker="o",
                markersize=3,
                linewidth=1.4,
                markerfacecolor="white",
                markeredgewidth=0.9,
                label=f"{other_coeff_label} = {fix_val}",
            )
            legend_handles_ax2.append(proxy_handle)

        if primary_opt_info is not None:
            opt_proxy = mlines.Line2D(
                [],
                [],
                color="#b7094c",
                marker="d",
                markersize=7,
                linestyle="",
                label=f"Opt. {opt_err_val:.2f}%",
            )
            legend_handles_ax2.append(opt_proxy)

        ax2.legend(
            handles=legend_handles_ax2,
            labels=[h.get_label() for h in legend_handles_ax2],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=len(legend_handles_ax2),
            frameon=False,
            fontsize=8.5,
        )

        # Save figure cleanly with descriptive naming
        output_plot_path = (
            output_dir
            / f"reynolds_univariate_var_{varied_coeff_name}_int_{t_start}_{t_end:.2f}.png"
        )
        plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
        output_plot_path = (
            output_dir
            / f"reynolds_univariate_var_{varied_coeff_name}_int_{t_start}_{t_end:.2f}.pdf"
        )
        plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {output_plot_path.name}")

    print(
        f"\n[SUCCESS] All Reynolds stress double sweep comparison plots saved to: {output_dir}"
    )


if __name__ == "__main__":
    # Plot Sweep 1 Reynolds Stress (Varying c1 with c2 fixed at 4.9 and 1.0)
    create_univariate_reynolds_sweep_plot(
        sweep_tag="vary_c1",
        varied_coeff_name="c1",
        fixed_coeff_vals=[4.9, 1.0],
        max_allowable_error=45.0,
    )

    # Plot Sweep 2 Reynolds Stress (Varying c2 with c1 fixed at 4.9 and 1.0)
    create_univariate_reynolds_sweep_plot(
        sweep_tag="vary_c2",
        varied_coeff_name="c2",
        fixed_coeff_vals=[4.9, 1.0],
        max_allowable_error=45.0,
    )
