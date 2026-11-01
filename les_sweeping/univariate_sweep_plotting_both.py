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
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.8,
    }
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERVALS_DIR = PROJECT_ROOT / "dns_data" / "intervals"


def load_sweep_error_data(sweep_tag: str, varied_coeff_name: str, profile_type: str = "mean",
                          fixed_coeff_vals: tuple[float, float] = (4.9, 1.0),
                          z_domain: tuple[float, float] = (0.0, 2.0)):
    base_sweep_dir = PROJECT_ROOT / "solver_data" / "les_sweeps" / f"univariate_sweep_{sweep_tag}"
    if not base_sweep_dir.exists():
        print(f"[Warning] Directory {base_sweep_dir} does not exist.")
        return None, None

    time_dirs = sorted(base_sweep_dir.glob("t_*"))
    if not time_dirs:
        return None, None
    solver_data_dir = time_dirs[0]

    coeff_folders = sorted(solver_data_dir.glob("les_n17_p2_dt_0.001_c1_*_c2_*"))
    parsed_runs = []

    subfolder = "mean_profiles" if profile_type == "mean" else "reynolds_profiles"
    file_pattern = "mean_w_*.npy" if profile_type == "mean" else "reynolds_stress_profile_*.npy"

    for d in coeff_folders:
        match = re.search(r"c1_(\d+\.\d+)_c2_(\d+\.\d+)$", d.name)
        if match:
            c1, c2 = float(match.group(1)), float(match.group(2))
            var_val = c1 if varied_coeff_name == "c1" else c2
            fix_val = c2 if varied_coeff_name == "c1" else c1

            if any(np.isclose(fix_val, f_val) for f_val in fixed_coeff_vals):
                prof_dir = d / subfolder
                if prof_dir.exists():
                    for les_path in prof_dir.glob(file_pattern):
                        parsed_runs.append((var_val, fix_val, les_path, les_path.name))

    if not parsed_runs:
        return None, None

    frame_files = {}
    for var_val, fix_val, les_path, fname in parsed_runs:
        frame_files.setdefault(fname, {}).setdefault(fix_val, {})[var_val] = les_path

    return frame_files, base_sweep_dir


def create_dual_sweep_error_plot_for_type(
        profile_type: str = "mean",
        fixed_coeff_vals: tuple[float, float] = (4.9, 1.0),
        z_domain: tuple[float, float] = (0.0, 2.0),
        max_allowable_error: float = 45.0,
):
    c1_frames, _ = load_sweep_error_data("vary_c1", "c1", profile_type, fixed_coeff_vals, z_domain)
    c2_frames, _ = load_sweep_error_data("vary_c2", "c2", profile_type, fixed_coeff_vals, z_domain)

    if not c1_frames or not c2_frames:
        print(f"[Error] Missing sweep data for profile type: {profile_type}")
        return

    common_frames = sorted(list(set(c1_frames.keys()).intersection(set(c2_frames.keys()))))
    output_dir = PROJECT_ROOT / "solver_data" / "les_sweeps" / f"plots_dual_error_{profile_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    styles = [
        {"color": "#1b4965", "linestyle": "-"},
        {"color": "#e07a5f", "linestyle": "--"},
        {"color": "#81b29a", "linestyle": "-."},
    ]

    for filename in common_frames:
        if profile_type == "mean":
            match = re.search(r"mean_w_(\d+\.\d+)\.npy$", filename)
            dns_glob = "w_mean_*.npy"
            prop_label = r"$\langle w \rangle$"
        else:
            match = re.search(r"reynolds_stress_profile_(\d+\.\d+)\.npy$", filename)
            if not match:
                match = re.search(r"reynolds_stress_profile_(\d+)\.npy$", filename)
            dns_glob = "ww_*.npy"
            prop_label = r"$\langle w'w' \rangle$"

        if not match:
            continue
        t_end = float(match.group(1))

        matching_dns = [
            f for f in INTERVALS_DIR.glob(dns_glob)
            if f.name.endswith(f"_{t_end:.2f}.npy") or f.name.endswith(f"_{t_end}.npy")
        ]
        if not matching_dns:
            # Fallback search for reynolds if ww not found
            if profile_type == "reynolds":
                matching_dns = [
                    f for f in INTERVALS_DIR.glob("*reynolds*.npy")
                    if f.name.endswith(f"_{t_end:.2f}.npy") or f.name.endswith(f"_{t_end}.npy")
                ]
        if not matching_dns:
            continue

        dns_profile = np.load(matching_dns[0])
        mesh_dns = np.linspace(z_domain[0], z_domain[1], len(dns_profile))

        def compute_sweep_errors(frame_dict):
            sweep_data = {}
            all_errs = []
            for fix_val, var_dict in frame_dict.items():
                for var_val, les_path in var_dict.items():
                    les_data = np.load(les_path)
                    mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_data))
                    dns_interp = np.interp(mesh_les, mesh_dns, dns_profile)
                    rel_err = (np.linalg.norm(les_data - dns_interp) / (np.linalg.norm(dns_interp) + 1e-12)) * 100.0
                    sweep_data.setdefault(fix_val, {})[var_val] = rel_err
                    if rel_err <= max_allowable_error:
                        all_errs.append(rel_err)
            return sweep_data, all_errs

        data_c1, errs_c1 = compute_sweep_errors(c1_frames[filename])
        data_c2, errs_c2 = compute_sweep_errors(c2_frames[filename])

        # --- Create 1x2 Figure ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=300)

        for ax in (ax1, ax2):
            ax.axhline(0, color="black", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True, length=4, width=0.8)
            ax.tick_params(which="minor", length=2, width=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

        # --- Subplot (a): Varying c1 ---
        opt_c1_info = None
        min_err_c1 = float("inf")
        for idx, fix_val in enumerate(fixed_coeff_vals):
            if fix_val not in data_c1:
                continue
            vars_sorted = sorted(data_c1[fix_val].keys())
            vals = [data_c1[fix_val][v] for v in vars_sorted]
            st = styles[idx % len(styles)]
            ax1.plot(
                vars_sorted, vals, color=st["color"], marker="o", markersize=3.5,
                markerfacecolor="white", markeredgewidth=0.9, linestyle=st["linestyle"],
                label=f"$c_2 = {fix_val}$"
            )
            for v, e in zip(vars_sorted, vals):
                if e <= max_allowable_error and e < min_err_c1:
                    min_err_c1 = e
                    opt_c1_info = (v, fix_val, e)

        if opt_c1_info is not None:
            opt_v, _, opt_e = opt_c1_info
            ax1.plot(opt_v, opt_e, color="#b7094c", marker="d", markersize=7, linestyle="", label=f"Opt. {opt_e:.1f}%",
                     zorder=4)

        ax1.set_title(rf"(a) {prop_label} Error vs. $c_1$", fontsize=10, pad=6, loc="left")
        ax1.set_xlabel(r"Coefficient $c_1$", fontsize=9)
        ax1.set_ylabel(r"Relative $L_2$ Error (%)", fontsize=10)
        if errs_c1:
            ax1.set_ylim(max(0.0, min(errs_c1) - 2), max_allowable_error)

        # --- Subplot (b): Varying c2 ---
        opt_c2_info = None
        min_err_c2 = float("inf")
        for idx, fix_val in enumerate(fixed_coeff_vals):
            if fix_val not in data_c2:
                continue
            vars_sorted = sorted(data_c2[fix_val].keys())
            vals = [data_c2[fix_val][v] for v in vars_sorted]
            st = styles[idx % len(styles)]
            ax2.plot(
                vars_sorted, vals, color=st["color"], marker="o", markersize=3.5,
                markerfacecolor="white", markeredgewidth=0.9, linestyle=st["linestyle"],
                label=f"$c_1 = {fix_val}$"
            )
            for v, e in zip(vars_sorted, vals):
                if e <= max_allowable_error and e < min_err_c2:
                    min_err_c2 = e
                    opt_c2_info = (fix_val, v, e)

        if opt_c2_info is not None:
            _, opt_v, opt_e = opt_c2_info
            ax2.plot(opt_v, opt_e, color="#b7094c", marker="d", markersize=7, linestyle="", label=f"Opt. {opt_e:.1f}%",
                     zorder=4)

        ax2.set_title(rf"(b) {prop_label} Error vs. $c_2$", fontsize=10, pad=6, loc="left")
        ax2.set_xlabel(r"Coefficient $c_2$", fontsize=9)
        ax2.set_ylabel(r"Relative $L_2$ Error (%)", fontsize=10)
        if errs_c2:
            ax2.set_ylim(max(0.0, min(errs_c2) - 2), max_allowable_error)

        # External bottom legends
        for ax in (ax1, ax2):
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles=handles, labels=labels,
                loc="upper center", bbox_to_anchor=(0.5, -0.16),
                ncol=len(handles), frameon=False, fontsize=8.5
            )

        output_dir_save = PROJECT_ROOT / "solver_data" / "les_sweeps" / "plots_dual_error"
        output_dir_save.mkdir(parents=True, exist_ok=True)

        output_path_png = output_dir_save / f"dual_sweep_error_lines_{profile_type}_t_{t_end:.2f}.png"
        output_path_pdf = output_dir_save / f"dual_sweep_error_lines_{profile_type}_t_{t_end:.2f}.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
        plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {output_path_png.name} & .pdf")


if __name__ == "__main__":
    print("Generating Mean Velocity dual-sweep error lines...")
    create_dual_sweep_error_plot_for_type("mean", fixed_coeff_vals=[4.9, 1.0], max_allowable_error=45.0)

    print("\nGenerating Reynolds Stress dual-sweep error lines...")
    create_dual_sweep_error_plot_for_type("reynolds", fixed_coeff_vals=[4.9, 1.0], max_allowable_error=15.0)