import re
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# Directory Setup
PROJECT_ROOT = Path(__file__).resolve().parent
PARSED_STAT_DIR = PROJECT_ROOT / "dns_data" / "curated" / "stat"
solver_data_dir = PROJECT_ROOT / "solver_data" / "les_offset_tests"

n_nodes_les = 17
tau_params = 2
dt = 0.001
t_start = 308.01
z_domain = (0.0, 2.0)

mean_w_profiles = np.load(PARSED_STAT_DIR / "mean_w.npy")

# Discover all available coefficient runs dynamically
coeff_dirs = sorted(solver_data_dir.glob("les_n17_p2_dt_0.001_coeff_*"))
coeffs = []
for d in coeff_dirs:
    match = re.search(r"coeff_(\d+\.\d+)$", d.name)
    if match:
        coeffs.append(float(match.group(1)))

coeffs = np.array(sorted(coeffs))

# Map frames
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

output_dir = solver_data_dir / "coefficient_comparison_plots"
output_dir.mkdir(parents=True, exist_ok=True)

# Setup colormap normalized to the coefficient range
norm = plt.Normalize(vmin=coeffs.min(), vmax=coeffs.max())
cmap = cm.plasma

for frame_idx, filename in enumerate(sorted(frame_files.keys())):
    match = re.search(r"mean_w_(\d+\.\d+)\.npy$", filename)
    if not match:
        continue

    t_end = float(match.group(1))
    mean_profile_dns = mean_w_profiles[frame_idx]
    mesh_dns = np.linspace(z_domain[0], z_domain[1], len(mean_profile_dns))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), dpi=150, gridspec_kw={"width_ratios": [1.3, 1]}
    )

    # --- LEFT SUBPLOT: Profile Sweep with Colorbar ---
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

    # DNS Reference
    ax1.plot(
        mesh_dns,
        mean_profile_dns,
        color="black",
        linestyle="-",
        linewidth=2.5,
        label="DNS Reference",
        zorder=5,
    )

    l2_errors = []

    for coeff in coeffs:
        if coeff not in frame_files[filename]:
            continue
        les_mean = np.load(frame_files[filename][coeff])
        mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_mean))

        dns_interp = (
            np.interp(mesh_les, mesh_dns, mean_profile_dns)
            if len(mean_profile_dns) != len(les_mean)
            else mean_profile_dns
        )

        rel_err = (
            np.linalg.norm(les_mean - dns_interp) / (np.linalg.norm(dns_interp) + 1e-12)
        ) * 100.0
        l2_errors.append((coeff, rel_err))

        # Plot thin semi-transparent curves using colormap
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

    # Attach Colorbar to Left Plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, orientation="vertical", pad=0.02)
    cbar.set_label("External Coefficient", fontsize=10)

    # --- RIGHT SUBPLOT: Error vs. Correction Coefficient ---
    coeffs_arr, errs_arr = zip(*l2_errors)
    ax2.plot(
        coeffs_arr, errs_arr, color="tab:blue", marker="o", markersize=4, linestyle="-"
    )

    # Highlight Minimum Error
    min_idx = np.argmin(errs_arr)
    ax2.plot(
        coeffs_arr[min_idx],
        errs_arr[min_idx],
        color="tab:red",
        marker="d",
        markersize=12,
        label=f"Optimal: {coeffs_arr[min_idx]:.1f} ({errs_arr[min_idx]:.1f}%)",
    )

    ax2.set_title(r"Relative $L_2$ Error vs. Coefficient", fontsize=12)
    ax2.set_xlabel("External Coefficient", fontsize=11)
    ax2.set_ylabel("Relative $L_2$ Error (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    plt.suptitle(
        f"LES Parameter Sweep (t ∈ [{t_start:.2f}, {t_end:.2f}] s)", fontsize=14, y=0.98
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"profile_sweep_{t_end:.2f}.png", dpi=300)
    plt.close(fig)
