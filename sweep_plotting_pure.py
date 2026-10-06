import re
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# Directory Setup
PROJECT_ROOT = Path(__file__).resolve().parent
PARSED_STAT_DIR = PROJECT_ROOT / "dns_data" / "curated" / "stat"

n_nodes_les = 17
tau_params = 2
dt = 0.001

# Path to the offset test runs directory
solver_data_dir = PROJECT_ROOT / "solver_data" / "les_offset_tests"

# Load DNS profiles array
mean_w_profiles = np.load(PARSED_STAT_DIR / "mean_w.npy")

# Define target coefficient values [0.6, ..., 1.6] excluding 1.0
coeffs = [
    round(c, 1) for c in np.arange(0.6, 5.05, 0.1) if not np.isclose(round(c, 1), 1.0)
]

# Discover available files per timestep frame across all coefficient directories
frame_files = {}  # Map: filename -> {coeff: full_path}

for coeff in coeffs:
    profiles_dir = (
        solver_data_dir
        / f"les_n{n_nodes_les}_p{tau_params}_dt_{dt}_coeff_{coeff:.1f}"
        / "mean_profiles"
    )

    if not profiles_dir.exists():
        continue

    for les_path in profiles_dir.glob("mean_w_*.npy"):
        frame_files.setdefault(les_path.name, {})[coeff] = les_path

# Output directory for summary plot comparisons
output_dir = solver_data_dir / "coefficient_comparison_plots"
output_dir.mkdir(parents=True, exist_ok=True)

t_start = 308.01
z_domain = (0.0, 2.0)

# Colors for plotting coefficients using a colormap
colors = cm.plasma(np.linspace(0.1, 0.9, len(coeffs)))

# Iterate through each timestamp frame
for frame_idx, filename in enumerate(sorted(frame_files.keys())):
    match = re.search(r"mean_w_(\d+\.\d+)\.npy$", filename)
    if not match:
        continue

    t_end = float(match.group(1))
    time_interval_str = f"t ∈ [{t_start:.2f}, {t_end:.2f}] s"

    # DNS Reference setup
    mean_profile_dns = mean_w_profiles[frame_idx]
    mesh_dns = np.linspace(z_domain[0], z_domain[1], len(mean_profile_dns))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # Zero Baseline
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)

    # Plot DNS Reference
    ax.plot(
        mesh_dns,
        mean_profile_dns,
        color="black",
        linestyle="-",
        linewidth=2.0,
        label=f"DNS Reference ({len(mesh_dns)} pts)",
        zorder=4,
    )

    all_data = [mean_profile_dns]

    # Plot each LES run for this frame
    for idx, coeff in enumerate(coeffs):
        if coeff not in frame_files[filename]:
            continue

        les_path = frame_files[filename][coeff]
        les_mean = np.load(les_path)
        all_data.append(les_mean)

        mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_mean))

        # Interpolate DNS onto LES grid for relative error calculation
        dns_interp = (
            np.interp(mesh_les, mesh_dns, mean_profile_dns)
            if len(mean_profile_dns) != len(les_mean)
            else mean_profile_dns
        )

        l2_error = float(np.linalg.norm(les_mean - dns_interp))
        dns_norm = np.linalg.norm(dns_interp)
        relative_l2_error = (l2_error / (dns_norm + 1e-12)) * 100.0

        ax.plot(
            mesh_les,
            les_mean,
            color=colors[idx],
            linestyle="--",
            linewidth=1.2,
            marker="o",
            markersize=3,
            label=f"Coeff {coeff:.1f} (Rel Err: {relative_l2_error:.1f}%)",
            zorder=3,
        )

    # Compute Dynamic Y-Limits with 10% Padding
    flat_data = np.concatenate(all_data)
    y_min, y_max = flat_data.min(), flat_data.max()
    y_range = y_max - y_min if y_max != y_min else 1.0
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Formatting and Labels
    ax.set_title(
        r"Mean Velocity Profile Comparison $\langle w \rangle$ Across External Coefficients"
        + f"\n({time_interval_str})",
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel(r"Domain Coordinate $z$", fontsize=11)
    ax.set_ylabel(r"Mean Velocity $\langle w \rangle$", fontsize=11)

    ax.set_xlim(z_domain)
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    ax.minorticks_on()

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=8.5,
        ncol=2,
    )

    plt.tight_layout()

    # Save summary plot per frame
    output_plot_path = output_dir / f"profile_comparison_{t_end:.2f}.png"
    plt.savefig(output_plot_path, dpi=300)
    plt.close(fig)

print(f"Plots generated and saved to: {output_dir}")
