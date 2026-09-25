import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Directory Setup
PROJECT_ROOT = Path(__file__).resolve().parent
PARSED_STAT_DIR = PROJECT_ROOT / "dns_data" / "curated" / "stat"

n_nodes_les = 65
tau_params = 2

# Load DNS profiles array
mean_w_profiles = np.load(PARSED_STAT_DIR / "mean_w.npy")

# Resolve target directory for LES profiles

profiles_dir = (
    PROJECT_ROOT / "solver_data" / f"les_n{n_nodes_les}_p{tau_params}" / "mean_profiles"
)

# Automatically locate all .npy LES profiles sorted by timestamp
les_files = sorted(profiles_dir.glob("mean_w_*.npy"))

t_start = 308.01
z_domain = (0.0, 2.0)

# Iterate through each LES profile file
for frame_idx, les_path in enumerate(les_files):
    # Extract end time dynamically from filename (e.g., 'mean_w_309.01.npy' -> 309.01)
    match = re.search(r"mean_w_(\d+\.\d+)\.npy$", les_path.name)
    if not match:
        continue

    t_end = float(match.group(1))
    time_interval_str = f"t ∈ [{t_start:.2f}, {t_end:.2f}] s"

    # Load arrays
    mean_profile_dns = mean_w_profiles[frame_idx]
    les_mean = np.load(les_path)

    # Spatial Grids
    mesh_dns = np.linspace(z_domain[0], z_domain[1], len(mean_profile_dns))
    mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_mean))

    # Figure Setup
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    # Zero Baseline
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)

    # Plot DNS Reference
    ax.plot(
        mesh_dns,
        mean_profile_dns,
        color="royalblue",
        linestyle="-",
        linewidth=1.2,
        label=f"DNS Reference ({len(mesh_dns)} pts)",
        zorder=2,
    )

    # Plot LES Simulation
    ax.plot(
        mesh_les,
        les_mean,
        color="tab:orange",
        linestyle="--",
        linewidth=1.0,
        marker="o",
        markevery=max(1, len(mesh_les) // 16),
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label=f"LES Model ({len(mesh_les)} pts)",
        zorder=3,
    )

    # Dynamic Y-Limits with 10% Padding
    all_data = np.concatenate([mean_profile_dns, les_mean])
    y_min, y_max = all_data.min(), all_data.max()
    y_range = y_max - y_min if y_max != y_min else 1.0
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Styling and Labels
    ax.set_title(
        r"Mean Velocity Profile Comparison $\langle w \rangle$ "
        + f"({time_interval_str})",
        fontsize=13,
        pad=10,
    )
    ax.set_xlabel(r"Domain Coordinate $z$", fontsize=11)
    ax.set_ylabel(r"Mean Velocity $\langle w \rangle$", fontsize=11)

    ax.set_xlim(z_domain)
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    ax.minorticks_on()

    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)

    plt.tight_layout()

    # Save figure per frame
    output_plot_path = profiles_dir / f"profile_comparison_{t_end:.2f}.png"
    plt.savefig(output_plot_path, dpi=300)
    plt.close(fig)
