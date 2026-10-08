import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Directory Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_STAT_DIR = PROJECT_ROOT / "dns_data" / "curated" / "stat"

n_nodes_les = 17
tau_params = 2

# Load target DNS profile explicitly (Frame index 0)
mean_w_profiles = np.load(PARSED_STAT_DIR / "mean_w.npy")
mean_profile_dns = mean_w_profiles[0]

# Resolve target directory for LES profiles
profiles_dir = (
    PROJECT_ROOT
    / "solver_data"
    / "les_offset_sweeps_vary_c1"
    / "les_n17_p2_dt_0.001_c1_4.90_c2_4.90"
    / "mean_profiles"
)

# Automatically locate all .npy LES profiles sorted by timestamp
les_files = sorted(profiles_dir.glob("mean_w_*.npy"))

t_start = 308.01
z_domain = (0.0, 2.0)

# Iterate through each LES profile file
for les_path in les_files:
    match = re.search(r"mean_w_(\d+\.\d+)\.npy$", les_path.name)
    if not match:
        continue

    t_end = float(match.group(1))
    time_interval_str = f"t ∈ [{t_start:.2f}, {t_end:.2f}] s"

    # Load LES array
    les_mean = np.load(les_path)

    # Spatial Grids
    mesh_dns = np.linspace(z_domain[0], z_domain[1], len(mean_profile_dns))
    mesh_les = np.linspace(z_domain[0], z_domain[1], len(les_mean))

    # Interpolate DNS onto LES mesh
    dns_interp = np.interp(mesh_les, mesh_dns, mean_profile_dns)

    # L2 Error Metrics
    l2_error = float(np.linalg.norm(les_mean - dns_interp))
    dns_norm = np.linalg.norm(dns_interp)
    relative_l2_error = (l2_error / (dns_norm + 1e-12)) * 100.0

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

    # L2 Error Score Box
    score_text = (
        rf"$\mathrm{{L}}_2$ Error: {l2_error:.4e}"
        + f"\nRel. Error: {relative_l2_error:.2f}%"
    )
    ax.text(
        0.03,
        0.05,
        score_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.85,
        ),
        zorder=5,
    )

    plt.tight_layout()

    # Save figure per frame
    output_plot_path = profiles_dir / f"profile_comparison_{t_end:.2f}.png"
    plt.savefig(output_plot_path, dpi=300)
    plt.close(fig)
