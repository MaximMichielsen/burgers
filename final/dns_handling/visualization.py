"""
visualizer.py

Visualization module for DNS and LES simulation fields, projected forcing comparison,
and statistical profiles.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

# Set interactive backend
matplotlib.use("TkAgg")

# Project Directories
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PARSED_DATA_DIR = PROJECT_ROOT / "dns_data" / "curated"
DAT_PARSED_DIR = PARSED_DATA_DIR / "parsed_dat"
STAT_PARSED_DIR = PARSED_DATA_DIR / "parsed_stat"
FILTERED_DIR = PROJECT_ROOT / "dns_data" / "filtered"
FORCING_PATH = DAT_PARSED_DIR / "f_forcing.npy"


def animate_forcing_comparison(
    z_dns: NDArray,
    f_dns: NDArray,
    projected_fields: Dict[str, Tuple[NDArray, NDArray]],
    interval: int = 1,
    save_path: Optional[Path] = None,
) -> FuncAnimation:
    """Animates the high-resolution DNS forcing alongside projected LES forcing fields.

    Parameters
    ----------
    z_dns : NDArray
        Spatial grid coordinates for DNS (513,).
    f_dns : NDArray
        High-resolution DNS forcing array (N_timesteps, 513).
    projected_fields : Dict[str, Tuple[NDArray, NDArray]]
        Dictionary mapping label names to tuples of (z_les_coords, f_les_data).
        Example: {"L2 (17 nodes)": (z_les, f_l2), "H10 (17 nodes)": (z_les, f_h10)}
    interval : int, default=1
        Animation frame delay in milliseconds.
    save_path : Optional[Path], default=None
        If provided, saves the animation to disk (.mp4 or .gif).

    Returns
    -------
    FuncAnimation
        Matplotlib animation instance.
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)

    # 1. Plot DNS Reference line
    (line_dns,) = ax.plot(
        z_dns,
        f_dns[0],
        color="black",
        lw=1.2,
        alpha=0.6,
        linestyle="--",
        label="DNS Forcing (513 pts)",
    )

    # 2. Plot Projected LES lines
    lines_les = {}
    styles = [
        ("crimson", "o-", 2.0),
        ("royalblue", "s--", 2.0),
        ("forestgreen", "^-.", 2.0),
    ]

    for idx, (label, (z_les, f_les)) in enumerate(projected_fields.items()):
        color, marker, lw = styles[idx % len(styles)]
        (line,) = ax.plot(
            z_les,
            f_les[0],
            marker,
            color=color,
            lw=lw,
            markersize=5,
            label=f"LES Projection ({label})",
        )
        lines_les[label] = (line, z_les, f_les)

    # Calculate global y-bounds across DNS and all projected fields
    all_min = min([f_dns.min()] + [f[1].min() for f in projected_fields.values()])
    all_max = max([f_dns.max()] + [f[1].max() for f in projected_fields.values()])
    margin = 0.1 * abs(all_max - all_min) if abs(all_max - all_min) > 1e-12 else 0.1
    down_scale = 0.2

    ax.set_xlim(z_dns.min(), z_dns.max())
    ax.set_ylim((all_min - margin) * down_scale, (all_max + margin) * down_scale)
    ax.set_xlabel("Domain Coordinate $z$")
    ax.set_ylabel("Forcing $f(z, t)$")
    ax.set_title("DNS vs. Filtered LES Forcing Projection Comparison")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")

    time_text = ax.text(
        0.02,
        0.92,
        "",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def update(frame: int):
        # Update DNS Line
        line_dns.set_ydata(f_dns[frame])

        # Update LES Projection Lines
        for label, (line, _, f_les) in lines_les.items():
            line.set_ydata(f_les[frame])

        time_text.set_text(f"Timestep: {frame} / {len(f_dns) - 1}")
        return [line_dns] + [line[0] for line in lines_les.values()] + [time_text]

    anim = FuncAnimation(fig, update, frames=len(f_dns), interval=interval, blit=False)
    plt.tight_layout()

    if save_path is not None:
        anim.save(str(save_path))

    plt.show()
    return anim


def plot_mean_statistics(
    z_coords: NDArray,
    stat_times: NDArray,
    w_mean_data: NDArray,
    field_title: str = r"Mean Velocity $\langle w \rangle$",
):
    """Plots spatial mean profiles at each time frame and domain-averaged statistics over time.

    Parameters
    ----------
    z_coords : NDArray
        Spatial grid coordinates (N_nodes,).
    stat_times : NDArray
        Timestamps for statistical frames (N_frames,).
    w_mean_data : NDArray
        Spatial mean profiles across time (N_frames, N_nodes).
    field_title : str, default=r"Mean Velocity $\langle w \rangle$"
        Latex title formatted string for headers.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    # Subplot 1: Spatial Profiles for each time frame
    for idx, t in enumerate(stat_times):
        ax1.plot(z_coords, w_mean_data[idx], label=f"t = {t:.2f} s")

    ax1.set_title(f"Spatial {field_title} Profiles")
    ax1.set_xlabel("Grid coordinate $z$")
    ax1.set_ylabel(field_title)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # Subplot 2: Domain-Averaged Mean over Time
    domain_avg_w = np.mean(w_mean_data, axis=1)
    ax2.plot(stat_times, domain_avg_w, marker="o", color="darkred", linestyle="-")

    ax2.set_title(f"Domain-Averaged {field_title} over Time")
    ax2.set_xlabel("Time $t$ (s)")
    ax2.set_ylabel(f"Domain Mean {field_title}")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def main():
    """Main execution: Loads DNS and filtered LES forcing data, animates the comparison,
    and then plots velocity mean profile statistics."""
    print("=== Launching Forcing Comparison Animation ===")

    # 1. Load High-Resolution DNS Forcing
    if not FORCING_PATH.exists():
        raise FileNotFoundError(
            f"Missing DNS forcing file at {FORCING_PATH}. Parse .dat files first."
        )

    f_dns = np.load(FORCING_PATH)
    n_dns_nodes = f_dns.shape[1]
    z_dns = np.linspace(0.0, 2.0, n_dns_nodes)  # Matches domain z ∈ [0, 2]

    # 2. Search for saved L2 and H10 filtered arrays
    les_resolution = 9
    l2_file = FILTERED_DIR / f"forcing_l2_{les_resolution}.npy"
    h10_file = FILTERED_DIR / f"forcing_h10_{les_resolution}.npy"

    projected_fields = {}
    z_les = np.linspace(0.0, 2.0, les_resolution)

    if l2_file.exists():
        f_l2 = np.load(l2_file)
        projected_fields[f"$L^2$ ({les_resolution} nodes)"] = (z_les, f_l2)
        print(f"Loaded {l2_file.name}")

    if h10_file.exists():
        f_h10 = np.load(h10_file)
        projected_fields[f"$H^1_0$ ({les_resolution} nodes)"] = (z_les, f_h10)
        print(f"Loaded {h10_file.name}")

    if not projected_fields:
        raise FileNotFoundError(
            f"No filtered files found in {FILTERED_DIR}. Run filter_forcing first."
        )

    # 3. Animate Overlaid Comparison
    animate_forcing_comparison(
        z_dns=z_dns,
        f_dns=f_dns,
        projected_fields=projected_fields,
        interval=1,
    )

    # 4. Plot Mean Velocity Statistics after Animation
    print("=== Plotting Velocity Mean Statistics <w> ===")

    w_stat_path = STAT_PARSED_DIR / "stat_mean_w.npy"
    w_dat_path = DAT_PARSED_DIR / "w_field.npy"

    if w_stat_path.exists():
        w_data = np.load(w_stat_path)
    elif w_dat_path.exists():
        w_data = np.load(w_dat_path)
    else:
        print(
            f"Velocity statistics file not found at {w_stat_path} or {w_dat_path}. "
            "Skipping velocity mean plot."
        )
        return

    sample_indices = np.arange(len(w_data))
    w_sampled_profiles = w_data[sample_indices]

    time_path = STAT_PARSED_DIR / "t_stat.npy"
    if time_path.exists():
        stat_times = np.load(time_path)[sample_indices]
    else:
        stat_times = sample_indices.astype(float)

    plot_mean_statistics(
        z_coords=z_dns,
        stat_times=stat_times,
        w_mean_data=w_sampled_profiles,
        field_title=r"Mean Velocity $\langle w \rangle$",
    )


if __name__ == "__main__":
    main()
