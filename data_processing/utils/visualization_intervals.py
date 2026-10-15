"""
visualizer_intervals.py

Plots interval-mean profiles (<w> and <w'w'>) computed per window, instead of the
cumulative profiles. Reads files named <prefix>_<start>_<end>.npy from
dns_data/intervals.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

matplotlib.use("TkAgg")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
INTERVAL_DIR = PROJECT_ROOT / "dns_data" / "intervals"
OUTPUT_DIR = PROJECT_ROOT / "dns_data" / "visualized"

MEAN_PREFIX = "w_mean"
REYNOLDS_PREFIX = "ww"
DOMAIN_LENGTH = 2.0

_FILENAME_PATTERN = re.compile(r"^(?P<start>\d+\.?\d*)_(?P<end>\d+\.?\d*)$")


@dataclass
class IntervalProfile:
    """One interval-mean profile with its time window."""

    start: float
    end: float
    data: NDArray

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.start + self.end)

    @property
    def label(self) -> str:
        return f"t = {self.start:.2f}-{self.end:.2f} s"


def load_interval_profiles(directory: Path, prefix: str) -> List[IntervalProfile]:
    """Load all <prefix>_<start>_<end>.npy files, sorted by start time."""
    profiles = []
    for path in directory.glob(f"{prefix}_*.npy"):
        match = _FILENAME_PATTERN.match(path.stem[len(prefix) + 1 :])
        if match is None:
            print(f"Skipping unrecognised file name: {path.name}")
            continue
        profiles.append(
            IntervalProfile(
                start=float(match["start"]),
                end=float(match["end"]),
                data=np.load(path),
            )
        )

    if not profiles:
        raise FileNotFoundError(f"No '{prefix}_*.npy' files found in {directory}")

    profiles.sort(key=lambda p: p.start)
    return profiles


def report_gaps(profiles: List[IntervalProfile], name: str, tol: float = 0.05) -> None:
    """Print a note where consecutive windows do not touch."""
    for prev, nxt in zip(profiles[:-1], profiles[1:]):
        if abs(nxt.start - prev.end) > tol:
            print(
                f"[{name}] gap between {prev.end:.2f} s and {nxt.start:.2f} s "
                "(no interval file for that window)"
            )


def plot_interval_statistics(
    z_coords: NDArray,
    mean_profiles: List[IntervalProfile],
    reynolds_profiles: List[IntervalProfile],
    save_dir: Path,
    show_plot: bool = True,
) -> None:
    """Plot interval-mean <w> profiles and interval <w'w'> profiles."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # FILE 1: interval_mean_w_profiles.png
    # -------------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    for profile in mean_profiles:
        ax1.plot(z_coords, profile.data, label=profile.label)

    ax1.set_title(r"Interval Mean Velocity $\langle w \rangle$ Profiles")
    ax1.set_xlabel(r"Grid coordinate $z$")
    ax1.set_ylabel(r"Mean Velocity $\langle w \rangle$")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    midpoints = np.array([p.midpoint for p in mean_profiles])
    domain_avg = np.array([p.data.mean() for p in mean_profiles])
    ax2.plot(midpoints, domain_avg, marker="o", color="darkred", linestyle="-")

    ax2.set_xlabel("Time $t$ (s), interval midpoint")
    ax2.set_title(r"Domain-Averaged Interval Mean $\langle w \rangle$ over Time")
    ax2.set_ylabel(r"Domain Mean Velocity $\langle w \rangle$")
    ax2.grid(True, linestyle="--", alpha=0.6)

    fig1.tight_layout()
    fig1.savefig(save_dir / "interval_mean_w_profiles.png", dpi=300)
    if show_plot:
        plt.show()
    plt.close(fig1)

    # -------------------------------------------------------------
    # FILE 2: interval_reynolds_stress_profiles.png
    # -------------------------------------------------------------
    fig2, ax_re = plt.subplots(figsize=(8, 5), dpi=120)

    for profile in reynolds_profiles:
        ax_re.plot(z_coords, profile.data, label=profile.label)

    ax_re.set_title(r"Interval Reynolds Stress $\langle w'w' \rangle$ Profiles")
    ax_re.set_xlabel("Grid coordinate $z$")
    ax_re.grid(True, linestyle="--", alpha=0.6)
    ax_re.legend(loc="upper right", fontsize=8)

    fig2.tight_layout()
    fig2.savefig(save_dir / "interval_reynolds_stress_profiles.png", dpi=300)
    if show_plot:
        plt.show()
    plt.close(fig2)


def main() -> None:
    mean_profiles = load_interval_profiles(INTERVAL_DIR, MEAN_PREFIX)
    reynolds_profiles = load_interval_profiles(INTERVAL_DIR, REYNOLDS_PREFIX)

    report_gaps(mean_profiles, MEAN_PREFIX)
    report_gaps(reynolds_profiles, REYNOLDS_PREFIX)

    n_points = mean_profiles[0].data.shape[-1]
    z_coords = np.linspace(0.0, DOMAIN_LENGTH, n_points)

    plot_interval_statistics(
        z_coords=z_coords,
        mean_profiles=mean_profiles,
        reynolds_profiles=reynolds_profiles,
        save_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
