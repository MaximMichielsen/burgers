"""
File to read DNS data (.crd, .dat, .stat).
Parses, saves binary files/metadata, and visualizes mesh, solution evolution, and statistics.
"""

import re
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("TkAgg")  # Must be set before importing pyplot

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paths Setup
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

RAW_DATA_DIR = PROJECT_ROOT / "dns_data" / "raw"
PARSED_DATA_DIR = PROJECT_ROOT / "dns_data" / "curated"
DAT_PARSED_DIR = PARSED_DATA_DIR / "parsed_dat"
STAT_PARSED_DIR = PARSED_DATA_DIR / "parsed_stat"

CRD_PATH = RAW_DATA_DIR / "burgers_1D.crd"
DAT_PATH = RAW_DATA_DIR / "burgers_1D.dat"
STAT_PATH = RAW_DATA_DIR / "burgers_1D.stat"


# Data Parsers
def read_crd_file(file_path: Path) -> tuple[dict, NDArray]:
    """
    Parses a .crd file.
    Returns metadata dict and a 1D numpy array of z-coordinates.
    """
    metadata = {}
    coordinates = []

    header_pattern = re.compile(r"npNorm=(\d+)\s+at\s+x=([\d.-]+)\s+y=([\d.-]+)")

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                match = header_pattern.search(line)
                if match:
                    metadata["npNorm"] = int(match.group(1))
                    metadata["x"] = float(match.group(2))
                    metadata["y"] = float(match.group(3))
                continue

            try:
                coordinates.append(float(line))
            except ValueError:
                continue

    return metadata, np.array(coordinates)


def parse_dns_forcing_file(
    file_path: Path,
) -> tuple[NDArray, NDArray, NDArray, NDArray, dict]:
    """
    Parses multi-timestep DNS forcing files.
    Returns timestamps, velocity field w, forcing field f, time-steps dt, and metadata.
    """
    times = []
    w_list = []
    f_list = []
    dt_list = []

    current_w = []
    current_f = []
    metadata = {}

    time_pattern = re.compile(r"dt=([\d.eE+-]+)\s+time=([\d.eE+-]+)")
    resolution_pattern = re.compile(r"resolution:(\d+)")

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                res_match = resolution_pattern.search(line)
                if res_match:
                    metadata["resolution"] = int(res_match.group(1))

                time_match = time_pattern.search(line)
                if time_match:
                    if current_w:
                        w_list.append(current_w)
                        f_list.append(current_f)
                        current_w = []
                        current_f = []

                    dt_list.append(float(time_match.group(1)))
                    times.append(float(time_match.group(2)))
                continue

            parts = line.split()
            if len(parts) == 2:
                try:
                    w_val, f_val = float(parts[0]), float(parts[1])
                    current_w.append(w_val)
                    current_f.append(f_val)
                except ValueError:
                    continue

        if current_w:
            w_list.append(current_w)
            f_list.append(current_f)

    return (
        np.array(times),
        np.array(w_list),
        np.array(f_list),
        np.array(dt_list),
        metadata,
    )


def parse_stat_file(file_path: Path):
    """
    Parses a DNS statistics (.stat) file and returns time series for:
    <w>, <w'w'>, <fz>, <fzfz>, as well as timestamps, step numbers, and sample counts.
    """
    times = []
    steps = []
    samples = []

    mean_w_list = []
    mean_reynolds_w_list = []
    mean_fz_list = []
    mean_reynolds_fz_list = []

    cur_w, cur_ww, cur_fz, cur_fzfz = [], [], [], []

    header_pattern = re.compile(
        r"#\s*t=([\d.eE+-]+)\s+step=(\d+),\s*(\d+)(?:\s+samples)?"
    )

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                match = header_pattern.search(line)
                if match:
                    if cur_w:
                        mean_w_list.append(cur_w)
                        mean_reynolds_w_list.append(cur_ww)
                        mean_fz_list.append(cur_fz)
                        mean_reynolds_fz_list.append(cur_fzfz)
                        cur_w, cur_ww, cur_fz, cur_fzfz = [], [], [], []

                    times.append(float(match.group(1)))
                    steps.append(int(match.group(2)))
                    samples.append(int(match.group(3)))
                continue

            parts = line.split()
            if len(parts) == 4:
                try:
                    vals = [float(p) for p in parts]
                    cur_w.append(vals[0])
                    cur_ww.append(vals[1])
                    cur_fz.append(vals[2])
                    cur_fzfz.append(vals[3])
                except ValueError:
                    continue

        if cur_w:
            mean_w_list.append(cur_w)
            mean_reynolds_w_list.append(cur_ww)
            mean_fz_list.append(cur_fz)
            mean_reynolds_fz_list.append(cur_fzfz)

    return (
        np.array(times),
        np.array(steps),
        np.array(samples),
        np.array(mean_w_list),
        np.array(mean_reynolds_w_list),
        np.array(mean_fz_list),
        np.array(mean_reynolds_fz_list),
    )


# Execution Pipeline
if __name__ == "__main__":
    # Ensure output directories exist
    PARSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DAT_PARSED_DIR.mkdir(parents=True, exist_ok=True)
    STAT_PARSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. CRD Mesh Parsing
    crd_metadata, z_coords = read_crd_file(CRD_PATH)
    print(f"Loaded mesh from {CRD_PATH.name} with {len(z_coords)} nodes.")

    # 2. DAT Forcing Parsing & Saving
    if DAT_PATH.exists():
        times, w_arr, f_arr, dt_arr, dat_metadata = parse_dns_forcing_file(DAT_PATH)
        np.save(DAT_PARSED_DIR / "dns_times.npy", times)
        np.save(DAT_PARSED_DIR / "w_field.npy", w_arr)
        np.save(DAT_PARSED_DIR / "f_forcing.npy", f_arr)
        np.save(DAT_PARSED_DIR / "dt_steps.npy", dt_arr)
        print(f"Saved DAT fields with shape {w_arr.shape} to {DAT_PARSED_DIR}")

    # 3. STAT File Parsing, Binary Saving, and Metadata Generation
    if STAT_PATH.exists():
        (
            times_arr,
            steps_arr,
            samples_arr,
            w_mean_arr,
            mean_reynolds_w_arr,
            fz_mean_arr,
            mean_reynolds_fz_arr,
        ) = parse_stat_file(STAT_PATH)

        # Save individual statistical binary arrays
        np.save(STAT_PARSED_DIR / "stat_times.npy", times_arr)
        np.save(STAT_PARSED_DIR / "stat_steps.npy", steps_arr)
        np.save(STAT_PARSED_DIR / "stat_samples.npy", samples_arr)
        np.save(STAT_PARSED_DIR / "stat_mean_w.npy", w_mean_arr)
        np.save(STAT_PARSED_DIR / "stat_var_w.npy", mean_reynolds_w_arr)
        np.save(STAT_PARSED_DIR / "stat_mean_fz.npy", fz_mean_arr)
        np.save(STAT_PARSED_DIR / "stat_var_fz.npy", mean_reynolds_fz_arr)

        # Generate and save metadata text file
        metadata_txt_path = STAT_PARSED_DIR / "stat_metadata.txt"
        with open(metadata_txt_path, "w") as meta_file:
            meta_file.write("=== DNS STAT FILE METADATA ===\n")
            meta_file.write(f"Source File       : {STAT_PATH.name}\n")
            meta_file.write(f"Total Time Frames : {len(times_arr)}\n")
            meta_file.write(
                f"Spatial Nodes     : {w_mean_arr.shape[1] if w_mean_arr.ndim > 1 else 0}\n"
            )
            if len(times_arr) > 0:
                meta_file.write(
                    f"Time Range        : {times_arr.min():.4f} to {times_arr.max():.4f}\n"
                )
            meta_file.write("\n=== FRAME DETAILS ===\n")
            meta_file.write("FrameIdx | Time (t)   | Step       | Samples\n")
            meta_file.write("-" * 46 + "\n")
            for i, (t, st, sm) in enumerate(zip(times_arr, steps_arr, samples_arr)):
                meta_file.write(f"{i:<8} | {t:<10.4f} | {st:<10} | {sm:<8}\n")

        print(f"Saved STAT arrays and metadata text file to {STAT_PARSED_DIR}")

    # 4. Animation of Velocity w Evolution
    w_field_path = DAT_PARSED_DIR / "w_field.npy"
    if w_field_path.exists():
        w_field = np.load(w_field_path)

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        (line,) = ax.plot(z_coords, w_field[0], color="navy", lw=2, label=r"$w(z, t)$")

        ax.set_xlim(z_coords.min(), z_coords.max())
        ax.set_ylim(
            w_field.min() - 0.1 * abs(w_field.min()),
            w_field.max() + 0.1 * abs(w_field.max()),
        )
        ax.set_xlabel("Grid coordinate $z$")
        ax.set_ylabel("Velocity $w$")
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

        def update(frame):
            line.set_ydata(w_field[frame])
            time_text.set_text(f"Timestep: {frame} / {len(w_field) - 1}")
            return line, time_text

        anim = FuncAnimation(fig, update, frames=len(w_field), interval=1, blit=False)

        plt.tight_layout()
        plt.show()

    # 5. Plotting Mean Velocity <w> Statistics
    stat_mean_w_path = STAT_PARSED_DIR / "stat_mean_w.npy"
    stat_times_path = STAT_PARSED_DIR / "stat_times.npy"

    if stat_mean_w_path.exists() and stat_times_path.exists():
        w_mean_data = np.load(stat_mean_w_path)  # Shape: (N_frames, N_nodes)
        stat_times = np.load(stat_times_path)  # Shape: (N_frames,)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

        # Subplot 1: Spatial Mean Velocity Profiles <w(z)> for each time frame
        for idx, t in enumerate(stat_times):
            ax1.plot(z_coords, w_mean_data[idx], label=f"t = {t:.2f} s")

        ax1.set_title(r"Mean Velocity Profiles $\langle w(z) \rangle$")
        ax1.set_xlabel("Grid coordinate $z$")
        ax1.set_ylabel(r"$\langle w \rangle$")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper right")

        # Subplot 2: Domain-Averaged Mean Velocity over Time
        domain_avg_w = np.mean(w_mean_data, axis=1)  # Spatial average per frame
        ax2.plot(stat_times, domain_avg_w, marker="o", color="darkred", linestyle="-")

        ax2.set_title(r"Domain-Averaged $\langle w \rangle$ over Time")
        ax2.set_xlabel("Time $t$ (s)")
        ax2.set_ylabel(r"Domain Mean $\langle w \rangle$")
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.show()
