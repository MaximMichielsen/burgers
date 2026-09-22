"""
File to read DNS data (.crd, .dat, .stat).
Parses, saves binary files/metadata, and visualizes mesh, solution evolution, and statistics.
"""

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Paths Setup
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

RAW_DATA_DIR = PROJECT_ROOT / "dns_data" / "raw"
PARSED_DATA_DIR = PROJECT_ROOT / "dns_data" / "curated"
DAT_PARSED_DIR = PARSED_DATA_DIR / "parsed_dat"
STAT_PARSED_DIR = PARSED_DATA_DIR / "parsed_stat"

CRD_PATH = RAW_DATA_DIR / "burgers_1D.crd"
DAT_PATH = RAW_DATA_DIR / "burgers_1D.dat"
STAT_PATH = RAW_DATA_DIR / "burgers_1D.stat"


# TODO: rename saving path files of the reynolds terms to actually reflect the reynolds stuff, not var!


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
