"""Utility functions for input/output related aspects."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def read_data(
    directory: str | Path,
    final_only: bool = False,
) -> tuple[NDArray, NDArray] | tuple[NDArray, list, list, list]:
    """Read chronologically sorted solution snapshots from directory.

    Returns (solution, mesh) if final_only=True, else (mesh, times, solutions, forcings).
    """
    directory = Path(directory)
    files = sorted(directory.glob("sol_t*.csv"))
    if not files:
        raise FileNotFoundError(f"No sol_t*.csv files found in {directory}")

    times, solutions, forcings = [], [], []
    mesh = None

    for file_path in files:
        time_value = float(file_path.stem.split("t")[-1])
        times.append(time_value)
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        if mesh is None:
            mesh = data[:, 1]
        solutions.append(data[:, 2])
        try:
            forcings.append(data[:, 3])
        except IndexError:
            forcings.append(np.zeros_like(data[:, 2]))

    times, solutions = zip(*sorted(zip(times, solutions)))

    if final_only:
        return list(solutions)[-1], mesh

    return mesh, list(times), list(solutions), list(forcings)


def load_first_projected_solution(projection_dir: Path) -> np.ndarray:
    """Load the first projected solution snapshot from the projection directory."""
    csv_files = sorted(projection_dir.glob("sol_t*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No projected solution CSVs found in {projection_dir}")
    first_csv_path = csv_files[0]
    velocity_values: list[float] = []
    with open(first_csv_path, newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for csv_row in reader:
            velocity_values.append(float(csv_row["velocity"]))
    return np.array(velocity_values)


def compute_adjusted_dt(
    dt_nominal: float, time_end: float | int, time_start: float = 0.0
) -> tuple[float, int]:
    """Adjust dt so that time_end is hit exactly.

    Rounds to the nearest integer step count and recomputes dt.
    The relative change in dt is O(1/n_steps), negligible in practice.
    """
    time_span = time_end - time_start
    n_steps = max(1, round(time_span / dt_nominal))
    dt_adjusted = time_span / n_steps
    return dt_adjusted, n_steps
