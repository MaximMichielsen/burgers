"""Utility functions for input/output related aspects."""

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
