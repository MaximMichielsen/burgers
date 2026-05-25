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


def set_extractions(
    duration: float,
    extraction_amount: int,
    time_step: float,
    mode: str = "linear",
    strict: bool = False,
) -> NDArray | None:
    """Return snapshot extraction times over [0, duration].

    Falls back to every time step if the requested interval is finer than dt.
    """
    extraction_interval = (
        duration / (extraction_amount - 1) if extraction_amount > 1 else duration
    )
    if extraction_interval < time_step:
        extractions = np.arange(0, duration + time_step / 2, step=time_step)
        msg = (
            f"Requested extraction interval ({extraction_interval:.6f}) < dt ({time_step:.6f}). "
            f"Falling back to every time step ({len(extractions)} extractions)."
        )
        if strict:
            raise ValueError(msg)
        print(msg)
        return extractions

    if mode == "linear":
        return np.linspace(0, duration, extraction_amount)

    raise ValueError(f"Mode '{mode}' is not supported. Use 'linear'.")
