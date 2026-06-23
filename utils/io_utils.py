"""Utility functions for input/output related aspects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from solvers.sgsp_training_data_generator_stash import BurgersDataGenerator

if TYPE_CHECKING:
    from problems_and_configurations.disc_config import DiscretisationConfig
    from problems_and_configurations.problems import Problem


def run_data_generator(
    problem: Problem,
    disc_cfg: DiscretisationConfig,
    master_path: Path,
    dns_save_path: Path,
    projection_data_path: Path,
    sgsp_data_training_path: Path,
    t_start: float = 0.0,
    append_mode: bool = False
) -> None:
    """Run DNS and assemble SGSP training data."""
    solver = BurgersDataGenerator(
        problem,
        disc_cfg,
        "dns",
        master_path,
        dns_save_path,
        sgsp_training_data_path=sgsp_data_training_path,
        projection_save_path=projection_data_path,
        t_start=t_start,
        append_mode = append_mode
    )
    solver.print_configuration()
    solver.run_simulation()
    solver.post_processing()


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
