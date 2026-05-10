"""File for general functions."""

from math import ceil, floor, log2
from pathlib import Path
from typing import Any, Callable

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from burgers import Burgers


def set_extractions(
    duration: float,
    extraction_amount: int,
    time_step: float,
    mode: str = "linear",
    strict: bool = False,
) -> NDArray | None:
    """Set step size for extracting instances."""
    extraction_interval = (
        duration / (extraction_amount - 1) if extraction_amount > 1 else duration
    )
    if time_step > extraction_interval and strict:
        raise ValueError(
            f"Time step ({time_step}) is larger than the requested extraction "
            f"interval ({extraction_interval:.4f}). Reduce extraction_amount."
        )

    elif time_step > extraction_interval and not strict:
        extraction_amount_ = int(duration / (time_step))
        print(
            f"Time step ({time_step}) is larger than requested extraction. "
            f"Setting extractions to maximum possible value."
        )
        return np.linspace(start=0, stop=duration, num=extraction_amount_ + 1)

    if mode == "linear":
        return np.linspace(start=0, stop=duration, num=extraction_amount)

    else:
        raise ValueError("NO OTHER METHOD THAN LINEAR CURRENTLY")


def plot_solutions_from_directory(directory: str | Path) -> None:
    """Plot all solution CSVs in a directory."""
    directory = Path(directory)
    files = sorted(directory.glob("sol_t*.csv"))

    times = []
    solutions = []
    x = None

    for file in files:
        time = float(file.stem.split("t")[-1])
        times.append(time)
        data = np.loadtxt(file, delimiter=",", skiprows=1)

        if x is None:
            x = data[:, 1]

        u = data[:, 2]
        solutions.append(u)
    # sort by time (in case filesystem order is weird)
    times, solutions = zip(*sorted(zip(times, solutions)))
    # ---- plotting ----
    plt.figure(figsize=(10, 6))

    for t, u in zip(times, solutions):
        plt.plot(x, u, label=f"t={t:.2f}")

    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Solution evolution")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_solutions_from_directory_animated(
    directory: str | Path, interval: int = 100, repeat: bool = True
) -> FuncAnimation:
    """Animate all solution CSVs in a directory over time."""
    directory = Path(directory)
    files = sorted(directory.glob("sol_t*.csv"))

    if not files:
        raise FileNotFoundError(f"No sol_t*.csv files found in {directory}")

    times = []
    solutions = []
    x = None

    for file in files:
        time = float(file.stem.split("t")[-1])
        times.append(time)

        data = np.loadtxt(file, delimiter=",", skiprows=1)
        if x is None:
            x = data[:, 1]
        solutions.append(data[:, 2])
    # sort by time
    times, solutions = zip(*sorted(zip(times, solutions)))
    solutions = list(solutions)
    # ---- fixed y-axis limits across all frames ----
    u_min = min(u.min() for u in solutions)
    u_max = max(u.max() for u in solutions)
    pad = 0.05 * (u_max - u_min)
    # ---- set up figure ----
    fig, ax = plt.subplots(figsize=(10, 6))
    (line,) = ax.plot(x, solutions[0], lw=2, color="royalblue")
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment="top"
    )

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(u_min - pad, u_max + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Solution evolution")
    ax.grid(True)

    def update(frame: int) -> tuple[Any, Any]:
        """Update frame."""
        line.set_ydata(solutions[frame])
        time_text.set_text(f"t = {times[frame]:.4f}")
        return line, time_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=interval,
        blit=True,
        repeat=repeat,
    )

    plt.tight_layout()
    plt.show()
    return ani  # keep reference alive so GC doesn't kill it


def create_config_variables(
    n_nodes: list, length: float, viscosity: float, initial_condition: Callable = np.sin
) -> tuple[list, list, list, list]:
    """Create the necessary configuration variables based on the amount of nodes in a list."""
    meshes = []
    delta_xs = []
    initial_solutions = []
    time_steps = []

    for i, amount in enumerate(n_nodes):
        mesh, delta_x = np.linspace(start=0, stop=length, num=amount, retstep=True)
        initial_solution = initial_condition(mesh)
        time_step = compute_time_step(mesh, max(initial_solution), viscosity)

        meshes.append(mesh)
        delta_xs.append(delta_x)
        initial_solutions.append(initial_solution)
        time_steps.append(time_step)

    return meshes, delta_xs, initial_solutions, time_steps


def calc_required_grid_points(
    length: float,
    reynolds: float,
    factor_spatial: float,
    factor_points: float,
    round_to_power_of_2: bool = True,
) -> int:
    """Calculate the smallest characteristic length scale associated with the DNS run."""
    spatial_step = factor_spatial * length / reynolds
    required_points = factor_points * length / spatial_step
    if not round_to_power_of_2:
        return int(required_points)

    power = log2(required_points)
    ceiled_power = ceil(power)
    return 2**ceiled_power


def round_down(value: float, decimals: int) -> float:
    """Round a float down to the specified decimal."""
    factor = 10**decimals
    return floor(value * factor) / factor


def compute_time_step(
    mesh: np.ndarray, max_velocity: float, viscosity: float, do_round_down: bool = True
) -> float:
    """CFL-based time step: minimum of convective and diffusive limits."""
    dx = abs(mesh[1] - mesh[0])
    dt = min(dx / max_velocity, dx**2 / viscosity)
    return round_down(dt, 4) if do_round_down else dt


def implicit_euler_first_order(field: NDArray | float, h: float) -> NDArray:
    """Approximate the derivative of a field with step-size h."""
    return (np.roll(field, -1) - np.roll(field, 1)) / (2 * h)


def run_config(
    configuration: dict, return_directory: bool = True
) -> tuple[Path, str] | None:
    """Run a config and return (absolute solver_data path, relative run folder name)."""
    solver = Burgers(configuration=configuration)
    solver.print_configuration()
    solver.run_simulation()
    solver.post_logging()
    return (
        solver.run_dir,
        solver.run_dir.parent.name if return_directory else None,
    )  # absolute, relative run folder
