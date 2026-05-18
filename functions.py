"""File for general functions."""

from math import ceil, floor, log2
from pathlib import Path
from typing import Any, Callable

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray
from typing import List, Union

from burgers_2 import Burgers

from dataclasses import dataclass, field
from typing import Optional


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
    if extraction_interval < time_step:
        if strict:
            raise ValueError(
                f"Requested extraction interval ({extraction_interval}) is lower than the given time_step ({time_step})."
            )

        else:
            extractions = np.arange(0, duration + (time_step / 2), step=time_step)
            print(
                f"Requested extraction interval ({extraction_interval}) is lower than the given time_step ({time_step})."
                f"Setting extractions to maximum possible frequency (every time step): extraction amount: ({len(extractions)})."
            )
            return extractions

    if mode == "linear":
        return np.linspace(start=0, stop=duration, num=extraction_amount)

    raise ValueError(f"Mode '{mode}' is not supported. Use 'linear'.")


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


def plot_multiple_solutions_animated(
    directories: List[Union[str, Path]],
    labels: List[str] = None,
    interval: int = 100,
    repeat: bool = True,
) -> animation.FuncAnimation:
    """
    Animate solution CSVs from multiple directories to compare them.
    Assumes all directories contain files with matching timestamps.
    """
    if labels is None:
        labels = [Path(d).name for d in directories]

    all_times = []
    all_solutions = []  # List of lists: [dir_idx][time_idx]
    x_axis = None

    # ---- Data Collection ----
    for directory in directories:
        directory = Path(directory)
        files = sorted(directory.glob("sol_t*.csv"))

        if not files:
            print(f"Warning: No files found in {directory}")
            continue

        dir_times = []
        dir_sols = []

        for file in files:
            t = float(file.stem.split("t")[-1])
            data = np.loadtxt(file, delimiter=",", skiprows=1)
            if x_axis is None:
                x_axis = data[:, 1]
            dir_times.append(t)
            dir_sols.append(data[:, 2])

        # Sort current directory by time
        dir_times, dir_sols = zip(*sorted(zip(dir_times, dir_sols)))
        all_times.append(dir_times)
        all_solutions.append(list(dir_sols))

    # Use the first directory's timestamps as the master clock
    master_times = all_times[0]
    num_frames = len(master_times)

    # ---- Axis Limits ----
    u_min = min(np.min(s) for traj in all_solutions for s in traj)
    u_max = max(np.max(s) for traj in all_solutions for s in traj)
    pad = 0.05 * (u_max - u_min) if u_max != u_min else 1.0

    # ---- Set up Plot ----
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []
    for i, label in enumerate(labels):
        (line,) = ax.plot(x_axis, all_solutions[i][0], lw=2, label=label)
        lines.append(line)

    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment="top"
    )

    ax.set_xlim(x_axis[0], x_axis[-1])
    ax.set_ylim(u_min - pad, u_max + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Comparison of Solution Evolutions")
    ax.legend(loc="upper right")
    ax.grid(True)

    def update(frame: int):
        """Update all lines for the current frame."""
        for i, line in enumerate(lines):
            # Ensure we don't index out of bounds if directories have different file counts
            if frame < len(all_solutions[i]):
                line.set_ydata(all_solutions[i][frame])

        time_text.set_text(f"t = {master_times[frame]:.4f}")
        return lines + [time_text]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval,
        blit=True,
        repeat=repeat,
    )

    plt.tight_layout()
    plt.show()
    return ani


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
    h: float, max_velocity: float, viscosity: float, do_round_down: bool = True
) -> float:
    """CFL-based time step: minimum of convective and diffusive limits."""
    dt = min(h / max_velocity, h**2 / viscosity)
    return round_down(dt, 4) if do_round_down else dt


def implicit_euler_first_order(field: NDArray | float, h: float) -> NDArray:
    """Approximate the derivative of a field with step-size h."""
    return (np.roll(field, -1) - np.roll(field, 1)) / (2 * h)


def run_config(
    configuration: dict,
    save_path: Path | str | None = None,
    return_directory: bool = True,
) -> str | None:
    """Run a config and return (absolute solver_data path, relative run folder name)."""
    configuration["save_path"] = save_path
    solver = Burgers(configuration=configuration)
    solver.print_configuration()
    solver.run_simulation()
    solver.post_logging()
    if save_path is not None and return_directory:
        return solver.save_path_dir

    return None


def read_data(
    directory: str | Path,
    final_only: bool = False,
) -> (
    tuple[NDArray, list[float], list[NDArray], list[NDArray]] | tuple[NDArray, NDArray]
):
    """Read chronologically sorted DNS snapshots from *directory*.

    Returns:
    -------
    x:
        Spatial coordinate array (N_dns,).
    times:
        Sorted list of snapshot times.
    solutions:
        Corresponding list of solution arrays, each of shape (N_dns,).
    """
    directory = Path(directory)
    files = sorted(directory.glob("sol_t*.csv"))

    if not files:
        raise FileNotFoundError(f"No sol_t*.csv files found in {directory}")

    times, solutions, forcings = [], [], []
    x = None

    for file in files:
        time = float(file.stem.split("t")[-1])
        times.append(time)

        data = np.loadtxt(file, delimiter=",", skiprows=1)
        if x is None:
            x = data[:, 1]
        solutions.append(data[:, 2])

        # PLACEHOLDER: Assuming column 3 will be forcing
        # For now, we create a dummy array of zeros matching the DNS shape
        try:
            forcings.append(data[:, 3])
        except IndexError:
            forcings.append(np.zeros_like(data[:, 2]))

    times, solutions = zip(*sorted(zip(times, solutions)))

    if final_only:
        solutions = list(solutions)
        return solutions[-1], x

    return x, list(times), list(solutions), list(forcings)


@dataclass
class SolutionConfig:
    """Configuration for a single solution to plot."""

    data_path: Path | str
    label: str
    color: str
    linestyle: str = "-."
    marker: str = "o"
    alpha: float = 0.8
    mesh: Optional[object] = field(default=None, repr=False)
    solution: Optional[object] = field(default=None, repr=False)


def plot_solution_comparison(
    configs: list[SolutionConfig],
    output_path: Path,
    title: str = "Comparison of DNS and LES Solutions",
    xlabel: str = "Spatial Domain",
    ylabel: str = "Solution Value",
    figsize: tuple = (10, 6),
    filename: str = "comparison_solvers.png",
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot and save a comparison of multiple solver solutions.

    Parameters
    ----------
    configs : list[SolutionConfig]
        Each entry holds the path, label, style, and optionally pre-loaded
        mesh/solution arrays for one curve.
    output_path : Path
        Directory (or file path) where the figure is saved.
    title, xlabel, ylabel : str
        Axis annotations.
    figsize : tuple
        Matplotlib figure size.
    filename : str
        Output filename (used when output_path is a directory).
    dpi : int
        Resolution of the saved figure.

    Returns
    -------
    fig, ax : the Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for cfg in configs:
        mesh = cfg.mesh
        solution = cfg.solution

        if mesh is None or solution is None:
            solution, mesh = read_data(directory=cfg.data_path, final_only=True)

        label = f"{cfg.label} ({len(mesh)})"

        ax.plot(
            mesh,
            solution,
            label=label,
            color=cfg.color,
            linestyle=cfg.linestyle,
            marker=cfg.marker,
            alpha=cfg.alpha,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()

    save_path = output_path / filename if output_path.is_dir() else output_path
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return fig, ax
