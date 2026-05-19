from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from matplotlib import animation, pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from solvers.burgers_pure import BurgersPure

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


def run_config(config: dict) -> None:
    """Run a given configuration file."""
    solver = BurgersPure(config)
    solver.print_configuration()
    solver.run_simulation()
    solver.post_processing()


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

    power = np.log2(required_points)
    ceiled_power = np.ceil(power)
    return int(2**ceiled_power)


def round_down(value: float, decimals: int) -> float:
    """Round a float down to the specified decimal."""
    factor = 10**decimals
    return np.floor(value * factor) / factor


def compute_time_step(
    h: float, max_velocity: float, viscosity: float, do_round_down: bool = True
) -> float:
    """CFL-based time step: minimum of convective and diffusive limits."""
    if max_velocity == 0:
        return h**2 / viscosity

    dt = min(h / max_velocity, h**2 / viscosity)
    return round_down(dt, 4) if do_round_down else dt


def implicit_euler_first_order(field: NDArray | float, h: float) -> NDArray:
    """Approximate the derivative of a field with step-size h."""
    return (np.roll(field, -1) - np.roll(field, 1)) / (2 * h)


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
