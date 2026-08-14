"""Utility functions for plotting related aspects."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from matplotlib import animation, pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from problems_and_configurations.disc_config import DiscretizationConfig

from utils.io_utils import read_data

import re

from utils.pipeline_utils import RunPaths


@dataclass
class VelocityPlotConfig:
    """Style and data config for one solution curve in a comparison plot."""

    data_path: Path
    label: str
    color: str
    linestyle: str = "--"
    marker: str = "o"
    alpha: float = 1
    mesh: Optional[NDArray] = field(default=None, repr=False)
    solution: Optional[NDArray] = field(default=None, repr=False)


# utils/velocity_profiles.py


def build_plot_configs(
    paths: RunPaths,
    disc_cfg: DiscretizationConfig,
    dns_solution: NDArray,
    projected_solution: NDArray,
    les_sgsp_data_path: Path,
    extra_configs: list[VelocityPlotConfig] | None = None,
) -> list[VelocityPlotConfig]:
    """Build standard solution plot configs, with optional extra configs appended.

    The five base configs (DNS, LES-A, LES-no-model, projection, SGSP) are always
    included. Any caller-specific configs (e.g. AVC variants) are passed via
    ``extra_configs`` and appended at the end.
    """
    base_configs: list[VelocityPlotConfig] = [
        VelocityPlotConfig(
            data_path=paths.dns_data,
            label="DNS",
            color="gray",
            linestyle="-",
            marker="",
            alpha=0.7,
            mesh=disc_cfg.mesh_dns,
            solution=dns_solution,
        ),
        VelocityPlotConfig(
            data_path=paths.dns_data,
            label="LES - projection",
            color="lightgreen",
            marker="x",
            mesh=disc_cfg.mesh_les,
            solution=projected_solution,
        ),
        VelocityPlotConfig(
            data_path=paths.les_a_data,
            label="LES - A",
            color="tab:orange",
            marker="^",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=paths.les_shakib_one_data,
            label="LES - Shakib 1",
            color="royalblue",
            marker="d",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=paths.les_nm_data,
            label="LES - no model",
            color="gold",
            marker=".",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=les_sgsp_data_path,
            label="LES - SGSP",
            color="crimson",
            marker="d",
            mesh=disc_cfg.mesh_les,
        ),
    ]
    return base_configs + (extra_configs or [])


def create_velocity_plot_configs(
    paths: RunPaths, disc_cfg: DiscretizationConfig
) -> list[VelocityPlotConfig]:
    """Create configs for the velocity profile comparison plot."""
    reference_time = _infer_final_time_from_directory(paths.sgsp_data)

    if paths.dns_data is None:
        raise ValueError("No viable DNS path found! Cannot plot velocity profile.")
    dns_solution = (
        _read_snapshot_at_time(paths.dns_data, reference_time)
        if reference_time is not None
        else read_data(directory=paths.dns_data, final_only=True)[0]
    )
    projected_solution = (
        _read_snapshot_at_time(paths.projection, reference_time)
        if reference_time is not None and paths.projection is not None
        else read_data(directory=paths.projection, final_only=True)[0]
        if paths.projection is not None
        else None
    )
    configs = [
        VelocityPlotConfig(
            data_path=paths.dns_data,
            label="DNS",
            color="gray",
            linestyle="-",
            marker="",
            alpha=0.7,
            mesh=disc_cfg.mesh_dns,
            solution=dns_solution,
        ),
        VelocityPlotConfig(
            data_path=paths.projection,
            label="LES - projection",
            color="lightgreen",
            marker="x",
            mesh=disc_cfg.mesh_les,
            solution=projected_solution,
        ),
        VelocityPlotConfig(
            data_path=paths.les_nm_data,
            label="LES - No SGS Model",
            color="tab:orange",
            marker="x",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=paths.les_a_data,
            label="LES - A",
            color="gold",
            marker="^",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=paths.les_shakib_one_data,
            label="LES - Shakib 1",
            color="dodgerblue",
            marker="d",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=paths.les_shakib_two_data,
            label="LES - Shakib 2",
            color="royalblue",
            marker="d",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=paths.les_shakib_three_data,
            label="LES - Shakib 3",
            color="darkblue",
            marker="d",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=paths.sgsp_data,
            label="LES - SGSP",
            color="crimson",
            marker="d",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=paths.avc_data / "global",
            label="LES - AVC (global)",
            color="crimson",
            linestyle="--",
            marker="s",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=paths.avc_data / "gl_hybrid",
            label="LES - AVC (GL-hybrid)",
            color="blueviolet",
            linestyle="--",
            marker="v",
            mesh=disc_cfg.mesh_les,
        ),
    ]
    return [config for config in configs if config.data_path.exists()]


def _read_snapshot_at_time(directory: Path, target_time: float) -> NDArray:
    """Load the snapshot from directory whose filename time is closest to target_time."""
    csv_files = list(directory.glob("sol_t*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No sol_t*.csv files found in {directory}")

    def _parse_time(p: Path) -> float:
        match = re.search(r"sol_t([\d.]+)\.csv", p.name)
        return float(match.group(1)) if match else float("inf")

    closest_file = min(csv_files, key=lambda p: abs(_parse_time(p) - target_time))
    data = np.loadtxt(closest_file, delimiter=",", skiprows=1)
    return data[:, 2]


def _infer_final_time_from_directory(data_path: Path | None) -> float | None:
    """Parse the highest t-value from sol_t*.csv filenames in data_path."""
    if data_path is None or not data_path.exists():
        return None
    time_values: list[float] = []
    for csv_file in data_path.glob("sol_t*.csv"):
        match = re.search(r"sol_t([\d.]+)\.csv", csv_file.name)
        if match:
            time_values.append(float(match.group(1)))
    return max(time_values) if time_values else None


# Labels for which a final-time annotation is appended.
_TIME_ANNOTATED_LABEL_SUBSTRINGS: frozenset[str] = frozenset(
    {"LES - A", "LES - SGSP", "LES - AVC", "LES - fm-AVC"}
)


def _build_legend_label(base_label: str, final_time: float | None) -> str:
    """Append (t=…) to labels that benefit from a final-time annotation."""
    should_annotate = any(
        substring in base_label for substring in _TIME_ANNOTATED_LABEL_SUBSTRINGS
    )
    if should_annotate and final_time is not None:
        return f"{base_label} (t={final_time:.4f})"
    return base_label


def plot_solution_comparison(
    configs: list[VelocityPlotConfig],
    output_path: Path,
    title: str = "Comparison of DNS and LES Solutions",
    xlabel: str = "Spatial Domain",
    ylabel: str = "Solution Value",
    figsize: tuple = (12, 6),
    filename: str = "comparison_solvers.png",
    dpi: int = 150,
    show_fig: bool = False,
) -> None:
    """Plot and save a comparison of multiple solver solutions."""
    fig, ax = plt.subplots(figsize=figsize)

    # node_count → first cfg label at that size, for subtitle construction.
    node_count_labels: dict[int, str] = {}

    for cfg in configs:
        mesh = cfg.mesh
        solution = cfg.solution
        if mesh is None or solution is None:
            solution, mesh = read_data(directory=cfg.data_path, final_only=True)

        node_count_labels.setdefault(len(mesh), cfg.label)

        ax.plot(
            mesh,
            solution,
            label=cfg.label,
            color=cfg.color,
            linestyle=cfg.linestyle,
            marker=cfg.marker,
            alpha=cfg.alpha,
        )

    subtitle_parts = [
        f"{'DNS' if count == max(node_count_labels) else 'LES'}: {count} nodes"
        for count in sorted(node_count_labels, reverse=True)
    ]
    full_title = f"{title}\n{'  |  '.join(subtitle_parts)}"

    ax.set_title(full_title, linespacing=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    save_path = output_path / filename if output_path.is_dir() else output_path
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show_fig:
        plt.show()
    else:
        plt.close(fig)


def plot_solutions_from_directory_animated(
    directory: str | Path,
    interval: int = 100,
    repeat: bool = True,
) -> FuncAnimation:
    """Animate all solution CSVs in a directory over time."""
    directory = Path(directory)
    files = sorted(directory.glob("sol_t*.csv"))
    if not files:
        raise FileNotFoundError(f"No sol_t*.csv files found in {directory}")

    times, solutions, mesh = [], [], None
    for file_path in files:
        times.append(float(file_path.stem.split("t")[-1]))
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        if mesh is None:
            mesh = data[:, 1]
        solutions.append(data[:, 2])

    times, solutions = zip(*sorted(zip(times, solutions)))
    solutions = list(solutions)

    u_min = min(u.min() for u in solutions)
    u_max = max(u.max() for u in solutions)
    pad = 0.05 * (u_max - u_min)

    fig, ax = plt.subplots(figsize=(10, 6))
    (line,) = ax.plot(mesh, solutions[0], lw=2, color="royalblue")
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment="top"
    )
    ax.set_xlim(mesh[0], mesh[-1])
    ax.set_ylim(u_min - pad, u_max + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Solution evolution")
    ax.grid(True)

    def update(frame: int) -> tuple[Any, Any]:
        """Update animation frame."""
        line.set_ydata(solutions[frame])
        time_text.set_text(f"t = {times[frame]:.4f}")
        return line, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(times), interval=interval, blit=True, repeat=repeat
    )
    plt.tight_layout()
    plt.show()
    return ani


def is_viable_solution_path(data_path: Path | str | None) -> bool:
    """True if the path exists and contains at least one solution CSV."""
    if data_path is None:
        return False
    path = Path(data_path)
    return path.exists() and any(path.glob("sol_t*.csv"))


RESULTS_ROOT = Path(
    "solver_data/LES_SGSP"
)  # <- where sol_t*.csv output dirs live, per run_id
PROJECTION_ROOT = Path(
    "training_data/pre_split"
)  # <- where solutions_projection.npy / times.npy live, per run_id
ROOT = Path(r"/")


def plot_solutions_from_run_id_animated(
    run_id: str,
    interval: int = 100,
    repeat: bool = True,
) -> FuncAnimation:
    """Animate SGSP solution vs. the projected (filtered-DNS) reference for run_id."""
    sgsp_directory = ROOT / run_id / RESULTS_ROOT
    projection_directory = ROOT / run_id / PROJECTION_ROOT

    # --- SGSP solution (sol_t*.csv) ---
    sgsp_files = sorted(sgsp_directory.glob("sol_t*.csv"))
    if not sgsp_files:
        raise FileNotFoundError(f"No sol_t*.csv files found in {sgsp_directory}")

    sgsp_times_list: list[float] = []
    sgsp_solutions_list: list[NDArray] = []
    mesh: NDArray | None = None
    for file_path in sgsp_files:
        sgsp_times_list.append(float(file_path.stem.split("t")[-1]))
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        if mesh is None:
            mesh = data[:, 1]
        sgsp_solutions_list.append(data[:, 2])

    sgsp_times_list, sgsp_solutions_list = zip(
        *sorted(zip(sgsp_times_list, sgsp_solutions_list))
    )
    sgsp_times = np.array(sgsp_times_list)
    sgsp_solutions = list(sgsp_solutions_list)

    # --- Projected (filtered-DNS) reference ---
    projection_times = np.load(projection_directory / "times.npy")
    projection_solutions = np.load(projection_directory / "solutions_projection.npy")

    # Match each SGSP frame to the nearest projection snapshot in time
    projection_indices = np.searchsorted(projection_times, sgsp_times)
    projection_indices = np.clip(projection_indices, 0, len(projection_times) - 1)

    n_frames = len(sgsp_times)
    all_values = np.concatenate(
        [np.array(sgsp_solutions), projection_solutions[projection_indices]]
    )
    u_min, u_max = float(all_values.min()), float(all_values.max())
    pad = 0.05 * (u_max - u_min)

    fig, ax = plt.subplots(figsize=(10, 6))
    (sgsp_line,) = ax.plot(
        mesh, sgsp_solutions[0], lw=2, color="royalblue", label="SGSP"
    )
    (projection_line,) = ax.plot(
        mesh,
        projection_solutions[projection_indices[0]],
        lw=2,
        ls="--",
        color="black",
        label="Projected DNS",
    )
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment="top"
    )
    ax.set_xlim(mesh[0], mesh[-1])
    ax.set_ylim(u_min - pad, u_max + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"Solution evolution — run {run_id}")
    ax.legend()
    ax.grid(True)

    def update(frame_idx: int) -> tuple[Any, Any, Any]:
        """Update animation frame."""
        sgsp_line.set_ydata(sgsp_solutions[frame_idx])
        projection_line.set_ydata(projection_solutions[projection_indices[frame_idx]])
        time_text.set_text(f"t = {sgsp_times[frame_idx]:.4f}")
        return sgsp_line, projection_line, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=True, repeat=repeat
    )
    plt.tight_layout()
    plt.show()
    return ani
