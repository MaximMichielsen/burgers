"""Utility functions for plotting related aspects."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from matplotlib import animation, pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray
from packaging import markers

from pipeline_settings import RunPaths
from problems_and_configurations.mesh_config import MeshConfig
from utils.io_utils import read_data

import re


@dataclass
class SolutionConfig:
    """Style and data config for one solution curve in a comparison plot."""

    data_path: Path | str
    label: str
    color: str
    linestyle: str = "-."
    marker: str = "o"
    alpha: float = 0.8
    mesh: Optional[NDArray] = field(default=None, repr=False)
    solution: Optional[NDArray] = field(default=None, repr=False)


def build_plot_configs(
    paths: RunPaths,
    dns_mesh: MeshConfig,
    les_mesh: MeshConfig,
    dns_solution: NDArray,
    projected_solution: NDArray,
    les_sgsp_data_path: Path,
    les_avc_data_path: Path,
    les_avc_fixed_mean_path: Path,
) -> list[SolutionConfig]:
    """Build the five standard solution plot configs for a pipeline run."""
    return [
        SolutionConfig(
            data_path=paths.dns_data,
            label="DNS",
            color="gray",
            linestyle="-",
            marker="",
            alpha=0.7,
            mesh=dns_mesh.mesh,
            solution=dns_solution,
        ),
        SolutionConfig(
            data_path=paths.les_a_data,
            label="LES - A",
            color="royalblue",
            marker="x",
            mesh=les_mesh.mesh,
        ),
        SolutionConfig(
            data_path=paths.les_nm_data,
            label="LES - no model",
            color="tab:orange",
            marker=".",
            mesh=les_mesh.mesh,
        ),
        SolutionConfig(
            data_path=paths.dns_data,
            label="LES - projection",
            color="lightgreen",
            marker="^",
            mesh=les_mesh.mesh,
            solution=projected_solution,
        ),
        SolutionConfig(
            data_path=les_sgsp_data_path,
            label="LES - SGSP",
            color="salmon",
            marker="d",
            mesh=les_mesh.mesh,
        ),
        SolutionConfig(
            data_path=les_avc_data_path,
            label="LES - AVC",
            color="mediumorchid",
            marker="*",
            mesh=les_mesh.mesh,
        ),
        SolutionConfig(
            data_path=les_avc_fixed_mean_path,
            label="LES - fm-AVC",
            color="plum",
            linestyle="--",
            marker="*",
            mesh=les_mesh.mesh,
            alpha=0.6,
        ),
    ]


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
    configs: list[SolutionConfig],
    output_path: Path,
    title: str = "Comparison of DNS and LES Solutions",
    xlabel: str = "Spatial Domain",
    ylabel: str = "Solution Value",
    figsize: tuple = (10, 6),
    filename: str = "comparison_solvers.png",
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes]:
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

        final_time = _infer_final_time_from_directory(cfg.data_path)
        legend_label = _build_legend_label(cfg.label, final_time)

        should_annotate = any(
            substring in cfg.label for substring in _TIME_ANNOTATED_LABEL_SUBSTRINGS
        )
        legend_label = (
            f"{cfg.label} (t={final_time:.4f})"
            if should_annotate and final_time is not None
            else cfg.label
        )

        ax.plot(
            mesh,
            solution,
            label=legend_label,
            color=cfg.color,
            linestyle=cfg.linestyle,
            marker=cfg.marker,
            alpha=cfg.alpha,
        )

    # Subtitle: "DNS: 1025 nodes  |  LES: 17 nodes" (sorted large → small).
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
    plt.show()
    return fig, ax


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
