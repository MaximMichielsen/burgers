"""Utility functions for plotting related aspects."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from numpy.typing import NDArray

from setup.config_discretization import DiscretizationConfig
from utils.io_utils import read_data
from utils.pipeline_utils import RunPaths


@dataclass(frozen=True)
class PlotConfig:
    """Style and source metadata for energy and dissipation plots."""

    label: str
    data_path: Path | None
    color: str
    linestyle: str = "-"
    linewidth: float = 1.4
    alpha: float = 1.0


def plotting_configs(paths: RunPaths) -> list[PlotConfig]:
    """Standard plotting configurations for energy and dissipation plots."""
    configs: list[PlotConfig] = [
        PlotConfig("DNS", paths.dns_data, "gray", linestyle="-", linewidth=1.8),
        PlotConfig(
            "Projection", paths.projection, "lightgreen", linestyle="-", linewidth=1.2
        ),
        PlotConfig(
            "LES - No Model", paths.les_nm, "gold", linestyle="-.", linewidth=1.4
        ),
        PlotConfig(
            "LES - Tau 2-Param",
            paths.les_two,
            "dodgerblue",
            linestyle="--",
            linewidth=1.4,
        ),
        PlotConfig(
            "LES - Tau 3-Param",
            paths.les_three,
            "royalblue",
            linestyle="--",
            linewidth=1.4,
        ),
        PlotConfig(
            "LES - Tau 4-Param",
            paths.les_four,
            "darkblue",
            linestyle="--",
            linewidth=1.4,
        ),
        PlotConfig("LES - ANN", paths.ann_data, "purple", linestyle="-", linewidth=1.8),
    ]

    # Only return configurations where the path is defined and exists on disk
    return [
        cfg for cfg in configs if cfg.data_path is not None and cfg.data_path.exists()
    ]


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


def create_velocity_plot_configs(
    paths: RunPaths,
    disc_cfg: DiscretizationConfig,
    extra_configs: list[VelocityPlotConfig] | None = None,
) -> list[VelocityPlotConfig]:
    """Create configurations for velocity profile comparison plots.

    Loads DNS and projection solutions if paths are available, constructs
    the standard set of LES comparison configs, and appends any optional
    extra_configs. Configurations with non-existent data paths are filtered out.
    """
    if paths.dns_data is None or not paths.dns_data.exists():
        raise ValueError("No viable DNS path found! Cannot plot velocity profile.")

    # Safely load reference solutions
    dns_solution = read_data(directory=paths.dns_data, final_only=True)[0]
    projected_solution = (
        read_data(directory=paths.projection, final_only=True)[0]
        if paths.projection is not None and paths.projection.exists()
        else None
    )

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
            data_path=paths.projection,
            label="LES - projection",
            color="lightgreen",
            marker="x",
            mesh=disc_cfg.mesh_les,
            solution=projected_solution,
        ),
        VelocityPlotConfig(
            data_path=paths.les_nm,
            label="LES - No SGS Model",
            color="gold",
            marker=".",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=paths.les_two,
            label="LES - Tau 2-Param",
            color="dodgerblue",
            marker="d",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=paths.les_three,
            label="LES - Tau 3-Param",
            color="royalblue",
            marker="d",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=paths.les_four,
            label="LES - Tau 4-Param",
            color="darkblue",
            marker="d",
            mesh=disc_cfg.mesh_les,
        ),
        VelocityPlotConfig(
            data_path=paths.ann_data,
            label="LES - ANN Coupled",
            color="crimson",
            linestyle="--",
            marker="s",
            mesh=disc_cfg.mesh_les,
        ),
    ]

    all_configs = base_configs + (extra_configs or [])

    # Keep configs that have explicitly attached solutions OR existing directory paths
    return [
        config
        for config in all_configs
        if config.solution is not None
        or (config.data_path is not None and config.data_path.exists())
    ]
