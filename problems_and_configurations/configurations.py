"""Solver config builders for DNS, LES, and ANN-coupled Burgers simulations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from constants import (
    LES_ANN_BLOWN_UP_FOLDER,
    LES_ANN_PUSULURI_FOLDER,
    LES_ANN_RAJAMPETA_FOLDER,
    LES_ANN_SAVE_PATH,
    LES_ANN_STABLE_FOLDER,
    LES_ANN_UNCLIPPED_FOLDER,
    DNS_TO_LES_RATIO,
)
from functions import set_extractions
from solvers.burgers_coupled import BurgersCoupled
from solvers.burgers_pure import BurgersPure as Burgers


@dataclass
class DiscretisationConfig:
    """Spatial and temporal discretisation parameters derived from LES element count."""

    n_elements_les: int
    temporal_refinement: int
    courant_les: float
    domain_length: float

    def __post_init__(self) -> None:
        self.n_nodes_les: int = self.n_elements_les + 1
        self.n_elements_dns: int = self.n_elements_les * DNS_TO_LES_RATIO
        self.n_nodes_dns: int = self.n_elements_dns + 1
        self.h_les: float = self.domain_length / self.n_elements_les
        self.dt_les: float = self.courant_les * self.h_les
        self.dt_dns: float = self.dt_les / self.temporal_refinement


@dataclass
class MeshConfig:
    """Resolved grid, time-step, and initial condition for one resolution level."""

    n_nodes: int
    mesh: NDArray
    element_size: float
    time_step: float
    initial_solution: NDArray


def build_mesh_config(
    n_nodes: int,
    domain_length: float,
    time_step: float,
    initial_condition_fn: Callable,
) -> MeshConfig:
    """Build a MeshConfig from node count, domain length, time step, and IC function."""
    mesh, element_size = np.linspace(0, domain_length, n_nodes, retstep=True)
    return MeshConfig(
        n_nodes=n_nodes,
        mesh=mesh,
        element_size=float(element_size),
        time_step=time_step,
        initial_solution=initial_condition_fn(mesh),
    )


def create_solver_configs(
    problem_definition: dict,
    dns_mesh: MeshConfig,
    les_mesh: MeshConfig,
    dns_dir: Path | str | None = None,
    les_a_dir: Path | str | None = None,
    les_nm_dir: Path | str | None = None,
) -> tuple[dict, dict, dict]:
    """Return (config_dns, config_les_analytical, config_les_no_model)."""
    duration: float = problem_definition["domain_timespan"]
    domain_length: float = problem_definition["domain_length"]
    bc_type = problem_definition["boundary_condition_type"]
    bc_value = problem_definition["boundary_condition_value"]
    forcing = problem_definition["external_forcing"]
    forcing_is_steady: bool = problem_definition["forcing_steady"]
    viscosity: float = problem_definition["viscosity"]

    n_dns_steps = int(round(duration / dns_mesh.time_step))
    dns_extractions = set_extractions(duration, n_dns_steps, dns_mesh.time_step)
    les_extractions = set_extractions(
        duration, int(duration / les_mesh.time_step), les_mesh.time_step
    )

    config_dns = Burgers.create_config(
        initial_condition=dns_mesh.initial_solution,
        simulation_mode="dns",
        run_objective="data_generation",
        node_amount=dns_mesh.n_nodes,
        boundary_condition_type=bc_type,
        boundary_condition_value=bc_value,
        external_forcing=forcing,
        forcing_steady=forcing_is_steady,
        domain_timespan=duration,
        time_step=dns_mesh.time_step,
        domain_length=domain_length,
        convergence_tol_residual=1e-6,
        convergence_tol_update=1e-6,
        max_iterations=100,
        relaxation=None,
        viscosity=viscosity,
        extract_at_times=dns_extractions,
        master_path=dns_dir,
    )

    shared_les_kwargs: dict = dict(
        initial_condition=les_mesh.initial_solution,
        node_amount=les_mesh.n_nodes,
        boundary_condition_type=bc_type,
        boundary_condition_value=bc_value,
        external_forcing=forcing,
        forcing_steady=forcing_is_steady,
        domain_timespan=duration,
        time_step=les_mesh.time_step,
        domain_length=domain_length,
        convergence_tol_residual=1e-4,
        convergence_tol_update=1e-4,
        max_iterations=20,
        relaxation=None,
        viscosity=viscosity,
        extract_at_times=les_extractions,
        run_objective="data_generation",
    )
    config_les_analytical = Burgers.create_config(
        **shared_les_kwargs, simulation_mode="les", master_path=les_a_dir
    )
    config_les_no_model = Burgers.create_config(
        **shared_les_kwargs, simulation_mode="no_model", master_path=les_nm_dir
    )
    return config_dns, config_les_analytical, config_les_no_model


def _resolve_ann_output_paths(
    solver_data_dir: Path,
    clip_pusuluri: bool,
    clip_rajampeta: bool,
) -> tuple[Path, Path]:
    """Resolve and create (stable_path, blown_up_path) for the clipping variant."""
    if clip_rajampeta and not clip_pusuluri:
        raise ValueError("clip_rajampeta requires clip_pusuluri to be enabled.")

    if clip_rajampeta:
        clip_variant_folder = LES_ANN_RAJAMPETA_FOLDER
    elif clip_pusuluri:
        clip_variant_folder = LES_ANN_PUSULURI_FOLDER
    else:
        clip_variant_folder = LES_ANN_UNCLIPPED_FOLDER

    base_ann_dir = solver_data_dir / LES_ANN_SAVE_PATH / clip_variant_folder
    stable_path = base_ann_dir / LES_ANN_STABLE_FOLDER
    blown_up_path = base_ann_dir / LES_ANN_BLOWN_UP_FOLDER

    stable_path.mkdir(parents=True, exist_ok=True)
    blown_up_path.mkdir(parents=True, exist_ok=True)

    return stable_path, blown_up_path


def create_ann_config(
    problem_definition: dict,
    les_mesh: MeshConfig,
    ann_model_path: Path,
    normalisation_stats_path: Path,
    les_ann_dir: Path | str | None = None,
    clip_pusuluri: bool = False,
    clip_rajampeta: bool = False,
    blowup_threshold: float = 1e4,
    blowup_buffer_size: int = 5_000,
    ann_warmup_steps: int = 2,
) -> tuple[dict, Path, Path]:
    """ANN-coupled LES config built from a pre-resolved MeshConfig.

    Returns (config, stable_path, blown_up_path).
    """
    duration: float = problem_definition["domain_timespan"]
    viscosity: float = problem_definition["viscosity"]

    les_extractions = set_extractions(
        duration, int(duration / les_mesh.time_step), les_mesh.time_step
    )
    stable_path, blown_up_path = _resolve_ann_output_paths(
        solver_data_dir=Path(les_ann_dir),
        clip_pusuluri=clip_pusuluri,
        clip_rajampeta=clip_rajampeta,
    )
    config = BurgersCoupled.create_coupled_config(
        ann_model_path=ann_model_path,
        normalisation_stats_path=normalisation_stats_path,
        ann_warmup_steps=ann_warmup_steps,
        blowup_threshold=blowup_threshold,
        blowup_buffer_size=blowup_buffer_size,
        blown_up_path=str(blown_up_path),
        initial_condition=les_mesh.initial_solution,
        simulation_mode="ann",
        run_objective="data_generation",
        node_amount=les_mesh.n_nodes,
        viscosity=viscosity,
        time_step=les_mesh.time_step,
        domain_timespan=duration,
        domain_length=problem_definition["domain_length"],
        boundary_condition_type=problem_definition["boundary_condition_type"],
        boundary_condition_value=problem_definition["boundary_condition_value"],
        external_forcing=problem_definition["external_forcing"],
        forcing_steady=problem_definition["forcing_steady"],
        convergence_tol_residual=1e-4,
        convergence_tol_update=1e-4,
        max_iterations=20,
        relaxation=None,
        extract_at_times=les_extractions,
        master_path=stable_path,
    )
    return config, stable_path, blown_up_path
