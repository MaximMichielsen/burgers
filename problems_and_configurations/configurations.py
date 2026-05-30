"""Solver config builders for DNS, LES, and ANN-coupled Burgers simulations."""

from pathlib import Path


from constants import (
    BLOWN_UP_FOLDER,
    LES_ANN_PUSULURI_FOLDER,
    LES_ANN_RAJAMPETA_FOLDER,
    LES_ANN_SAVE_PATH,
    STABLE_FOLDER,
    LES_ANN_UNCLIPPED_FOLDER,
    LES_AVC_SAVE_PATH,
)
from problems_and_configurations.mesh_config import DiscretisationConfig
from problems_and_configurations.problems import Problem
from solvers.burgers_base import BurgersBase as Burgers
from solvers.burgers_sgsp import BurgersSGSP
from utils.io_utils import set_extractions


def create_solver_configs(
    problem_definition: Problem,
    disc_cfg: DiscretisationConfig,
    dns_dir: Path | str | None = None,
    les_a_dir: Path | str | None = None,
    les_nm_dir: Path | str | None = None,
) -> tuple[dict, dict, dict]:
    """Return (config_dns, config_les_analytical, config_les_no_model)."""
    dns_extractions = set_extractions(
        problem_definition.domain_timespan,
        int(round(problem_definition.domain_timespan / disc_cfg.dt_dns)),
        disc_cfg.dt_dns,
    )
    les_extractions = set_extractions(
        problem_definition.domain_timespan,
        int(problem_definition.domain_timespan / disc_cfg.dt_les),
        disc_cfg.dt_les,
    )

    config_dns = Burgers.create_config(
        initial_condition=disc_cfg.initial_solution_dns,
        simulation_mode="dns",
        run_objective="data_generation",
        node_amount=disc_cfg.n_nodes_dns,
        boundary_condition_type=problem_definition.boundary_condition_type,
        boundary_condition_value=problem_definition.boundary_condition_value,
        external_forcing=problem_definition.external_forcing,
        forcing_steady=problem_definition.forcing_steady,
        domain_timespan=problem_definition.domain_timespan,
        time_step=disc_cfg.dt_dns,
        domain_length=problem_definition.domain_length,
        convergence_tol_residual=1e-6,
        convergence_tol_update=1e-6,
        max_iterations=100,
        relaxation=None,
        viscosity=problem_definition.viscosity,
        extract_at_times=dns_extractions,
        master_path=dns_dir,
    )

    shared_les_kwargs: dict = dict(
        initial_condition=disc_cfg.initial_solution_les,
        node_amount=disc_cfg.n_nodes_les,
        boundary_condition_type=problem_definition.boundary_condition_type,
        boundary_condition_value=problem_definition.boundary_condition_value,
        external_forcing=problem_definition.external_forcing,
        forcing_steady=problem_definition.forcing_steady,
        domain_timespan=problem_definition.domain_timespan,
        time_step=disc_cfg.dt_les,
        domain_length=problem_definition.domain_length,
        convergence_tol_residual=1e-3,
        convergence_tol_update=1e-3,
        max_iterations=20,
        relaxation=None,
        viscosity=problem_definition.viscosity,
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


def _resolve_sgsp_output_paths(
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

    base_sgsp_dir = solver_data_dir / LES_ANN_SAVE_PATH / clip_variant_folder
    stable_path = base_sgsp_dir / STABLE_FOLDER
    blown_up_path = base_sgsp_dir / BLOWN_UP_FOLDER

    stable_path.mkdir(parents=True, exist_ok=True)
    blown_up_path.mkdir(parents=True, exist_ok=True)

    return stable_path, blown_up_path


def _resolve_avc_output_paths(solver_data_dir: Path) -> tuple[Path, Path]:
    """Resolve and create (stable_path, blown_up_path) for the AVC solver."""
    base_avc_dir = solver_data_dir / LES_AVC_SAVE_PATH
    stable_path = base_avc_dir / STABLE_FOLDER
    blown_up_path = base_avc_dir / BLOWN_UP_FOLDER
    stable_path.mkdir(parents=True, exist_ok=True)
    blown_up_path.mkdir(parents=True, exist_ok=True)
    return stable_path, blown_up_path


def create_sgsp_config(
    problem_definition: Problem,
    disc_cfg: DiscretisationConfig,
    sgsp_model_path: Path,
    normalisation_stats_path: Path,
    data_dir: Path | str | None = None,
    clip_pusuluri: bool = False,
    clip_rajampeta: bool = False,
    blowup_threshold: float = 1e4,
    blowup_buffer_size: int = 5000,
    sgsp_warmup_steps: int = 2,
) -> tuple[dict, Path, Path]:
    """ANN-coupled LES config built from a DiscretisationConfig.

    Returns (config, stable_path, blown_up_path).
    """
    les_extractions = set_extractions(
        problem_definition.domain_timespan,
        int(problem_definition.domain_timespan / disc_cfg.dt_les),
        disc_cfg.dt_les,
    )
    stable_path, blown_up_path = _resolve_sgsp_output_paths(
        solver_data_dir=Path(data_dir),
        clip_pusuluri=clip_pusuluri,
        clip_rajampeta=clip_rajampeta,
    )
    config = BurgersSGSP.create_sgsp_config(
        sgsp_model_path=sgsp_model_path,
        normalisation_stats_path=normalisation_stats_path,
        sgsp_warmup_steps=sgsp_warmup_steps,
        blowup_threshold=blowup_threshold,
        blowup_buffer_size=blowup_buffer_size,
        blown_up_path=str(blown_up_path),
        initial_condition=disc_cfg.initial_solution_les,
        simulation_mode="sgsp",
        run_objective="data_generation",
        node_amount=disc_cfg.n_nodes_les,
        viscosity=problem_definition.viscosity,
        time_step=disc_cfg.dt_les,
        domain_timespan=problem_definition.domain_timespan,
        domain_length=problem_definition.domain_length,
        boundary_condition_type=problem_definition.boundary_condition_type,
        boundary_condition_value=problem_definition.boundary_condition_value,
        external_forcing=problem_definition.external_forcing,
        forcing_steady=problem_definition.forcing_steady,
        convergence_tol_residual=1e-4,
        convergence_tol_update=1e-4,
        max_iterations=20,
        relaxation=None,
        extract_at_times=les_extractions,
        master_path=stable_path,
    )
    return config, stable_path, blown_up_path
