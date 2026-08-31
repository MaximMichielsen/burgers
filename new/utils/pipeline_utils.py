"""Utility functions for pipeline related aspects."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from new.solvers.solver_projection import BurgersDataGenerator
from new.utils.dns_caching import (
    DNSCacheKey,
    resolve_dns_cache,
    DNSCacheStatus,
    extend_dns_run,
    write_dns_parameters,
)
from old.constants import (
    DNS_FOLDER,
    RUNS_FOLDER,
    LES_SHAKIB_ONE_SAVE_PATH,
    LES_SHAKIB_TWO_SAVE_PATH,
    LES_SHAKIB_THREE_SAVE_PATH,
)

from dataclasses import dataclass

from old.constants import (
    SOLVER_DATA_FOLDER,
    LES_ANALYTICAL_SAVE_PATH,
    LES_NO_MODEL_SAVE_PATH,
    AGENT_FOLDER,
    A_PRIORI_FOLDER,
    LES_SGSP_SAVE_PATH,
    LES_AVC_SAVE_PATH,
)


if TYPE_CHECKING:
    from old.problems_and_configurations.disc_config import DiscretizationConfig
    from old.problems_and_configurations.problems import Problem


def get_run_id(problem_name: str) -> str:
    """Generate a timestamped run ID."""
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    return f"run_{problem_name}_{timestamp}"


def resolve_pathing(problem_name: str, root_directory: Path) -> RunPaths:
    """Create pipeline paths and directories."""
    master_path = root_directory / RUNS_FOLDER / get_run_id(problem_name)
    paths = RunPaths.from_master(master_path)
    paths.create_master()
    return paths


def load_manual_models(
    paths: RunPaths, sgsp_path: str = "", training_path: str = "", avcg_path: str = ""
) -> None:
    """Manually set paths to the SGSP and/or AVC models."""
    paths.sgsp_model = Path(sgsp_path) if sgsp_path != "" else paths.sgsp_model
    paths.training = Path(training_path) if training_path != "" else paths.training
    paths.avc_gg_model = Path(avcg_path) if avcg_path != "" else paths.avc_gg_model


def run_dns(
    cache_root: Path, problem: Problem, disc_cfg: DiscretizationConfig, paths
) -> None:
    """Check DNS caches for existing data."""
    dns_cache_key = DNSCacheKey(
        problem_name=problem.name,
        domain_length=problem.domain_length,
        viscosity=problem.viscosity,
        forcing_name=problem.forcing.__name__,
        bc_type=problem.boundary_condition_type,
        bc_value=problem.boundary_condition_value,
        n_nodes_dns=disc_cfg.n_nodes_dns,
        temporal_refinement=disc_cfg.temporal_refinement,
        courant_les=disc_cfg.courant_les,
    )

    cache_result = resolve_dns_cache(cache_root, dns_cache_key, problem.domain_timespan)

    if cache_result.status == DNSCacheStatus.HIT:
        print(f"[DNS cache] HIT — reusing {cache_result.cache_dir}")
        paths.dns_data = cache_result.cache_dir / DNS_FOLDER
        paths.projection = (
            cache_result.cache_dir / f"projection_{disc_cfg.n_nodes_les}nodes"
        )
        paths.training = (
            cache_result.cache_dir / f"training_{disc_cfg.n_nodes_les}nodes"
        )

    elif cache_result.status == DNSCacheStatus.HIT_SHORT:
        print(
            f"[DNS cache] HIT_SHORT — extending from t={cache_result.cached_timespan:.4f}"
        )
        projection_dir = (
            cache_result.cache_dir / f"projection_{disc_cfg.n_nodes_les}nodes"
        )
        training_dir = cache_result.cache_dir / f"training_{disc_cfg.n_nodes_les}nodes"

        extend_dns_run(
            cache_dir=cache_result.cache_dir,
            cache_result=cache_result,
            problem=problem,
            disc_cfg=disc_cfg,
            requested_timespan=problem.domain_timespan,
            projection_dir=projection_dir,
            training_dir=training_dir,
            run_data_generator_fn=run_data_generator,
        )
        paths.dns_data = cache_result.cache_dir / DNS_FOLDER
        paths.projection = projection_dir
        paths.training = training_dir

    else:
        print("[DNS cache] MISS — running DNS")
        cache_dir = cache_root / dns_cache_key.dir_to_name()
        paths.dns_data = cache_dir / DNS_FOLDER
        paths.projection = cache_dir / f"projection_{disc_cfg.n_nodes_les}nodes"
        paths.training = cache_dir / f"training_{disc_cfg.n_nodes_les}nodes"
        paths.projection.mkdir(parents=True, exist_ok=True)
        paths.training.mkdir(parents=True, exist_ok=True)
        run_data_generator(
            problem=problem,
            disc_cfg=disc_cfg,
            master_path=cache_dir,
            dns_save_path=paths.dns_data,
            projection_data_path=paths.projection,
        )
        write_dns_parameters(cache_dir, dns_cache_key, problem.domain_timespan)


def run_data_generator(
    problem: Problem,
    disc_cfg: DiscretizationConfig,
    master_path: Path,
    dns_save_path: Path,
    projection_data_path: Path,
    t_start: float = 0.0,
    append_mode: bool = False,
) -> None:
    """Run DNS and assemble SGSP training data."""
    solver = BurgersDataGenerator(
        problem=problem,
        disc_cfg=disc_cfg,
        simulation_mode="dns",
        master_path=master_path,
        dns_save_path=dns_save_path,
        projection_save_path=projection_data_path,
        t_start=t_start,
        append_mode=append_mode,
    )
    solver.print_configuration()
    solver.run_simulation()
    solver.post_processing()


@dataclass
class RunPaths:
    """All output directories for a single pipeline run."""

    master: Path
    solver_data: Path
    dns_data: Path | None
    les_a_data: Path
    les_shakib_one_data: Path
    les_shakib_two_data: Path
    les_shakib_three_data: Path
    les_nm_data: Path
    avc_data: Path
    avc_gg_data: Path
    avc_gl_data: Path
    projection: Path | None
    training: Path | None
    agents: Path
    avc_gg_model: Path
    avc_gl_model: Path
    apriori: Path

    @classmethod
    def from_master(cls, master_path: Path) -> "RunPaths":
        """Derive all subdirectories from the master run path."""
        return cls(
            master=master_path,
            solver_data=master_path / SOLVER_DATA_FOLDER,
            dns_data=None,
            les_a_data=master_path / SOLVER_DATA_FOLDER / LES_ANALYTICAL_SAVE_PATH,
            les_shakib_one_data=master_path
            / SOLVER_DATA_FOLDER
            / LES_SHAKIB_ONE_SAVE_PATH,
            les_shakib_two_data=master_path
            / SOLVER_DATA_FOLDER
            / LES_SHAKIB_TWO_SAVE_PATH,
            les_shakib_three_data=master_path
            / SOLVER_DATA_FOLDER
            / LES_SHAKIB_THREE_SAVE_PATH,
            les_nm_data=master_path / SOLVER_DATA_FOLDER / LES_NO_MODEL_SAVE_PATH,
            avc_gg_data=master_path / SOLVER_DATA_FOLDER / LES_AVC_SAVE_PATH / "global",
            avc_gl_data=master_path
            / SOLVER_DATA_FOLDER
            / LES_AVC_SAVE_PATH
            / "global_local",
            projection=None,
            training=None,
            agents=master_path / AGENT_FOLDER,
            avc_gg_model=master_path / AGENT_FOLDER / "av_global_corrector.pt",
            avc_gl_model=master_path / AGENT_FOLDER / "av_gl_hybrid.pt",
            apriori=master_path / A_PRIORI_FOLDER,
            avc_data=master_path / SOLVER_DATA_FOLDER / LES_AVC_SAVE_PATH,
        )

    def create_master(self) -> None:
        """Create the master directory; subdirectories are created on demand by each step."""
        self.master.mkdir(parents=True, exist_ok=True)
