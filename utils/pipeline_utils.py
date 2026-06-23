"""Utility functions for pipeline related aspects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from constants import DNS_FOLDER
from dns_caching import (
    DNSCacheKey,
    resolve_dns_cache,
    DNSCacheStatus,
    extend_dns_run,
    write_dns_parameters,
)
from solvers.sgsp_training_data_generator import BurgersDataGenerator
from ml.data_assembly.a_priori_verification_stash import run_apriori_verification
from ml.ml_agents.predictor import (
    plot_training_diagnostics,
    evaluate_on_val_set,
    train_predictor,
)
from ml.ml_agents.solver_configs import SGSPConfig
from solvers.burgers_sgsp import BurgersSGSP

if TYPE_CHECKING:
    from problems_and_configurations.disc_config import DiscretisationConfig
    from problems_and_configurations.problems import Problem


def resolve_dns_caching(
    cache_root: Path, problem: Problem, disc_cfg: DiscretisationConfig, paths
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
            problem,
            disc_cfg,
            cache_dir,
            paths.dns_data,
            paths.projection,
            paths.training,
        )
        write_dns_parameters(cache_dir, dns_cache_key, problem.domain_timespan)


def run_data_generator(
    problem: Problem,
    disc_cfg: DiscretisationConfig,
    master_path: Path,
    dns_save_path: Path,
    projection_data_path: Path,
    sgsp_data_training_path: Path,
    t_start: float = 0.0,
    append_mode: bool = False,
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
        append_mode=append_mode,
    )
    solver.print_configuration()
    solver.run_simulation()
    solver.post_processing()


def run_sgsp_training(
    data_path: Path,
    output_dir: Path,
    domain_length: float,
    n_elements: int,
) -> None:
    """Train SGSP predictor and run a priori verification."""
    model, training_stats = train_predictor(
        data_path=data_path,
        output_dir=output_dir,
    )
    plot_training_diagnostics(
        training_stats=training_stats,
        output_dir=output_dir,
        show_fig=False,
    )
    evaluate_on_val_set(
        model=model,
        data_path=data_path,
        output_dir=output_dir,
    )
    run_apriori_verification(
        model=model,
        data_dir=data_path,
        output_dir=output_dir,
        domain_length=domain_length,
        n_elements=n_elements,
    )


def run_sgsp_coupled_solver(
    problem: Problem,
    disc_cfg: DiscretisationConfig,
    master_path: Path,
    sgsp_cfg: SGSPConfig,
) -> None:
    """Run the LES solver with ANN-predicted SGS closure."""
    solver = BurgersSGSP(
        problem,
        disc_cfg,
        "sgsp",
        master_path,
        sgsp_cfg,
    )
    solver.print_configuration()
    solver.run_simulation()
    solver.post_processing()
