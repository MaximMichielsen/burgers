import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable

from constants import DNS_FOLDER
from problems_and_configurations.disc_config import DiscretizationConfig
from problems_and_configurations.problems import Problem
from solvers.implicit.sgsp_training_data_generator import BurgersDataGenerator
from utils.io_utils import read_data


@dataclass(frozen=True)
class DNSCacheKey:
    """Uniquely define a DNS run by its simulation parameters."""

    problem_name: str
    domain_length: float
    viscosity: float
    forcing_name: str
    bc_type: str
    bc_value: float | int | tuple[float | int, float | int] | None
    n_nodes_dns: int
    temporal_refinement: int
    courant_les: float

    def dir_to_name(self) -> str:
        """Short deterministic hash used as cache directory name."""
        key_str = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return "dns_" + hashlib.sha1(key_str.encode()).hexdigest()[:10]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def write_dns_parameters(
    cache_dir: Path, cache_key: DNSCacheKey, domain_timespan: float
) -> None:
    """Write DNS parameters to cache directory."""
    params = cache_key.to_dict()
    params["domain_timespan"] = domain_timespan
    (cache_dir / "dns_parameters.json").write_text(json.dumps(params, indent=2))


class DNSCacheStatus(Enum):
    MISS = auto()
    HIT = auto()
    HIT_SHORT = auto()


@dataclass
class DNSCacheResult:
    status: DNSCacheStatus
    cache_dir: Path | None = None
    cached_timespan: float | None = None


def resolve_dns_cache(
    cache_root: Path,
    cache_key: DNSCacheKey,
    requested_timespan: float,
) -> DNSCacheResult:
    """Check DNS cache for compatible DNS run."""
    cache_dir = cache_root / cache_key.dir_to_name()
    parameters_file = cache_dir / "dns_parameters.json"

    if not parameters_file.exists():
        return DNSCacheResult(status=DNSCacheStatus.MISS)

    stored = json.loads(parameters_file.read_text())
    cached_timespan: float = stored["domain_timespan"]

    if cached_timespan >= requested_timespan - 1e-10:
        return DNSCacheResult(
            status=DNSCacheStatus.HIT,
            cache_dir=cache_dir,
            cached_timespan=cached_timespan,
        )

    return DNSCacheResult(
        status=DNSCacheStatus.HIT_SHORT,
        cache_dir=cache_dir,
        cached_timespan=cached_timespan,
    )


def extend_dns_run(
    cache_dir: Path,
    projection_dir: Path,
    training_dir: Path,
    cache_result: DNSCacheResult,
    problem: Problem,
    disc_cfg: DiscretizationConfig,
    requested_timespan: float,
    run_data_generator_fn: Callable,
) -> None:
    """Extend an existing DNS run to cover the requested timespan."""
    cached_timespan = cache_result.cached_timespan
    extension_duration = requested_timespan - cached_timespan

    last_snapshot, _ = read_data(cache_dir / DNS_FOLDER, final_only=True)

    extension_problem = dataclasses.replace(
        problem,
        domain_timespan=extension_duration,
        initial_condition=last_snapshot,
    )

    run_data_generator_fn(
        problem=extension_problem,
        disc_cfg=disc_cfg,
        master_path=cache_dir,
        dns_save_path=cache_dir / DNS_FOLDER,
        projection_data_path=projection_dir,
        sgsp_data_training_path=training_dir,
        t_start=cached_timespan,
        append_mode=True,
    )

    projector = BurgersDataGenerator(
        problem=dataclasses.replace(problem, domain_timespan=requested_timespan),
        disc_cfg=disc_cfg,
        simulation_mode="dns",
        master_path=cache_dir,
        dns_save_path=cache_dir / "DNS",
        projection_save_path=projection_dir,
        sgsp_training_data_path=training_dir,
    )
    projector.run_projection_only()

    write_dns_parameters(
        cache_dir,
        DNSCacheKey(
            problem_name=extension_problem.name,
            domain_length=extension_problem.domain_length,
            viscosity=extension_problem.viscosity,
            forcing_name=extension_problem.forcing.__name__,
            bc_type=extension_problem.boundary_condition_type,
            bc_value=extension_problem.boundary_condition_value,
            n_nodes_dns=disc_cfg.n_nodes_dns,
            temporal_refinement=disc_cfg.temporal_refinement,
            courant_les=disc_cfg.courant_les,
        ),
        requested_timespan,
    )
