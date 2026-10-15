import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


@dataclass(frozen=True)
class LESCacheKey:
    """Uniquely define an LES run by its simulation parameters."""

    problem_name: str
    domain_length: float
    viscosity: float
    bc_type: str
    bc_value: float | int | tuple[float | int, float | int] | None
    n_nodes: int
    dt: float
    t_start: float
    t_end: float
    n_params: int

    def dir_to_name(self) -> str:
        """Short deterministic hash used as cache directory name."""
        key_str = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return "les_" + hashlib.sha1(key_str.encode()).hexdigest()[:10]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def write_les_parameters(cache_dir: Path, cache_key: LESCacheKey) -> None:
    """Write LES parameters to cache directory."""
    params = cache_key.to_dict()
    (cache_dir / "les_parameters.json").write_text(json.dumps(params, indent=2))


class LESCacheStatus(Enum):
    MISS = auto()
    HIT = auto()


@dataclass
class LESCacheResult:
    status: LESCacheStatus
    cache_dir: Path | None = None


def resolve_les_baseline_cache(
    cache_root: Path,
    cache_key: LESCacheKey,
) -> LESCacheResult:
    """Check LES cache for compatible LES baseline run."""
    cache_dir = cache_root / cache_key.dir_to_name()
    parameters_file = cache_dir / "les_parameters.json"

    if not parameters_file.exists():
        return LESCacheResult(status=LESCacheStatus.MISS)

    return LESCacheResult(
        status=LESCacheStatus.HIT,
        cache_dir=cache_dir,
    )
