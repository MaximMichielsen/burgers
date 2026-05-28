"""Pipeline stage flags and run-ID generation for the Burgers LES pipeline."""

import datetime
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable

from constants import (
    SOLVER_DATA_FOLDER,
    DNS_SAVE_PATH,
    LES_ANALYTICAL_SAVE_PATH,
    LES_NO_MODEL_SAVE_PATH,
    PRE_SPLIT_FOLDER,
    TRAINING_DATA_FOLDER,
    POST_SPLIT_FOLDER,
    AGENT_FOLDER,
    A_PRIORI_FOLDER,
)


@dataclass
class PipelineConfig:
    """Controls which pipeline stages run and optional I/O path overrides."""

    run_dns: bool = True
    run_solvers: bool = True
    run_projection: bool = True
    run_training_assembly: bool = True
    run_training_sgsp: bool = True
    verify_apriori: bool = True
    run_sgsp: bool = True
    debug_sgsp: bool = False
    run_avc_online_training: bool = True
    run_avc_offline_training: bool = False if run_avc_online_training else True
    run_avc_eval: bool = (
        False if not run_avc_offline_training or not run_avc_online_training else True
    )
    run_avc: bool = True
    run_plotting: bool = True
    clip_pusuluri: bool = False
    clip_rajampeta: bool = False
    manual_path: str = ""
    _stage_timings: OrderedDict[str, float] = field(
        default_factory=OrderedDict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate flag combinations."""
        if self.clip_rajampeta and not self.clip_pusuluri:
            raise ValueError("clip_rajampeta requires clip_pusuluri to be enabled.")

    def get_run_id(self, problem_name: str) -> str:
        """Return manual_path if set, else generate a timestamped run ID."""
        if self.manual_path:
            return self.manual_path
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        return f"run_{problem_name}_{timestamp}"

    @classmethod
    def all_stages(cls, manual_path: str = "") -> "PipelineConfig":
        """Full pipeline (identical to default construction)."""
        return cls(manual_path=manual_path)

    @classmethod
    def all_stages_clipped(cls, manual_path: str = "") -> "PipelineConfig":
        """Full pipeline with predictor clipping enabled."""
        return cls(manual_path=manual_path, clip_pusuluri=True, clip_rajampeta=True)

    @classmethod
    def all_but_dns_clipped(cls, manual_path: str = "") -> "PipelineConfig":
        """Full pipeline minus DNS, with predictor clipping enabled."""
        return cls(
            manual_path=manual_path,
            run_dns=False,
            clip_pusuluri=True,
            clip_rajampeta=True,
        )

    @classmethod
    def coupled_only(cls, manual_path: str = "") -> "PipelineConfig":
        """Coupled simulation and plotting only; skips data generation and training."""
        return cls(
            run_solvers=False,
            run_projection=False,
            run_training_assembly=False,
            run_training_sgsp=False,
            verify_apriori=False,
            run_sgsp=True,
            run_plotting=True,
            manual_path=manual_path,
        )

    @classmethod
    def coupled_only_clipped(cls, manual_path: str = "") -> "PipelineConfig":
        """Coupled simulation and plotting only, with predictor clipping enabled."""
        return cls(
            run_solvers=False,
            run_projection=False,
            run_training_assembly=False,
            run_training_sgsp=False,
            verify_apriori=False,
            run_sgsp=True,
            run_plotting=True,
            manual_path=manual_path,
            clip_pusuluri=True,
            clip_rajampeta=True,
        )

    @classmethod
    def only_plot(cls, manual_path: str) -> "PipelineConfig":
        """Plotting only; requires an existing run via manual_path."""
        return cls(
            run_solvers=False,
            run_projection=False,
            run_training_assembly=False,
            run_training_sgsp=False,
            verify_apriori=False,
            run_sgsp=False,
            run_plotting=True,
            manual_path=manual_path,
        )

    def stage(self, stage_name: str, enabled: bool = True) -> Callable:
        """Decorator that times a pipeline stage and stores elapsed seconds.

        Args:
            stage_name: Human-readable label stored in timing report.
            enabled: If False, the decorated function is a no-op (skipped stage).
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not enabled:
                    print(f"[pipeline] Skipping: {stage_name}")
                    return None
                print(f"[pipeline] Starting: {stage_name}")
                t_start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed_seconds = time.perf_counter() - t_start
                self._stage_timings[stage_name] = elapsed_seconds
                print(f"[pipeline] Done:     {stage_name}  ({elapsed_seconds:.2f}s)")
                return result

            return wrapper

        return decorator

    def report_timings(self) -> None:
        """Print a formatted timing summary for all completed stages."""
        total_elapsed = sum(self._stage_timings.values())
        print("\n" + "=" * 52)
        print(f"{'Pipeline Stage':<32} {'Time (s)':>10}  {'%':>6}")
        print("=" * 52)
        for stage_name, elapsed_seconds in self._stage_timings.items():
            pct = 100 * elapsed_seconds / total_elapsed if total_elapsed > 0 else 0.0
            print(f"  {stage_name:<30} {elapsed_seconds:>10.2f}  {pct:>5.1f}%")
        print("-" * 52)
        print(f"  {'TOTAL':<30} {total_elapsed:>10.2f}  100.0%")
        print("=" * 52)


@dataclass
class RunPaths:
    """All output directories for a single pipeline run."""

    master: Path
    solver_data: Path
    dns_data: Path
    les_a_data: Path
    les_nm_data: Path
    projection: Path
    training: Path
    model_output: Path
    apriori: Path

    @classmethod
    def from_master(cls, master_path: Path) -> "RunPaths":
        """Derive all subdirectories from the master run path."""
        return cls(
            master=master_path,
            solver_data=master_path / SOLVER_DATA_FOLDER,
            dns_data=master_path / SOLVER_DATA_FOLDER / DNS_SAVE_PATH,
            les_a_data=master_path / SOLVER_DATA_FOLDER / LES_ANALYTICAL_SAVE_PATH,
            les_nm_data=master_path / SOLVER_DATA_FOLDER / LES_NO_MODEL_SAVE_PATH,
            projection=master_path / TRAINING_DATA_FOLDER / PRE_SPLIT_FOLDER,
            training=master_path / TRAINING_DATA_FOLDER / POST_SPLIT_FOLDER,
            model_output=master_path / AGENT_FOLDER,
            apriori=master_path / A_PRIORI_FOLDER,
        )

    def create_master(self) -> None:
        """Create the master directory; subdirectories are created on demand by each step."""
        self.master.mkdir(parents=True, exist_ok=True)
