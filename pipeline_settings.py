"""Pipeline stage flags and run-ID generation for the Burgers LES pipeline."""

import datetime
from dataclasses import dataclass
from pathlib import Path

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
    LES_SGSP_SAVE_PATH,
    LES_AVC_SAVE_PATH,
    BLOWN_UP_FOLDER,
    NORM_STATS,
)


@dataclass
class PipelineConfig:
    """Controls which pipeline stages run and optional I/O path overrides."""

    run_dns: bool = True
    run_base_models: bool = True
    run_sgsp_block: bool = True
    run_avc_block: bool = True

    train_avc_online: bool = True
    train_avc_offline: bool = False

    clip_pusuluri: bool = False
    clip_rajampeta: bool = False

    manual_path: str = ""

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
    def all(cls, manual_path: str = "") -> "PipelineConfig":
        """Full pipeline (identical to default construction)."""
        return cls(manual_path=manual_path)

    @classmethod
    def all_but_dns(cls, manual_path: str = "") -> "PipelineConfig":
        """Full pipeline minus DNS, with predictor clipping enabled."""
        return cls(
            manual_path=manual_path,
            run_dns=False,
        )


@dataclass
class RunPaths:
    """All output directories for a single pipeline run."""

    master: Path
    solver_data: Path
    dns_data: Path
    les_a_data: Path
    les_nm_data: Path
    les_sgsp_data: Path
    les_avc_data: Path
    projection: Path
    training: Path
    normalization: Path
    model_output: Path
    sgsp_model: Path
    avcg_model: Path
    avcl_model: Path
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
            les_sgsp_data=master_path / SOLVER_DATA_FOLDER / LES_SGSP_SAVE_PATH,
            projection=master_path / TRAINING_DATA_FOLDER / PRE_SPLIT_FOLDER,
            training=master_path / TRAINING_DATA_FOLDER / POST_SPLIT_FOLDER,
            normalization=master_path
            / TRAINING_DATA_FOLDER
            / POST_SPLIT_FOLDER
            / NORM_STATS,
            model_output=master_path / AGENT_FOLDER,
            sgsp_model=master_path / AGENT_FOLDER / "sgs_predictor.pt",
            avcg_model=master_path / AGENT_FOLDER / "av_global_corrector.pt",
            avcl_model=master_path / AGENT_FOLDER / "av_local_corrector.pt",
            apriori=master_path / A_PRIORI_FOLDER,
            les_avc_data=master_path / SOLVER_DATA_FOLDER / LES_AVC_SAVE_PATH,
        )

    def create_master(self) -> None:
        """Create the master directory; subdirectories are created on demand by each step."""
        self.master.mkdir(parents=True, exist_ok=True)
