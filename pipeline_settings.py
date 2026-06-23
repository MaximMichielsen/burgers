"""Pipeline stage flags and run-ID generation for the Burgers LES pipeline."""

from dataclasses import dataclass
from pathlib import Path

from constants import (
    SOLVER_DATA_FOLDER,
    LES_ANALYTICAL_SAVE_PATH,
    LES_NO_MODEL_SAVE_PATH,
    AGENT_FOLDER,
    A_PRIORI_FOLDER,
    LES_SGSP_SAVE_PATH,
    LES_AVC_SAVE_PATH,
)


@dataclass
class RunPaths:
    """All output directories for a single pipeline run."""

    master: Path
    solver_data: Path
    dns_data: Path | None
    les_a_data: Path
    les_nm_data: Path
    les_sgsp_data: Path
    les_avc_data: Path
    projection: Path | None
    training: Path | None
    agents: Path
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
            dns_data=None,
            les_a_data=master_path / SOLVER_DATA_FOLDER / LES_ANALYTICAL_SAVE_PATH,
            les_nm_data=master_path / SOLVER_DATA_FOLDER / LES_NO_MODEL_SAVE_PATH,
            les_sgsp_data=master_path / SOLVER_DATA_FOLDER / LES_SGSP_SAVE_PATH,
            projection=None,
            training=None,
            agents=master_path / AGENT_FOLDER,
            sgsp_model=master_path / AGENT_FOLDER / "sgs_predictor.pt",
            avcg_model=master_path / AGENT_FOLDER / "av_global_corrector.pt",
            avcl_model=master_path / AGENT_FOLDER / "av_local_corrector.pt",
            apriori=master_path / A_PRIORI_FOLDER,
            les_avc_data=master_path / SOLVER_DATA_FOLDER / LES_AVC_SAVE_PATH,
        )

    def create_master(self) -> None:
        """Create the master directory; subdirectories are created on demand by each step."""
        self.master.mkdir(parents=True, exist_ok=True)
