from dataclasses import dataclass
from pathlib import Path

from numpy.typing import NDArray


@dataclass(frozen=True)
class SGSPConfig:
    sgsp_model_path: Path
    normalization_path: Path
    blown_up_path: Path
    clip_pusuluri: bool = True
    clip_rajampeta: bool = False
    sigma_multiplier: float = 3.0
    turn_off_predictor: bool = False


@dataclass(frozen=True)
class AVCTrainerConfig:
    avc_model_path: Path
    simulation_mode: str = "avc"
    correction_mode: str = "global"
    n_skip_steps: int = 5
    correction_is_fixed: bool = False
    perform_zero_run: bool = False


@dataclass(frozen=True)
class AVCRunConfig:
    avc_model_path: Path
    correction_mode: str = "global"
    n_skip_steps: int = 5
    correction_is_fixed: bool = False
    exclude_diss_from_reward: bool = False
