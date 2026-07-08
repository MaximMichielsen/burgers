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
class AVCConfig:
    avc_model_path: Path
    n_wavenumber_bins: int
    correction_mode: str
    input_mode: str = "global"
    simulation_mode: str = "avc"
    n_skip_steps: int = 5
    externally_driven: bool = False
