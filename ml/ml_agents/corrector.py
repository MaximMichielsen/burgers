from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from numpy.typing import NDArray
from torch import nn, Tensor

from constants import (
    AVC_ADDITIONAL_INPUT_DIMENSIONS,
    AVC_GLOBAL_OUTPUT_UNITS,
    AVC_HIDDEN_UNITS,
)

class Transition(NamedTuple):
    """Single MDP transition (sₙ, αₙ, rₙ, sₙ₊₁, done)."""

    state: NDArray
    action: NDArray | float
    reward: float
    next_state: NDArray
    done: bool

@dataclass(frozen=True)
class AVCTrainingConfig:
    """Configuration file for the AVC training algorithm."""

    batch_size: int = 256
    reward_weight_energy: float = 1.0
    reward_spectral_exponent: float = 5.0 / 3.0
    n_warmup_steps: int = 100
    exploration_bound_upper: float = 1.0
    update_every: int = 1
    updates_per_step: int = 1


@dataclass(frozen=True)
class AVCConfig:
    """Configuration file for the AVC model."""

    avc_model_path: Path
    n_wavenumber_bins: int
    input_scope: str = "global"
    output_scope: str = "global"
    n_skip_steps: int = 5


class AVController(nn.Module):
    """MLP policy πθ : S → A for the Global AV controller.

    Maps state sₙ = (Ê₁..Êₖ, ε⁻ⁿ, αₙ₋₁) ∈ ℝ^(K+2) to a scalar action αₙ ∈ R.
    """

    def __init__(
        self,
        avc_config: AVCConfig,
    ) -> None:
        super().__init__()

        self.avc_config = avc_config
        self.input_scope = avc_config.input_scope
        self.output_scope = avc_config.output_scope

        self.input_dimensions: int = (
            avc_config.n_wavenumber_bins + AVC_ADDITIONAL_INPUT_DIMENSIONS
        )
        self.output_dimensions: int = AVC_GLOBAL_OUTPUT_UNITS

        self.network = nn.Sequential(
            nn.Linear(self.input_dimensions, AVC_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(AVC_HIDDEN_UNITS, AVC_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(AVC_HIDDEN_UNITS, self.output_dimensions),
        )

    def forward(self, state_input: Tensor) -> Tensor:
        """Unbounded network output."""
        return self.network(state_input)


def save_corrector(model: AVController, save_path: Path) -> None:
    """Save corrector to save_path."""
    config_dict = {**vars(model.avc_config), "avc_model_path": str(model.avc_config.avc_model_path)}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "avc_config": config_dict,
        },
        save_path,
    )


def load_corrector(model_path: Path) -> AVController:
    """Load corrector from model_path."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    config_dict = checkpoint["avc_config"]
    avc_config = AVCConfig(
        **{**config_dict, "avc_model_path": Path(config_dict["avc_model_path"])}
    )
    model = AVController(avc_config=avc_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
