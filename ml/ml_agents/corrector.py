from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn, Tensor

from constants import (
    AVC_ADDITIONAL_INPUT_DIMENSIONS,
    AVC_GLOBAL_OUTPUT_UNITS,
    AVC_HIDDEN_UNITS,
)


@dataclass(frozen=True)
class AVCConfig:
    """Configuration file for the AVC."""

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
        n_wavenumber_bins: int,
        avc_config: AVCConfig,
    ) -> None:
        super().__init__()

        self.input_scope = avc_config.input_scope
        self.output_scope = avc_config.output_scope

        self.input_dimension: int = n_wavenumber_bins + AVC_ADDITIONAL_INPUT_DIMENSIONS
        self.output_dimensions: int = AVC_GLOBAL_OUTPUT_UNITS

        self.network = nn.Sequential(
            nn.Linear(self.input_dimension, AVC_HIDDEN_UNITS),
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
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_scope": model.input_scope,
            "output_scope": model.output_scope,
            "input_dimensions": model.input_dimensions,
            "output_dimensions": model.output_dimensions,
        },
        save_path,
    )


def load_corrector(model_path: Path) -> AVController:
    """Load corrector from model_path."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = AVController(
        input_scope=checkpoint["input_scope"],
        output_scope=checkpoint["output_scope"],
        input_dimensions=checkpoint["input_dimensions"],
        output_dimensions=checkpoint["output_dimensions"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
