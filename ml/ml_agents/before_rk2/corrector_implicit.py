from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from constants import AVC_HIDDEN_UNITS, AVC_GLOBAL_OUTPUT_UNITS


@dataclass(frozen=True)
class AVCConfig:
    avc_model_path: Path
    n_wavenumber_bins: int
    correction_mode: str
    input_mode: str = "global"
    simulation_mode: str = "avc"
    n_skip_steps: int = 5
    externally_driven: bool = False


class AVController(nn.Module):
    """MLP policy πθ : S → A for the Global AV controller.

    Maps state sₙ = (Ê₁..Êₖ, ε⁻ⁿ, αₙ₋₁) ∈ ℝ^(K+2) to a scalar action αₙ ∈ [0, output_scale]
    """

    def __init__(
        self,
        n_wavenumber_bins: int,
        output_scale: float,
        correction_mode: str,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.correction_mode = correction_mode

        self.input_dim: int = n_wavenumber_bins + 2
        self.n_wavenumber_bins = n_wavenumber_bins
        self.output_scale = output_scale
        self.output_dim: int = output_dim

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, AVC_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(AVC_HIDDEN_UNITS, AVC_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(AVC_HIDDEN_UNITS, self.output_dim),
        )

    # todo: just return raw output?
    def forward(self, state_input: Tensor) -> Tensor:
        """Sigmoid-bounded output scaled to physical range [0, output_scale]."""
        raw_output = self.network(state_input)
        return self.output_scale * torch.sigmoid(raw_output)


def save_corrector(model: AVController, save_path: Path) -> None:
    """Save corrector to save_path."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_wavenumber_bins": model.n_wavenumber_bins,
            "correction_mode": model.correction_mode,
            "n_output_nodes": model.output_dim,
            "output_scale": model.output_scale,
        },
        save_path,
    )


def load_corrector(model_path: Path) -> AVController:
    """Load corrector from model_path."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = AVController(
        n_wavenumber_bins=checkpoint["n_wavenumber_bins"],
        output_scale=checkpoint["output_scale"],
        correction_mode=checkpoint["correction_mode"],
        output_dim=checkpoint["n_output_nodes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
