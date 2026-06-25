from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from constants import SGSP_HIDDEN_UNITS, AVC_HIDDEN_UNITS, AVC_GLOBAL_OUTPUT_UNITS


def load_corrector_training_data(data_path: Path) -> dict:
    """Load training data for the AVCorrector, only used if training is offline."""
    pass


class AVControllerGlobal(nn.Module):
    """MLP policy πθ : S → A for the Global AV controller.

    Maps state sₙ = (Ê₁..Êₖ, ε⁻ⁿ, αₙ₋₁) ∈ ℝ^(K+2) to a scalar action αₙ ∈ [0, output_scale]
    """

    def __init__(
        self,
        n_wavenumber_bins: int,
        output_scale: float,
    ) -> None:
        super().__init__()

        self.correction_mode = "global"

        self.input_dim: int = n_wavenumber_bins + 2
        self.n_wavenumber_bins = n_wavenumber_bins
        self.output_scale = output_scale
        self.output_dim: int = AVC_GLOBAL_OUTPUT_UNITS

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, AVC_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(AVC_HIDDEN_UNITS, AVC_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(AVC_HIDDEN_UNITS, self.output_dim),
        )

    def forward(self, state_input: Tensor) -> Tensor:
        """Sigmoid-bounded output scaled to physical range [0, output_scale]."""
        raw_output = self.network(state_input)
        return self.output_scale * torch.sigmoid(raw_output)


def save_corrector(model: AVControllerGlobal, save_path: Path) -> None:
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


def load_corrector(model_path: Path) -> AVControllerGlobal:
    """Load corrector from model_path."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = AVControllerGlobal(
        n_wavenumber_bins=checkpoint["n_wavenumber_bins"],
        output_scale=checkpoint.get("output_scale", None),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
