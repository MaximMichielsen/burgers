from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from constants import HIDDEN_UNITS

# ---------------------------------------------------------------------------
# Data loading (if offline training)
# ---------------------------------------------------------------------------


def load_corrector_training_data(data_path: Path) -> dict:
    """Load training data for the AVCorrector, only used if training is offline."""
    pass


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_VALID_CORRECTION_MODES = frozenset({"global", "local"})


class AVController(nn.Module):
    """MLP policy πθ : S → A for the Global AV corrector.

    Maps state sₙ = (Ê₁..Êₖ, ε⁻ⁿ, αₙ₋₁) ∈ ℝ^(K+2) to a scalar
    action αₙ ∈ [0, alpha_max] via sigmoid output scaling:
        αₙ = alpha_max · σ(W^(L) h^(L-1) + b^(L))

    Architecture: 3 hidden layers × hidden_dim units, ReLU activations.
    Input dim: K + 2, where K = N_LES // 2 (number of resolved wavenumber bins).

    Reference: Research Proposal eq. (2.11).
    """

    def __init__(
        self,
        alpha_max: float,
        n_wavenumber_bins: int,
        hidden_dim: int = HIDDEN_UNITS,
        correction_mode: str = "global",
        n_output_nodes: int = 1,  # ignored for global; N_LES for local.
        output_scale: float | None = None,
    ) -> None:
        super().__init__()

        if correction_mode not in _VALID_CORRECTION_MODES:
            raise ValueError(
                f"Correction mode {correction_mode} not in valid modes: {_VALID_CORRECTION_MODES}"
            )

        if n_output_nodes <= 0:
            raise ValueError(
                f"Output nodes cannot be 0 or negative, got {n_output_nodes}."
            )

        if correction_mode == "global" and n_output_nodes != 1:
            raise ValueError(
                f"Global correction mode requires n_output_nodes=1, got {n_output_nodes}."
            )

        if correction_mode == "local" and n_output_nodes == 1:
            raise ValueError(
                f"Local correction mode requires n_output_nodes > 1, got {n_output_nodes}."
            )

        self.correction_mode = correction_mode

        # State dim: K energy bins + dissipation rate + previous action
        input_dim: int = n_wavenumber_bins + 2
        output_dim: int = n_output_nodes

        self.output_scale = output_scale if output_scale is not None else alpha_max
        self.alpha_max = alpha_max
        self.n_wavenumber_bins = n_wavenumber_bins
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state_input: Tensor) -> Tensor:
        """Sigmoid-bounded output, hard-clamped to alpha_max as a safety ceiling."""
        raw_output = self.network(state_input)
        scaled_output = self.output_scale * torch.sigmoid(raw_output)
        return torch.clamp(scaled_output, max=self.alpha_max, min=0.0)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def save_corrector(model: AVController, save_path: Path) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "alpha_max": model.alpha_max,
            "n_wavenumber_bins": model.n_wavenumber_bins,
            "hidden_dim": model.hidden_dim,
            "correction_mode": model.correction_mode,
            "n_output_nodes": model.output_dim,
            "output_scale": model.output_scale,
        },
        save_path,
    )


def load_corrector(model_path: Path) -> AVController:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = AVController(
        alpha_max=checkpoint["alpha_max"],
        n_wavenumber_bins=checkpoint["n_wavenumber_bins"],
        hidden_dim=checkpoint["hidden_dim"],
        correction_mode=checkpoint.get("correction_mode", "global"),
        n_output_nodes=checkpoint.get("n_output_nodes", 1),
        output_scale=checkpoint.get("output_scale", None),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
