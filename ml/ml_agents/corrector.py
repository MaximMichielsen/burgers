from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import Literal

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


class AVCorrector(nn.Module):
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
        correction_mode: Literal["global", "local"] = "global",
        n_output_nodes: int = 1,  # ignored for global; N_LES for local.
    ) -> None:
        super().__init__()

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

        # TODO: rethink this whole alpha max thing

    def forward(self, state_input: Tensor) -> Tensor:
        """Returns shape (batch, 1) for global or (batch, N) for local.

        Network outputs are passed through softplus to enforce positivity,
        then hard-clipped to alpha_max. This decouples the learned scale
        from the upper bound — raising alpha_max only expands headroom,
        it does not rescale existing outputs.
        """
        raw_output = self.network(state_input)
        positive_output = nn.functional.softplus(raw_output)
        return torch.clamp(positive_output, max=self.alpha_max)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_corrector() -> None:
    """RL algorithm for training the AVCorrector."""
    pass


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def save_corrector(model: AVCorrector, save_path: Path) -> None:
    """Save AVCorrector weights and config to a .pt checkpoint."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "alpha_max": model.alpha_max,
            "n_wavenumber_bins": model.n_wavenumber_bins,
            "hidden_dim": model.hidden_dim,
        },
        save_path,
    )


def load_corrector(model_path: Path) -> AVCorrector:
    """Load a saved AVCorrector from a .pt checkpoint."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = AVCorrector(
        alpha_max=checkpoint["alpha_max"],
        n_wavenumber_bins=checkpoint["n_wavenumber_bins"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
