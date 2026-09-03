from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn, Tensor

from ml.constants import N_HIDDEN_UNITS


@dataclass
class TauANNConfig:
    tau_model: str
    n_wavenumber_bins: int
    n_coefficients: int
    ann_path: Path
    n_skip_steps: int

    reward_weight_energy: float = 1.0
    reward_spectral_exponent: float = 5.0 / 3.0

    def __post_init__(self) -> None:
        self.input_dimension = self.n_wavenumber_bins + self.n_coefficients


class TauANN(nn.Module):
    """MLP policy πθ : S → A for the Coefficient Controller.

    Maps state sₙ = (Ê₁..Êₖ, c_...^{n-1}) ∈ ℝ^(K+4) to a coefficient vector.
    """

    def __init__(self, n_wavenumber_bins: int, n_coefficients: int, max_action: float = 1.0):
        super().__init__()

        self.input_dim: int = n_wavenumber_bins + n_coefficients
        self.n_wavenumber_bins = n_wavenumber_bins
        self.n_coefficients = n_coefficients
        self.max_action = max_action

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, N_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(N_HIDDEN_UNITS, N_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(N_HIDDEN_UNITS, n_coefficients),
        )

    def forward(self, state_input: Tensor) -> Tensor:
        """Forward pass of the ANN.."""
        raw_output= self.network(state_input)
        return self.max_action * torch.tanh(raw_output)


def save_tau_ann(model: TauANN, save_path: Path) -> None:
    """Save tau-ann to save_path."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_wavenumber_bins": model.n_wavenumber_bins,
            "n_coefficients": model.n_coefficients,
            "max_action": model.max_action,
        },
        save_path,
    )


def load_tau_ann(model_path: Path) -> TauANN:
    """Load tau-ann from model_path."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = TauANN(
        n_wavenumber_bins=checkpoint["n_wavenumber_bins"],
        n_coefficients=checkpoint["n_coefficients"],
        max_action=checkpoint.get("max_action", 1.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
