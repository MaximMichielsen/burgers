from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from torch import nn, Tensor

from solvers.solver_base import TauModel

N_HIDDEN_UNITS = 64
EPISODE_REWARD_CLIP = -1e3


class OutputScope(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"


@dataclass
class TauANNConfig:
    tau_model: TauModel
    n_wavenumber_bins: int
    n_coefficients: int
    ann_path: Path | None
    n_skip_steps: int
    n_nodes_les: int
    max_action: float = 1.0

    output_scope: OutputScope = OutputScope.GLOBAL

    reward_weight_energy: float = 1.0
    reward_spectral_exponent: float = 5.0 / 3.0

    def __post_init__(self) -> None:
        self.n_local_groups: int = (
            self.n_nodes_les * 1 if self.output_scope == OutputScope.LOCAL else 1
        )  # for now equal to element amount but in future developments can be less

        self.action_dimension = (
            self.n_coefficients
            if self.output_scope == OutputScope.GLOBAL
            else self.n_coefficients * self.n_local_groups
        )

        self.state_dimension = self.n_wavenumber_bins + self.action_dimension


class TauANN(nn.Module):
    """MLP policy πθ : S → A for the Coefficient Controller.

    Maps state sₙ = (Ê₁...Êₖ, c_...^{n-1}) ∈ ℝ^(K+4) to a coefficient vector.
    """

    def __init__(
        self,
        config: TauANNConfig,
    ):
        super().__init__()

        self.config = config
        self.max_action = config.max_action
        self.state_dim = config.state_dimension
        self.action_dim = config.action_dimension

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, N_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(N_HIDDEN_UNITS, N_HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(N_HIDDEN_UNITS, self.action_dim),
        )

    def forward(self, state_input: Tensor) -> Tensor:
        """Forward pass of the ANN."""
        raw_output = self.network(state_input)
        return self.max_action * torch.tanh(raw_output)


def save_tau_ann(model: TauANN, save_path: Path) -> None:
    """Save tau-ann to save_path."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config,
        },
        save_path,
    )


def load_tau_ann(model_path: Path) -> TauANN:
    """Load tau-ann from model_path."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = TauANN(
        config=checkpoint["config"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
