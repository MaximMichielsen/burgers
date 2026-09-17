from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn, Tensor

from solvers.solver_base import TauModel

N_HIDDEN_UNITS = 64
EPISODE_REWARD_CLIP = -1e3


class Scope(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"
    HYBRID = "hybrid"
    OUT_FULL_LOCAL = "full_local"


@dataclass
class TauANNConfig:
    tau_model: TauModel
    n_wavenumber_bins: int
    n_coefficients: int
    ann_path: Path | None
    n_skip_steps: int
    n_nodes_les: int

    n_local_action_groups: int

    max_action: float = 1.0
    min_action: float = 0.0 + 1e-3
    n_local_stencil_points: int = 8

    output_scope: Scope = Scope.GLOBAL
    input_scope: Scope = Scope.GLOBAL

    input_scope_mode: str = "spatial"

    reward_weight_energy: float = 1.0
    reward_spectral_exponent: float = 5.0 / 3.0

    group_map: NDArray = field(init=False, repr=False)

    _VALID_INPUT_MODES: frozenset[str] = frozenset({"spatial", "spectral"})

    @property
    def n_elements(self) -> int:
        return self.n_nodes_les - 1

    def __post_init__(self) -> None:
        mode_str = (
            self.input_scope_mode.value
            if hasattr(self.input_scope_mode, "value")
            else str(self.input_scope_mode)
        )
        if mode_str not in self._VALID_INPUT_MODES:
            raise ValueError(
                f"Invalid input stencil mode. Received '{self.input_scope_mode}' | "
                f"Valid options are: {set(self._VALID_INPUT_MODES)}"
            )

        if self.output_scope == Scope.GLOBAL:
            self.n_local_action_groups = 1
        elif self.output_scope in (Scope.LOCAL, Scope.HYBRID):
            self.n_local_action_groups = max(
                1, min(self.n_local_action_groups, self.n_elements)
            )
        elif self.output_scope == Scope.OUT_FULL_LOCAL:
            self.n_local_action_groups = int(self.n_elements)
        else:
            raise ValueError(
                f"Invalid output scope received: {self.output_scope} | "
                f"choose from: {', '.join(scope.name for scope in Scope)}"
            )

        self.group_map = self._create_group_map(
            n_elements=self.n_elements, n_groups=self.n_local_action_groups
        )

        if self.output_scope == Scope.GLOBAL:
            self.action_dimension = self.n_coefficients
        elif self.output_scope == Scope.HYBRID:
            self.action_dimension = self.n_coefficients * (
                self.n_local_action_groups + 1
            )
        else:
            self.action_dimension = self.n_coefficients * self.n_local_action_groups

        self.n_local_wavenumber_bins: int = (self.n_local_stencil_points - 1) // 2

        local_state_dimension = 0
        if self.input_scope == Scope.LOCAL:
            if self.input_scope_mode == "spatial":
                # u_local (n_points) + u_x_local (n_points) per node
                features_per_node = 2 * self.n_local_stencil_points
            else:  # spectral
                features_per_node = self.n_local_wavenumber_bins

            local_state_dimension = self.n_nodes_les * features_per_node

        self.state_dimension = (
            self.n_wavenumber_bins + self.action_dimension + local_state_dimension
        )

        self.hidden_dimension = max(64, int(self.state_dimension * 1.5))

    @staticmethod
    def _create_group_map(n_elements: int, n_groups: int) -> NDArray:
        """Builds a 1D mapping array mapping each element index to a group ID."""
        base_size = n_elements // n_groups
        remainder = n_elements % n_groups

        group_map = np.zeros(n_elements, dtype=int)
        current_element = 0

        for g in range(n_groups):
            group_size = base_size + (1 if g < remainder else 0)
            group_map[current_element : current_element + group_size] = g
            current_element += group_size

        return group_map


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
        self.hidden_dim = config.hidden_dimension

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def forward(self, state_input: Tensor) -> Tensor:
        raw_output = self.network(state_input)
        min_act = 0.01  # Minimum stabilization floor
        return min_act + (self.max_action - min_act) * torch.sigmoid(raw_output)


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
