from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn, Tensor

from final.ml.hyperparameters import TauANNHyperparameters, TD3Hyperparameters
from setup.config_discretization import DiscretizationConfig
from solvers.solver_base import TauModel


class Scope(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"
    HYBRID = "hybrid"


@dataclass
class TauANNConfig:
    tau_model: TauModel
    disc_config: DiscretizationConfig
    ann_path: Path | None

    n_skip_steps: int
    output_scope: Scope
    input_scope: Scope

    n_local_action_groups: int | None = None
    local_stencil_size: int | None = None

    group_map: NDArray = field(init=False, repr=False)

    @property
    def n_elements(self) -> int:
        return self.disc_config.n_nodes_les - 1

    def __post_init__(self):
        self.n_coefficients = self.tau_model.output_dimensions

        if self.output_scope == Scope.GLOBAL:
            self.n_local_action_groups = 1

        elif self.output_scope in (Scope.LOCAL, Scope.HYBRID):
            if self.n_local_action_groups is None or self.local_stencil_size is None:
                raise TypeError(
                    f"Set n_local_action_groups ({self.n_local_action_groups}) and local_stencil_size ({self.local_stencil_size})"
                )

            else:
                self.n_local_action_groups = max(
                    1, min(self.n_local_action_groups, self.n_elements)
                )
        else:
            raise ValueError(
                f"Invalid output scope received: {self.output_scope} | "
                f"choose from: {', '.join(scope.name for scope in Scope)}"
            )

        assert self.n_local_action_groups is not None
        self.group_map = self._create_group_map(
            n_elements=self.n_elements, n_groups=self.n_local_action_groups
        )

        self._set_action_dimension_size()
        self._set_state_dimension_size()

        self.hidden_dimension = max(64, int(self.state_dimension * 1.5))

    def _set_action_dimension_size(self) -> None:
        """Set the size of the action dimension the ANN will use based on the output scope."""
        if self.output_scope == Scope.GLOBAL:
            self.action_dimension = self.n_coefficients
        else:
            self.action_dimension = (
                self.n_coefficients * (self.n_local_action_groups + 1)
                if self.output_scope == Scope.HYBRID
                else self.n_coefficients * self.n_local_action_groups
            )

    def _set_state_dimension_size(self) -> None:
        """Set the size of the state dimension."""
        local_input_stencil_dimension = 0
        if self.input_scope == Scope.LOCAL:
            # u_local (n_points) + u_x_local (n_points) per node
            features_per_node = 2 * self.local_stencil_size
            local_input_stencil_dimension = (
                self.disc_config.n_nodes_les * features_per_node
            )

        self.state_dimension = (
            self.disc_config.n_wavenumber_bins
            + local_input_stencil_dimension
            + self.action_dimension
            + self.disc_config.n_nodes_les
        )

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
        hyperparams: TauANNHyperparameters | TD3Hyperparameters,
    ):
        super().__init__()

        self.config = config
        self.hyperparameters = hyperparams
        self.max_action = hyperparams.max_action
        self.min_action = hyperparams.min_action
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
        """
        Compute the TD3 actor's action output.

        Maps raw network output to the closed range [min_action, max_action]
        Bounds are enforced implicitly by sigmoid's saturation
        """
        raw_output = self.network(state_input)
        return self.min_action + (self.max_action - self.min_action) * torch.sigmoid(
            raw_output
        )


def save_tau_ann(model: TauANN, save_path: Path) -> None:
    """Save tau-ann to save_path."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config,
            "hyperparameters": model.hyperparameters,
        },
        save_path,
    )


def load_tau_ann(model_path: Path) -> TauANN:
    """Load tau-ann from model_path."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = TauANN(
        config=checkpoint["config"],
        hyperparams=checkpoint["hyperparameters"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
