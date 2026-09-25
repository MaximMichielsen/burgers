from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn, Tensor


from setup.config_discretization import DiscretizationConfig
from solvers.solver_base import TauModel

MAX_ACTION = 1.0
MIN_ACTION = 0.5


class Scope(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"
    HYBRID = "hybrid"


@dataclass
class TauANNHyperparameters:
    """Hyperparameters tied directly to the ANN, regardless of training mechanism."""

    max_action: float = MAX_ACTION
    min_action: float = MIN_ACTION + 1e-3

    smoothing_factor = 0.002
    burn_in_steps = 100

    weight_improvement = 100
    weight_absolute_error = 1e-3
    weight_spectral = 1e-3
    weight_action = 1.0
    gamma = 5.0 / 3.0


@dataclass
class TD3Hyperparameters(TauANNHyperparameters):
    """Hyperparameters for TD3 Agent training and environment interactions."""

    # Agent / Optimization Params
    lr: float = 1e-4
    discount: float = 0.99
    tau_polyak: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    # Training / Environment Setup Params
    batch_size: int = 64
    expl_noise: float = 0.1
    replay_buffer_max_size: int = int(1e5)
    stochastic_timesteps = 100


@dataclass
class TauANNConfig:
    tau_model: TauModel
    disc_config: DiscretizationConfig
    ann_path: Path | None

    n_skip_steps: int
    n_training_episodes: int
    output_scope: Scope
    input_scope: Scope

    training_mode: bool

    max_action: float = MAX_ACTION
    min_action: float = MIN_ACTION

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


class ReplayBuffer:
    """Experience replay memory storing transitions and returning Tensors."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = int(1e5),
        device: str | torch.device = "cpu",
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        state: NDArray,
        action: NDArray,
        next_state: NDArray,
        reward: float,
        done: bool,
    ) -> None:
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.as_tensor(self.state[ind], device=self.device),
            torch.as_tensor(self.action[ind], device=self.device),
            torch.as_tensor(self.next_state[ind], device=self.device),
            torch.as_tensor(self.reward[ind], device=self.device),
            torch.as_tensor(self.done[ind], device=self.device),
        )
