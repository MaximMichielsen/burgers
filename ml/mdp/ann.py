from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn, Tensor

from ml.original.ann_config import Scope
from solvers.solver_base import TauModel

MAX_ACTION = 5
MIN_ACTION = 0.9

# state dim: spectral w and spectral w_x
# TODO: add previous actions to state


@dataclass
class MDPConfig:
    tau_model: TauModel
    ann_path: Path | None
    n_training_episodes: int
    training_mode: bool
    include_reynolds_signal: bool
    n_agent_steps_per_episode: int

    random_exploration_ratio: float = 0.25

    max_action: float = MAX_ACTION
    min_action: float = MIN_ACTION

    hidden_dimension: int = 64
    n_nodes_les: int = 17

    with_improvement_signal: bool = False
    with_exponential_decay: bool = False

    input_scope = Scope.GLOBAL
    output_scope = Scope.GLOBAL

    def __post_init__(self):
        self.action_dimension = self.tau_model.output_dimensions
        self.n_random_episodes = int(
            self.n_training_episodes * self.random_exploration_ratio
        )
        self.state_dimension = (
            int(2 * ((self.n_nodes_les - 1) // 2)) + self.action_dimension + 1
        )


@dataclass
class MDPHyperparameters:
    weight_mean_profile: float = 10.0
    weight_reynolds: float = 10.0

    lr: float = 3e-4
    discount: float = 0.99
    tau_polyak: float = 0.005
    policy_noise: float = 0.10
    noise_clip: float = 0.3
    policy_freq: int = 2

    batch_size: int = 64
    expl_noise: float = 0.10
    replay_buffer_max_size: int = int(1e5)

    n_updates_post_episode = 20


class MDPANN(nn.Module):
    """ANN tailored specifically to the bandit reformulation."""

    def __init__(self, config):
        super().__init__()

        self.config: MDPConfig = config

        self.max_action = config.max_action
        self.min_action = config.min_action
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

        nn.init.uniform_(self.network[-1].weight, -3e-3, 3e-3)
        nn.init.zeros_(self.network[-1].bias)

        self.act_center = 0.5 * (self.max_action + self.min_action)
        self.act_half = 0.5 * (self.max_action - self.min_action)

    def forward(self, state_input: Tensor) -> Tensor:
        """
        Compute the TD3 actor's action output.

        Maps raw network output to normalized range [-1, 1] via tanh,
        then scales to physical range [min_action, max_action].
        """
        return self.act_center + self.act_half * torch.tanh(self.network(state_input))


def save_tau_ann(model: MDPANN, save_path: Path) -> None:
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


def load_tau_ann(model_path: Path) -> MDPANN:
    """Load tau-ann from model_path."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = MDPANN(
        config=checkpoint["config"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
