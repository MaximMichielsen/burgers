"""ANN Architecture Class."""

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from ml.ann_config import TauANNConfig, TauANNHyperparameters, TD3Hyperparameters
from proofing.action_based.config import ANNConfigProof
from proofing.contextual_bandit_reformulation.config import ANNBanditConfig


class TauANN(nn.Module):
    """MLP policy πθ : S → A for the Coefficient Controller.

    Maps state sₙ to physical action vector a ∈ [min_action, max_action]
    using a Tanh activation mapped linearly to physical action bounds.
    """

    def __init__(
        self,
        config: TauANNConfig | ANNConfigProof | ANNBanditConfig,
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

        # Action mapping constants: a = center + half * a_norm
        self.act_center = 0.5 * (self.max_action + self.min_action)
        self.act_half = 0.5 * (self.max_action - self.min_action)

        # Zero-initialize the final output layer for clean start near a = 1.0 (baseline)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, state_input: Tensor) -> Tensor:
        """
        Compute the TD3 actor's action output.

        Maps raw network output to normalized range [-1, 1] via tanh,
        then scales to physical range [min_action, max_action].
        """
        raw_out = self.network(state_input)

        if isinstance(self.config, ANNBanditConfig):
            raw_out = raw_out.mean(dim=-1, keepdim=True).expand(-1, self.action_dim)

        a_norm = torch.tanh(raw_out)
        return self.act_center + self.act_half * a_norm


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
