import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn, Tensor

from ml.constants import N_HIDDEN_UNITS


class TwinQCritic(nn.Module):
    """Twin Q-Networks sized to match the TauANN hidden layer dimension."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = N_HIDDEN_UNITS
    ):
        super().__init__()

        input_dim = state_dim + action_dim

        # Q1 architecture
        self.q1_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 architecture
        self.q2_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa), self.q2_net(sa)

    def q1(self, state: Tensor, action: Tensor) -> Tensor:
        return self.q1_net(torch.cat([state, action], dim=-1))


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
