import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from numpy._typing import NDArray
from torch import nn, Tensor
from torch.distributions import Normal

from ml.constants import N_HIDDEN_UNITS
from ml.training.shared_assets import TwinQCritic, ReplayBuffer


# =============================================================================
# Hyperparameters
# =============================================================================


@dataclass
class SACHyperparameters:
    """Complete hyperparameter configuration for SAC."""

    # Required Environment Parameter
    action_dim: int

    # Temperature & Entropy
    alpha: float = 0.2
    auto_temperature_tuning: bool = True
    target_entropy: float = field(init=False)

    # Optimization Learning Rates
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4
    learning_rate_alpha: float = 3e-4
    max_grad_norm: float | None = None

    # Network Architecture
    hidden_dim: int = 256
    n_hidden_layers: int = 2
    max_action: float = 1.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    # RL & Target Updates
    discount: float = 0.99
    polyak_tau: float = 0.005
    utd_ratio: int = 1  # Gradient steps per env step

    # Replay Memory & Sampling
    total_episodes: int = 100
    start_timesteps: int = 1000
    batch_size: int = 256
    replay_buffer_size: int = int(1e6)

    def __post_init__(self):
        self.target_entropy = -float(self.action_dim)


# =============================================================================
# Actor
# =============================================================================


class SACActor(nn.Module):
    """Stochastic Gaussian Policy with Tanh Squashing for SAC."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hp: Optional[SACHyperparameters] = None,
    ):
        super().__init__()

        if hp is None:
            hp = SACHyperparameters(action_dim=action_dim)
        self.hp = hp

        self.action_dim = action_dim
        self.max_action = hp.max_action
        self.hp = hp

        # Shared MLP backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hp.hidden_dim),
            nn.ReLU(),
            nn.Linear(hp.hidden_dim, hp.hidden_dim),
            nn.ReLU(),
        )

        # Output heads for Mean and Log Standard Deviation
        self.mean_head = nn.Linear(hp.hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hp.hidden_dim, action_dim)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Returns gaussian distribution parameters (mean, log_std)."""
        features = self.backbone(state)
        mean = self.mean_head(features)

        # Clamp log_std to avoid numerical instability during exponentiation
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=self.hp.log_std_min, max=self.hp.log_std_max)

        return mean, log_std

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()

        # Apply Tanh squashing
        y_t = torch.tanh(x_t)
        action = self.max_action * y_t

        # Corrected log-probability (dimension-wise)
        log_prob = normal.log_prob(x_t) - torch.log((1.0 - y_t.pow(2)).clamp(min=1e-6))

        # Optional: Add constant action scaling factor if exact likelihood value is needed
        if self.max_action != 1.0:
            log_prob -= torch.log(torch.tensor(self.max_action, device=state.device))

        # Sum across action dimensions -> shape (batch_size, 1)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = self.max_action * torch.tanh(mean)

        return action, log_prob, mean_action


class SACAgent:
    """Soft Actor-Critic Agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hp: Optional[SACHyperparameters] = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        if hp is None:
            hp = SACHyperparameters(action_dim=action_dim)
        self.hp = hp

        self.device = device

        # actor
        self.actor: SACActor = SACActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hp=hp,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.hp.learning_rate_actor)

        # critic
        self.critic: TwinQCritic = TwinQCritic(state_dim=state_dim,
                                               action_dim=action_dim,
                                               hidden_dim=N_HIDDEN_UNITS).to(self.device)
        self.critic_target: TwinQCritic = copy.deepcopy(self.critic)

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.hp.learning_rate_critic)

        # temperature
        self.log_alpha = torch.tensor(
            np.log(self.hp.alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        if self.hp.auto_temperature_tuning:
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=self.hp.learning_rate_alpha
            )
        else:
            self.alpha_optimizer = None

        # direct access to hyperparameters
        self.target_entropy = self.hp.target_entropy
        self.max_action = self.hp.max_action

        self.total_it = 0

    @property
    def alpha(self) -> Tensor:
        """Returns the current temperature value alpha = exp(log_alpha)."""
        return self.log_alpha.exp()

    def select_action(self, state: NDArray, is_deterministic: bool = False) -> NDArray:
        """Select action with optional deterministic setting."""
        state_tensor = torch.as_tensor(
            state.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            action, _, mean_action = self.actor.sample(state=state_tensor)

        selected_action = mean_action if is_deterministic else action
        return selected_action.cpu().data.numpy().flatten()

    def train(self, replay_buffer: ReplayBuffer, batch_size: Optional[int] = None) -> None:
        if batch_size is None:
            batch_size = self.hp.batch_size
        pass


