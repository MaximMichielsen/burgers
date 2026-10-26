import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as functional
from numpy.typing import NDArray
from torch import nn, Tensor

from ml.bandit.ann import BanditConfig, BanditHyperparameters, BanditANN
from ml.original.ann import TauANN
from ml.original.ann_config import TauANNConfig, TD3Hyperparameters
from ml.replay_buffer import ReplayBuffer
from proofing.contextual_bandit_reformulation.config import (
    TD3BanditHyperparameters,
)


class TwinQCritic(nn.Module):
    """Twin Q-Networks sized to match the TauANN hidden layer dimension."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Q1 architecture
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 architecture
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
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


class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient Agent."""

    def __init__(
        self,
        ann_config: TauANNConfig | BanditConfig,
        hp: TD3Hyperparameters
        | TD3BanditHyperparameters
        | BanditHyperparameters
        | None = None,
    ):
        self.config = ann_config
        self.device = torch.device("cpu")

        # Base TauANN wrapped into TD3 Policy
        if isinstance(ann_config, TauANNConfig):
            if hp is None:
                hp = TD3Hyperparameters()
            self.actor = TauANN(config=ann_config, hyperparams=hp).to(self.device)

        elif isinstance(ann_config, BanditConfig):
            self.actor = BanditANN(config=ann_config).to(self.device)
            if hp is None:
                hp = BanditHyperparameters()

        else:
            raise RuntimeError("blabla")

        self.hp = hp
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.lr)

        # Twin Q Critic
        self.critic = TwinQCritic(
            state_dim=ann_config.state_dimension,
            action_dim=ann_config.action_dimension,
            hidden_dim=ann_config.hidden_dimension,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.lr)

        self.total_it = 0
        self._last_actor_loss = None

    def select_action(self, state: NDArray, noise_std: float = 0.0) -> NDArray:
        """Select physical action with Gaussian exploration noise scaled by action span."""
        state_tensor = torch.as_tensor(
            state.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if noise_std > 0.0:
            act_span = self.config.max_action - self.config.min_action
            noise = np.random.normal(0, noise_std * act_span, size=action.shape)
            action = (action + noise).clip(
                self.config.min_action, self.config.max_action
            )

        return action

    def train(
        self, replay_buffer: ReplayBuffer, batch_size: Optional[int] = None
    ) -> tuple[float, float]:
        if batch_size is None:
            batch_size = self.hp.batch_size
        self.total_it += 1

        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Target policy smoothing scaled to physical action span
            act_span = self.config.max_action - self.config.min_action
            noise = (torch.randn_like(action) * self.hp.policy_noise * act_span).clamp(
                -self.hp.noise_clip * act_span,
                self.hp.noise_clip * act_span,
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                self.config.min_action, self.config.max_action
            )

            # Clipped double Q-learning (Critic receives physical actions directly)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done.float()) * self.hp.discount * target_q

        current_q1, current_q2 = self.critic(state, action)

        loss_fn = (
            functional.mse_loss if getattr(self.hp, "critic_mse", None) else functional.smooth_l1_loss
        )
        critic_loss = loss_fn(current_q1, target_q) + loss_fn(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss_val = getattr(self, "_last_actor_loss", 0.0)

        # Delayed Policy Updates
        if self.total_it % self.hp.policy_freq == 0:
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            actor_loss_val = float(actor_loss.item())
            self._last_actor_loss = actor_loss_val

            # Soft updates (Polyak averaging)
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.hp.tau_polyak * param.data
                    + (1.0 - self.hp.tau_polyak) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.hp.tau_polyak * param.data
                    + (1.0 - self.hp.tau_polyak) * target_param.data
                )

        return float(critic_loss.item()), actor_loss_val

    def evaluate_q_value(self, state: NDArray, action: NDArray) -> float:
        """Evaluate Q-value for physical state-action pair using main critic."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state.reshape(1, -1), dtype=torch.float32, device=self.device
            )
            action_tensor = torch.as_tensor(
                action.reshape(1, -1), dtype=torch.float32, device=self.device
            )
            q1, q2 = self.critic(state_tensor, action_tensor)
            q_val = torch.min(q1, q2).item()

        return float(q_val)
