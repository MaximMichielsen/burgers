import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn, Tensor
from torch.distributions import Normal
import torch.nn.functional as functional

from ml.constants import N_HIDDEN_UNITS
from ml.environment import EnvironmentTauAnn
from ml.projection_schedule import ProjectionReferenceSchedule
from ml.tau_ann import TauANN, TauANNConfig, save_tau_ann
from ml.training.shared_assets import TwinQCritic, ReplayBuffer
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem


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
    replay_buffer_max_size: int = int(1e6)

    def __post_init__(self):
        self.target_entropy = -float(self.action_dim)


# =============================================================================
# Actor
# =============================================================================


class SACActor(nn.Module):
    """Stochastic Gaussian Policy with Tanh Squashing for SAC."""

    def __init__(
        self,
        n_wavenumber_bins: int,
        n_coefficients: int,
        hp: Optional[SACHyperparameters] = None,
    ):
        super().__init__()

        if hp is None:
            hp = SACHyperparameters(action_dim=n_coefficients)
        self.hp = hp

        self.n_wavenumber_bins = n_wavenumber_bins
        self.action_dim = n_coefficients
        self.max_action = hp.max_action

        # base deployment policy
        self.tau_ann = TauANN(
            n_wavenumber_bins=n_wavenumber_bins,
            n_coefficients=n_coefficients,
            max_action=hp.max_action,
        )
        # feature extractor is all layers except final output projection
        self.backbone = self.tau_ann.network[:-1]

        # Output heads for Mean and Log Standard Deviation
        self.mean_head = self.tau_ann.network[-1]
        self.log_std_head = nn.Linear(N_HIDDEN_UNITS, n_coefficients)

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


# =============================================================================
# Agent
# =============================================================================


class SACAgent:
    """Soft Actor-Critic Agent."""

    def __init__(
        self,
        n_wavenumber_bins: int,
        n_coefficients: int,
        hp: Optional[SACHyperparameters] = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        if hp is None:
            hp = SACHyperparameters(action_dim=n_coefficients)
        self.hp = hp

        self.device = device
        self.state_dim = n_coefficients + n_wavenumber_bins

        # actor
        self.actor: SACActor = SACActor(
            n_wavenumber_bins=n_wavenumber_bins,
            n_coefficients=n_coefficients,
            hp=hp,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.hp.learning_rate_actor
        )

        # critic
        self.critic: TwinQCritic = TwinQCritic(
            state_dim=self.state_dim,
            action_dim=n_coefficients,
            hidden_dim=N_HIDDEN_UNITS,
        ).to(self.device)
        self.critic_target: TwinQCritic = copy.deepcopy(self.critic)

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.hp.learning_rate_critic
        )

        # temperature
        self.log_alpha = torch.tensor(
            np.log(self.hp.alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        if self.hp.auto_temperature_tuning:
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=self.hp.learning_rate_alpha
            )
        else:
            self.alpha_optimizer = None

        # direct access to hyperparameters
        self.polyak_tau = self.hp.polyak_tau
        self.discount = self.hp.discount
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

    def train(
        self, replay_buffer: ReplayBuffer, batch_size: Optional[int] = None
    ) -> None:
        if batch_size is None:
            batch_size = self.hp.batch_size
        self.total_it += 1

        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        # Soft target calculation
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            min_target_q = torch.min(target_q1, target_q2)

            target_q = reward + (1.0 - done) * self.discount * (
                min_target_q - self.alpha.detach() * next_log_prob
            )

        # critic optimization
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = functional.mse_loss(current_q1, target_q) + functional.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # policy optimization
        new_action, log_prob, _ = self.actor.sample(state)

        q1_new, q2_new = self.critic(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha.detach() * log_prob - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # temperature optimization
        if self.hp.auto_temperature_tuning and self.alpha_optimizer is not None:
            alpha_loss = -(
                self.log_alpha * (log_prob.detach() + self.target_entropy)
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # soft polyak target update
        with torch.no_grad():
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.mul_(1.0 - self.polyak_tau)
                target_param.data.add_(self.polyak_tau * param.data)


# =============================================================================
# Main Training Pipeline
# =============================================================================


def run_sac_tau_ann_training(
    problem: Problem,
    disc_config: DiscretizationConfig,
    tau_ann_config: TauANNConfig,
    master_path: Path,
    proj_ref_schedule: ProjectionReferenceSchedule,
    hp: SACHyperparameters | None = None,
) -> TauANN:
    """Main training loop connecting EnvironmentTauANN and SAC agent."""
    if hp is None:
        hp = SACHyperparameters(action_dim=tau_ann_config.n_coefficients)

    env = EnvironmentTauAnn(
        problem=problem,
        disc_config=disc_config,
        tau_ann_config=tau_ann_config,
        master_path=master_path,
        proj_ref_schedule=proj_ref_schedule,
    )

    state_dim = env.state_dim
    action_dim = env.action_dim

    agent = SACAgent(
        n_wavenumber_bins=tau_ann_config.n_wavenumber_bins,
        n_coefficients=tau_ann_config.n_coefficients,
        hp=hp,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    replay_buffer = ReplayBuffer(
        state_dim=state_dim, action_dim=action_dim, max_size=hp.replay_buffer_max_size
    )
    total_steps = 0

    for episode in range(hp.total_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        episode_steps = 0

        while not done:
            total_steps += 1
            episode_steps += 1

            if total_steps < hp.start_timesteps:
                action = np.random.uniform(
                    -hp.max_action, hp.max_action, size=action_dim
                )
            else:
                action = agent.select_action(state, is_deterministic=False)

            next_state, reward, done = env.step(action)

            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward

            if total_steps >= hp.start_timesteps:
                for _ in range(hp.utd_ratio):
                    agent.train(replay_buffer, hp.batch_size)

        print(
            f"Episode: {episode + 1}/{hp.total_episodes} | "
            f"Steps: {episode_steps} | "
            f"Total Steps: {total_steps} | "
            f"Reward: {episode_reward:.4f}"
        )

    save_tau_ann(agent.actor.tau_ann, tau_ann_config.ann_path)
    print(f"Successfully saved trained TauANN to {tau_ann_config.ann_path}")

    return agent.actor.tau_ann
