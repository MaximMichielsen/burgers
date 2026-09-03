import copy
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn, Tensor
import torch.nn.functional as functional

from ml.constants import N_HIDDEN_UNITS
from ml.environment import EnvironmentTauAnn
from ml.projection_schedule import ProjectionReferenceSchedule
from ml.tau_ann import TauANN, TauANNConfig, save_tau_ann
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem


# =============================================================================
# Network Architectures & Adapters
# =============================================================================


class TD3ActorWrapper(nn.Module):
    """Wraps existing TauANN model to enforce action bounds via tanh."""

    def __init__(self, tau_ann_model: TauANN, max_action: float = 1.0):
        super().__init__()
        self.tau_ann = tau_ann_model
        self.max_action = max_action

    def forward(self, state: Tensor) -> Tensor:
        """Pass through TauANN, then bound output to [-max_action, max_action]."""
        raw_output = self.tau_ann(state)
        return self.max_action * torch.tanh(raw_output)


class TauANNCritic(nn.Module):
    """Twin Q-Networks sized to match the TauANN hidden layer dimension."""

    def __init__(
        self, input_dim: int, action_dim: int, hidden_dim: int = N_HIDDEN_UNITS
    ):
        super().__init__()

        # Q1 architecture
        self.q1_net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Q2 architecture
        self.q2_net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: Tensor, action: Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa), self.q2_net(sa)

    def q1(self, state: Tensor, action: Tensor) -> Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa)


# =============================================================================
# Replay Buffer
# =============================================================================


class ReplayBuffer:
    """Experience replay memory storing transitions and returning Tensors."""

    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# =============================================================================
# 3. TD3 Agent
# =============================================================================


class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient Agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_wavenumber_bins: int,
        max_action: float = 1.0,
        discount: float = 0.99,
        tau_polyak: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        lr: float = 3e-4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base TauANN wrapped into TD3 Policy
        base_tau_ann = TauANN(
            n_wavenumber_bins=n_wavenumber_bins, n_coefficients=action_dim
        )
        self.actor = TD3ActorWrapper(base_tau_ann, max_action=max_action).to(
            self.device
        )
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Twin Q Critic
        self.critic = TauANNCritic(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=N_HIDDEN_UNITS
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.discount = discount
        self.tau_polyak = tau_polyak
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state: NDArray, noise_std: float = 0.0) -> NDArray:
        """Select action with optional Gaussian noise for exploration."""
        state_tensor = torch.as_tensor(
            state.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if noise_std > 0.0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = (action + noise).clip(-self.max_action, self.max_action)

        return action

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> None:
        self.total_it += 1

        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Clipped double Q-learning
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.discount * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = functional.mse_loss(current_q1, target_q) + functional.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed Policy Updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft updates (Polyak averaging)
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau_polyak * param.data
                    + (1 - self.tau_polyak) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau_polyak * param.data
                    + (1 - self.tau_polyak) * target_param.data
                )


# =============================================================================
# Main Training Pipeline
# =============================================================================


def run_td3_tau_ann_training(
    problem: Problem,
    disc_config: DiscretizationConfig,
    tau_ann_config: TauANNConfig,
    master_path: Path,
    proj_ref_schedule: ProjectionReferenceSchedule,
    total_episodes: int = 100,
    max_action: float = 1.0,
    start_timesteps: int = 1000,
    batch_size: int = 64,
    expl_noise: float = 0.1,
) -> TauANN:
    """Main training loop connecting EnvironmentTauANN and TD3 Agent."""

    # 1. Initialize environment
    env = EnvironmentTauAnn(
        problem=problem,
        disc_config=disc_config,
        tau_ann_config=tau_ann_config,
        master_path=master_path,
        proj_ref_schedule=proj_ref_schedule,
    )

    state_dim = env.state_dim
    action_dim = env.action_dim
    n_wavenumber_bins = env.n_wavenumber_bins

    # Instantiate agent & replay buffer
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        n_wavenumber_bins=n_wavenumber_bins,
        max_action=max_action,
        discount=0.99,
        tau_polyak=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        lr=3e-4,
    )

    replay_buffer = ReplayBuffer(
        state_dim=state_dim, action_dim=action_dim, max_size=int(1e5)
    )

    total_steps = 0

    # 3. Outer episode training loop
    for episode in range(total_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        episode_steps = 0

        while not done:
            total_steps += 1
            episode_steps += 1

            # Select Action: Pure random uniforms at start, then policy + noise
            if total_steps < start_timesteps:
                action = np.array(
                    np.random.uniform(-max_action, max_action, size=action_dim)
                )
            else:
                action = agent.select_action(state, noise_std=expl_noise)

            next_state, reward, done = env.step(action=action)
            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward

            # Train TD3 agent once warmup phase is complete
            if total_steps >= start_timesteps:
                agent.train(replay_buffer, batch_size)

        print(
            f"Episode: {episode + 1}/{total_episodes} | "
            f"Steps in Ep: {episode_steps} | "
            f"Total Steps: {total_steps} | "
            f"Reward: {episode_reward:.4f}"
        )

    # 4. Extract trained core TauANN and save to disk
    trained_tau_ann = agent.actor.tau_ann
    save_tau_ann(trained_tau_ann, tau_ann_config.ann_path)
    print(f"Successfully saved trained TauANN to {tau_ann_config.ann_path}")

    return trained_tau_ann
