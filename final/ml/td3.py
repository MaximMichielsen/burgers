import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray
import torch.nn.functional as functional

from final.ml.environment import EnvironmentForcingDNS
from final.ml.hyperparameters import TD3Hyperparameters
from final.ml.reference_scheduler import ReferenceTrajectory
from final.ml.shared_assets import TwinQCritic, ReplayBuffer
from final.ml.tau_ann import TauANNConfig, TauANN
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem


class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient Agent."""

    def __init__(
        self, ann_config: TauANNConfig, hp: TD3Hyperparameters = TD3Hyperparameters()
    ):

        self.hp = hp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base TauANN wrapped into TD3 Policy
        self.actor = TauANN(config=ann_config, hyperparams=hp).to(self.device)
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

    def select_action(self, state: NDArray, noise_std: float = 0.0) -> NDArray:
        """Select action with optional Gaussian noise for exploration."""
        state_tensor = torch.as_tensor(
            state.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if noise_std > 0.0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = (action + noise).clip(0.0, self.hp.max_action)

        return action

    def train(
        self, replay_buffer: ReplayBuffer, batch_size: Optional[int] = None
    ) -> None:
        if batch_size is None:
            batch_size = self.hp.batch_size
        self.total_it += 1

        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(action) * self.hp.policy_noise).clamp(
                -self.hp.noise_clip, self.hp.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                0.0, self.hp.max_action
            )

            # Clipped double Q-learning
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.hp.discount * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = functional.smooth_l1_loss(
            current_q1, target_q
        ) + functional.smooth_l1_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Delayed Policy Updates
        if self.total_it % self.hp.policy_freq == 0:
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # Soft updates (Polyak averaging)
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.hp.tau_polyak * param.data
                    + (1 - self.hp.tau_polyak) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.hp.tau_polyak * param.data
                    + (1 - self.hp.tau_polyak) * target_param.data
                )


# =============================================================================
# Main Training Pipeline
# =============================================================================


class TD3Trainer:
    """Training wrapper for the TD3 training pipeline."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        ann_config: TauANNConfig,
        master_path: Path,
        reference_trajectory: ReferenceTrajectory,
        hp: TD3Hyperparameters = TD3Hyperparameters(),
    ):

        self.problem = problem
        self.disc_config = disc_config
        self.ann_config = ann_config
        self.master_path = master_path
        self.reference_trajectory = reference_trajectory
        self.hp = hp

        self.baseline_reward: float | None = None
        self.end_of_random_episode: int | None = None
        self.reward_history: list = []

        self.best_historical_reward = -np.inf
        self.best_action_sequence: list = []

    def run_training(self) -> TauANN:
        """Main training loop connecting the environment and TD3 agent."""

        # 1. Initialize environment
        env = EnvironmentForcingDNS(
            problem=self.problem,
            disc_config=self.disc_config,
            ann_config=self.ann_config,
            hyperparameters=self.hp,
            reference_trajectory=self.reference_trajectory,
            master_path=self.master_path,
        )

        # Instantiate agent & replay buffer
        agent = TD3Agent(
            ann_config=self.ann_config,
            hp=self.hp,
        )

        replay_buffer = ReplayBuffer(
            state_dim=self.ann_config.state_dimension,
            action_dim=self.ann_config.action_dimension,
            max_size=self.hp.replay_buffer_max_size,
        )

        print(
            f"Architecture Actor: {agent.actor.state_dimension} -> {agent.actor.hidden_dimension} x 3 -> {agent.actor.action_dimension}"
        )
        print(
            f"Architecture Critic: {agent.critic.state_dimension} -> {agent.critic.hidden_dimension} x 3 -> {agent.critic.action_dimension}"
        )

        total_steps = 0
        for episode in range(self.hp.total_episodes):
            state = env.reset()
            episode_reward = 0.0
            done = False
            episode_steps = 0
            actions = []

            while not done:
                total_steps += 1
                episode_steps += 1

                if total_steps < self.hp.stochastic_timesteps:
                    action = np.random.uniform(
                        self.hp.min_action,
                        self.hp.max_action,
                        size=self.ann_config.action_dimension,
                    )

                else:
                    action = agent.select_action(state, noise_std=self.hp.expl_noise)

            actions.append(action)