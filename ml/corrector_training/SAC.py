import copy
import logging
from collections import deque
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn, Tensor, optim

from ml.ml_agents.corrector import AVController, AVCTrainingConfig, Transition

logger = logging.getLogger(__name__)



class ReplayBuffer:
    """Uniform-random experience replay buffer."""

    def __init__(self, capacity: int) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        """Add one transition."""
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return a random minibatch as float32 tensors."""
        indices = cast(
            NDArray, np.random.choice(len(self._buffer), size=batch_size, replace=False)
        )
        batch = [self._buffer[idx] for idx in indices]

        states_tensor = torch.tensor(
            np.array([t.state for t in batch]), dtype=torch.float32
        )
        actions_tensor = torch.tensor(
            np.array([t.action for t in batch]), dtype=torch.float32
        )  # shape (batch, action_dim): (batch, 1) global or (batch, N_nodes) local
        rewards_tensor = torch.tensor(
            np.array([[t.reward] for t in batch]), dtype=torch.float32
        )
        next_states_tensor = torch.tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float32
        )
        dones_tensor = torch.tensor(
            np.array([[float(t.done)] for t in batch]), dtype=torch.float32
        )
        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )

    def __len__(self) -> int:
        return len(self._buffer)



@dataclass
class SACConfig:
    """Hyperparameters for online SAC training.

    Attributes
    ----------
    gamma:
        Discount factor γ for the Bellman target.
    tau:
        Polyak soft target-network update coefficient.
    lr_actor, lr_critic, lr_alpha_temp:
        Learning rates for policy, twin Q-networks, and temperature.
    replay_capacity:
        Maximum transitions stored in the replay buffer.
    target_entropy:
        Desired policy entropy; -1.0 = -dim(A) for scalar action.
    critic_hidden_dim:
        Hidden layer width for the twin Q-networks.
    """

    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha_temp: float = 3e-4
    replay_capacity: int = 100_000
    target_entropy: float = -1.0
    critic_hidden_dim: int = 256
    tau_transient_warmup: float = 0.3


class _QNetwork(nn.Module):
    """Single Q(s, a) MLP approximator."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_input: Tensor, action_input: Tensor) -> Tensor:
        """Concatenate (s, a) and return Q-value."""
        sa_input = torch.cat([state_input, action_input], dim=-1)
        return self.network(sa_input)


class TwinQNetwork(nn.Module):
    """Twin Q-networks Q₁, Q₂ to reduce overestimation bias."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q1_network = _QNetwork(state_dim, action_dim, hidden_dim)
        self.q2_network = _QNetwork(state_dim, action_dim, hidden_dim)

    def forward(
        self, state_input: Tensor, action_input: Tensor
    ) -> tuple[Tensor, Tensor]:
        return (
            self.q1_network(state_input, action_input),
            self.q2_network(state_input, action_input),
        )

    def q1_only(self, state_input: Tensor, action_input: Tensor) -> Tensor:
        return self.q1_network(state_input, action_input)


class SACAgent:
    """Soft Actor–Critic agent for scalar continuous AV action.

    Actor  : AVCorrector policy (sigmoid-bounded → αₙ ∈ [0, αₘₐₓ]).
    Critic : TwinQNetwork (Q₁, Q₂) with soft target networks.
    Temp.  : Auto-tuned log-temperature to maintain target entropy.

    Parameters
    ----------
    av_corrector:
        The policy network πθ.
    state_dim:
        Dimension of sₙ (= K + 2).
    sac_config:
        Hyperparameter container.
    """

    def __init__(
        self,
        av_corrector: AVController,
        state_dim: int,
        sac_config: SACConfig,
        training_config: AVCTrainingConfig,
    ) -> None:
        self.policy = av_corrector
        self.sac_config = sac_config
        self.training_config = training_config

        action_dim = av_corrector.output_dimensions
        effective_target_entropy = (
            sac_config.target_entropy
            if sac_config.target_entropy != -1.0
            else -float(action_dim)
        )
        self._target_entropy = effective_target_entropy

        self._critic = TwinQNetwork(
            state_dim=state_dim,
            hidden_dim=sac_config.critic_hidden_dim,
            action_dim=action_dim,
        )
        self._critic_target = copy.deepcopy(self._critic)
        for param_tensor in self._critic_target.parameters():
            param_tensor.requires_grad = False

        self._actor_optimizer = optim.Adam(
            self.policy.parameters(), lr=sac_config.lr_actor
        )
        self._critic_optimizer = optim.Adam(
            self._critic.parameters(), lr=sac_config.lr_critic
        )

        self._log_alpha_temp: Tensor = torch.zeros(1, requires_grad=True)
        self._alpha_temp_optimizer = optim.Adam(
            [self._log_alpha_temp], lr=sac_config.lr_alpha_temp
        )

    @property
    def alpha_temp(self) -> Tensor:
        """Temperature scalar (always positive via exp)."""
        return self._log_alpha_temp.exp()

    def select_action(
        self, state_array: NDArray, *, deterministic: bool = False
    ) -> float:
        """Return αₙ for the given state sₙ.

        Returns shape (1,) for global or (N_nodes,) for local.
        """
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_tensor = self.policy(state_tensor).squeeze(0)  # (action_dim,)

        if deterministic:
            return action_tensor.numpy()

        noise_std = 0.05 * self.training_config.exploration_bound_upper
        noisy_action = action_tensor.numpy() + np.random.normal(
            0.0, noise_std, size=action_tensor.shape
        )
        logger.debug(
            "Selecting action:"
            f"| state tensor: {state_tensor} "
            f"| action_tensor: {action_tensor} "
            f"\n noisy action: {noisy_action}"
        )
        return np.clip(noisy_action, 0.0, self.training_config.exploration_bound_upper).astype(np.float32)

    def update(self, replay_buffer: ReplayBuffer) -> tuple[float, float, float]:
        """One SAC gradient step on critic, actor, and temperature.

        Returns
        -------
        critic_loss, actor_loss, alpha_temp : floats for logging.
        """
        (
            states_batch,
            actions_batch,
            rewards_batch,
            next_states_batch,
            dones_batch,
        ) = replay_buffer.sample(self.training_config.batch_size)

        # ---- Critic update ----
        with torch.no_grad():
            next_actions_batch = self.policy(next_states_batch)
            target_noise = (
                torch.randn_like(next_actions_batch) * 0.05 * self.policy.output_scale
            )
            next_actions_batch = (next_actions_batch + target_noise).clamp(
                0.0, self.training_config.exploration_bound_upper
            )
            q1_next_val, q2_next_val = self._critic_target(
                next_states_batch, next_actions_batch
            )
            q_next_min = torch.min(q1_next_val, q2_next_val)
            bellman_target = (
                rewards_batch + (1.0 - dones_batch) * self.sac_config.gamma * q_next_min
            )

        q1_pred_val, q2_pred_val = self._critic(states_batch, actions_batch)
        critic_loss_val = nn.functional.mse_loss(
            q1_pred_val, bellman_target
        ) + nn.functional.mse_loss(q2_pred_val, bellman_target)

        self._critic_optimizer.zero_grad()
        critic_loss_val.backward()
        self._critic_optimizer.step()

        # ---- Actor update ----
        predicted_actions_batch = self.policy(states_batch)
        q1_policy_val = self._critic.q1_only(states_batch, predicted_actions_batch)
        actor_loss_val = -q1_policy_val.mean()

        self._actor_optimizer.zero_grad()
        actor_loss_val.backward()
        self._actor_optimizer.step()

        # ---- Temperature update ----
        alpha_loss_val = -(
            self._log_alpha_temp * (actor_loss_val.detach() + self._target_entropy)
        )
        self._alpha_temp_optimizer.zero_grad()
        alpha_loss_val.backward()
        self._alpha_temp_optimizer.step()

        # ---- Polyak soft update of target networks ----
        tau_val = self.sac_config.tau
        for param_online, param_target in zip(
            self._critic.parameters(), self._critic_target.parameters()
        ):
            param_target.data.mul_(1.0 - tau_val)
            param_target.data.add_(tau_val * param_online.data)

        logger.debug(
            "Within agent.update()"
            f"\nq next_min: {q_next_min} "
            f"next_actions_batch: {next_actions_batch} "
            f"target_noise: {target_noise} "
            f"bellman target: {bellman_target} "
            f"q1_pred_val: {q1_pred_val} "
            f"q2_pred_val: {q2_pred_val}"
        )

        return (
            float(critic_loss_val.item()),
            float(actor_loss_val.item()),
            float(self.alpha_temp.item()),
        )
