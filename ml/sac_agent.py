import copy
from typing import Optional

import torch
import torch.nn.functional as functional
from numpy.typing import NDArray
from torch import nn, Tensor

from ml.bandit.ann import BanditConfig
from ml.mdp.ann import MDPConfig
from ml.original.ann_config import TauANNConfig, TD3Hyperparameters
from ml.replay_buffer import ReplayBuffer
from ml.td3_agent import TwinQCritic


class SACActor(nn.Module):
    """SAC Actor (Policy) network outputting mean and log standard deviation for reparameterized actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick (Gaussian)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sample

        # Enforce action bounds using tanh squashing (standard SAC practice)
        action = torch.tanh(x_t)

        # Calculate log probability with Jacobian correction for tanh squashing
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1.0e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean


class SACAgent:
    """Soft Actor-Critic (SAC) Agent with automatic entropy tuning."""

    def __init__(
        self,
        ann_config: TauANNConfig | BanditConfig | MDPConfig,
        hp: Optional[object] = None,
    ):
        self.config = ann_config
        self.device = torch.device("cpu")

        # Initialize hyperparameters (fallback to sensible defaults if none provided)
        if hp is None:
            hp = TD3Hyperparameters()  # or custom SAC hyperparameters container
        self.hp = hp

        state_dim = ann_config.state_dimension
        action_dim = ann_config.action_dimension
        hidden_dim = ann_config.hidden_dimension

        # Actor Network
        self.actor = SACActor(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=getattr(hp, "lr", 3e-4)
        )

        # Twin Q-Critics (same architecture as TD3)
        self.critic = TwinQCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=getattr(hp, "lr", 3e-4)
        )

        # Automatic Entropy Tuning (Alpha)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=getattr(hp, "lr", 3e-4)
        )

        self.total_it = 0

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def select_action(
        self, state: NDArray, evaluate: bool = False, noise_std=None
    ) -> NDArray:
        """Select action using deterministic mode (mean) for evaluation or stochastic sampling for exploration."""
        state_tensor = torch.as_tensor(
            state.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            if evaluate:
                _, _, mean = self.actor.sample(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(state_tensor)

            # Scale to physical action space if min/max bounds are present
            action_np = action.cpu().data.numpy().flatten()
            if hasattr(self.config, "min_action") and hasattr(
                self.config, "max_action"
            ):
                act_span = (self.config.max_action - self.config.min_action) / 2.0
                act_center = (self.config.max_action + self.config.min_action) / 2.0
                action_np = action_np * act_span + act_center

        return action_np

    def train(
        self, replay_buffer: ReplayBuffer, batch_size: Optional[int] = None
    ) -> tuple[float, float, float]:
        if batch_size is None:
            batch_size = getattr(self.hp, "batch_size", 256)
        self.total_it += 1

        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        # Rescale actions back to normalized [-1, 1] range if needed by tanh space
        if hasattr(self.config, "min_action") and hasattr(self.config, "max_action"):
            act_span = (self.config.max_action - self.config.min_action) / 2.0
            act_center = (self.config.max_action + self.config.min_action) / 2.0
            action = (action - act_center) / act_span

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_state_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_state_log_pi

            discount = getattr(self.hp, "discount", 0.99)
            target_q = reward + (1.0 - done.float()) * discount * target_q

        # --- Critic Loss & Update ---
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = functional.mse_loss(current_q1, target_q) + functional.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --- Actor Loss & Update ---
        pi, log_pi, _ = self.actor.sample(state)
        actor_q1, actor_q2 = self.critic(state, pi)
        actor_q = torch.min(actor_q1, actor_q2)

        actor_loss = (self.alpha.detach() * log_pi - actor_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # --- Alpha (Entropy Temperature) Loss & Update ---
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft Target Updates (Polyak Averaging) ---
        tau = getattr(self.hp, "tau_polyak", 0.005)
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        return (
            float(critic_loss.item()),
            float(actor_loss.item()),
            float(self.alpha.item()),
        )

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
