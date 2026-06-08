"""Online RL training for the AVCorrector using SAC.

The BurgersAVC solver acts as the environment.  At each control step n the
corrector policy πθ observes the MDP state

    sₙ = (Ê₁, …, Êₖ, ε⁻ⁿ, αₙ₋₁)  ∈ ℝ^(K+2)

and returns a scalar action αₙ ∈ [0, αₘₐₓ].  The solver is then advanced
Nₛₖᵢₚ LES timesteps under fixed αₙ (written directly into
solver.av_correction before each advance_time_step call), after which the
reward is computed from eq. (2.10) and a SAC gradient step is taken.

The environment wrapper bypasses BurgersAVC's own policy inference by using
correction_is_fixed=True and setting solver.av_correction externally before
each advance_time_step call.

SAC is chosen for online training because its entropy regularization
promotes exploration without manual noise schedules, and its off-policy
replay buffer makes sample use efficient (Haarnoja et al., 2018).

DNS reference targets
---------------------
By default the reward uses a single static DNS spectrum / dissipation value
(the terminal snapshot, as in the original implementation).  Passing a
``DNSReferenceSchedule`` via ``BurgersAVCEnvironment.dns_reference_schedule``
switches to a time-varying reference: the reward at control step n compares
the LES state against the DNS state at the same physical simulation time,
which is more meaningful for transient problems where energy decays
significantly over the simulation window.

References
----------
Haarnoja et al. (2018) "Soft Actor–Critic: Off-Policy Maximum Entropy Deep
    Reinforcement Learning with a Stochastic Actor."  arXiv:1801.01290.
Research Proposal §2.3.2, §3.1.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from torch import Tensor

from ml.corrector_training.DNS_snapshot_converter import DNSReferenceSchedule
from ml.ml_agents.corrector import AVCorrector, save_corrector
from problems_and_configurations.disc_config import DiscretisationConfig
from problems_and_configurations.problems import Problem
from problems_and_configurations.solver_configs import SGSPConfig, AVCConfig
from solvers.burgers_base import compute_adjusted_dt
from solvers.burgers_avc import BurgersAVC

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameter config
# ---------------------------------------------------------------------------


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
    batch_size:
        Minibatch size for gradient updates.
    warmup_steps:
        Random-action steps before policy gradient updates begin.
    update_every:
        Gradient update frequency (every N control steps).
    updates_per_step:
        Number of gradient steps taken each time an update is triggered.
    target_entropy:
        Desired policy entropy; -1.0 = -dim(A) for scalar action.
    n_skip_steps:
        LES timesteps per control interval Δtc = Nₛₖᵢₚ·Δt_LES.
    reward_weight_energy:
        w_E in eq. (2.10).
    reward_weight_dissipation:
        w_ε in eq. (2.10).
    reward_spectral_exponent:
        γ exponent per wavenumber in eq. (2.10); 5/3 for inertial range.
    critic_hidden_dim:
        Hidden layer width for the twin Q-networks.
    """

    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha_temp: float = 3e-4
    replay_capacity: int = 100_000
    batch_size: int = 256
    warmup_steps: int = 100
    update_every: int = 1
    updates_per_step: int = 1
    target_entropy: float = -1.0
    n_skip_steps: int = 5
    reward_weight_energy: float = 1.0
    reward_weight_dissipation: float = 0.1
    reward_spectral_exponent: float = 5.0 / 3.0
    critic_hidden_dim: int = 256


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class Transition(NamedTuple):
    """Single MDP transition (sₙ, αₙ, rₙ, sₙ₊₁, done)."""

    state: NDArray
    action: NDArray
    reward: float
    next_state: NDArray
    done: bool


class ReplayBuffer:
    """Uniform-random experience replay buffer."""

    def __init__(self, capacity: int) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        """Add one transition."""
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return a random minibatch as float32 tensors."""
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
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


# ---------------------------------------------------------------------------
# Twin Q-network (critic)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MDP environment wrapper around BurgersAVC
# ---------------------------------------------------------------------------


class BurgersAVCEnvironment:
    """MDP wrapper around BurgersAVC for the AV corrector control problem."""

    def __init__(
        self,
        problem: Problem,
        disc_cfg: DiscretisationConfig,
        sgsp_cfg: SGSPConfig,
        avc_cfg: AVCConfig,
        master_path: Path,
        sac_config: SACConfig,
        dns_reference_schedule: DNSReferenceSchedule | None = None,
    ) -> None:
        self._problem = problem
        self._disc_cfg = disc_cfg
        self._sgsp_cfg = sgsp_cfg
        self._avc_cfg = avc_cfg
        self._master_path = master_path
        self._sac_config = sac_config
        self._dns_reference_schedule = dns_reference_schedule

        self.exclude_diss_from_reward = avc_cfg.exclude_diss_from_reward

        self._solver: BurgersAVC | None = None
        self._total_les_steps: int = 0

        _, self._n_time_steps = compute_adjusted_dt(
            disc_cfg.dt_les, problem.domain_timespan
        )
        self._max_les_steps: int = self._n_time_steps

        self.n_wavenumber_bins: int = len(avc_cfg.dns_energy_spectrum)
        self.state_dim: int = self.n_wavenumber_bins + 2

    def reset(self) -> NDArray:
        """Instantiate a fresh BurgersAVC solver and return initial state sₙ."""
        self._solver = BurgersAVC(
            problem=self._problem,
            disc_cfg=dataclasses.replace(self._disc_cfg, suppress_file_logging=True),
            simulation_mode="avc",
            master_path=self._master_path,
            sgsp_cfg=self._sgsp_cfg,
            avc_cfg=self._avc_cfg,
        )
        self._solver.use_policy_inference = (
            False  # trainer drives av_correction externally
        )
        self._total_les_steps = 0
        self._correction_mode = self._solver._corrector.correction_mode
        self._n_output_nodes = self._solver._corrector.output_dim
        return self._solver._create_avc_input_stencil()

    def step(self, alpha_action: float) -> tuple[NDArray, float, bool]:
        """Set αₙ, advance Nₛₖᵢₚ LES steps, return (sₙ₊₁, rₙ, done)."""
        assert self._solver is not None, "Call reset() before step()."

        alpha_array = np.asarray(alpha_action, dtype=np.float64)
        if self._correction_mode == "local":
            self._solver.av_correction = alpha_array
        else:
            self._solver.av_correction = float(alpha_array.item())

        blown_up = False
        for _ in range(self._sac_config.n_skip_steps):
            step_ok = self._solver.advance_time_step()
            self._total_les_steps += 1
            if not step_ok:
                blown_up = True
                break

        reward_val = self._compute_reward(blown_up=blown_up)
        done_flag = blown_up or self._total_les_steps >= self._max_les_steps
        next_state_array = self._solver._create_avc_input_stencil()

        return next_state_array, reward_val, done_flag

    def _get_dns_targets(self) -> tuple[NDArray, float]:
        """Return (dns_spectrum_k, dns_dissipation) for the current solver time."""
        assert self._solver is not None

        if self._dns_reference_schedule is not None:
            current_time = self._solver.simulation_time_elapsed
            return self._dns_reference_schedule.query(current_time)

        return (
            np.asarray(self._avc_cfg.dns_energy_spectrum, dtype=np.float64),
            float(self._avc_cfg.dns_dissipation),
        )

    def _compute_reward(self, blown_up: bool) -> float:
        """Compute rₙ from eq. (2.10); large terminal penalty on blow-up."""
        if blown_up:
            return -1.0

        assert self._solver is not None

        wavenumbers_all, raw_spectrum_all = self._solver.compute_energy_spectrum(
            self._solver.solution
        )
        _, positive_spectrum = self._solver.get_positive_spectrum(
            wavenumbers_all, raw_spectrum_all
        )
        spectrum_k = positive_spectrum[: self.n_wavenumber_bins].astype(np.float64)
        dns_spectrum_k, dns_dissipation = self._get_dns_targets()

        wavenumber_indices = np.arange(1, self.n_wavenumber_bins + 1, dtype=np.float64)
        w_e = self._sac_config.reward_weight_energy
        w_eps = self._sac_config.reward_weight_dissipation
        gamma_exp = self._sac_config.reward_spectral_exponent

        # Normalise spectra by total energy to compare shape only
        normalised_les = spectrum_k / max(spectrum_k.sum(), 1e-12)
        normalised_dns = dns_spectrum_k / max(dns_spectrum_k.sum(), 1e-12)

        compensated_les = wavenumber_indices**gamma_exp * normalised_les
        compensated_dns = wavenumber_indices**gamma_exp * normalised_dns

        dns_safe = np.where(compensated_dns > 0.0, compensated_dns, 1.0)
        spectral_penalty = float(
            w_e * np.mean(((compensated_les - compensated_dns) / dns_safe) ** 2)
        )

        if self.exclude_diss_from_reward:
            return -(spectral_penalty / (1.0 + spectral_penalty))

        current_dissipation = (
            self._solver.dissipation_history[-1]
            if self._solver.dissipation_history
            else 0.0
        )
        current_av_drain = (
            self._solver.energy_drain_history[-1]
            if self._solver.energy_drain_history
            else 0.0
        )

        dns_diss_safe = max(abs(dns_dissipation), 1e-12)
        dissipation_penalty = float(
            w_eps
            * (
                (current_dissipation + current_av_drain - dns_dissipation)
                / dns_diss_safe
            )
            ** 2
        )

        total_penalty = spectral_penalty + dissipation_penalty
        return -(total_penalty / (1.0 + total_penalty))


# ---------------------------------------------------------------------------
# SAC agent
# ---------------------------------------------------------------------------


@dataclass
class TrainingStats:
    """Per-episode and per-update diagnostics."""

    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    critic_losses: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    alpha_temp_values: list[float] = field(default_factory=list)
    total_env_steps: int = 0


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
        av_corrector: AVCorrector,
        state_dim: int,
        sac_config: SACConfig,
    ) -> None:
        self._policy = av_corrector
        self._config = sac_config

        action_dim = av_corrector.output_dim
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
            self._policy.parameters(), lr=sac_config.lr_actor
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
    ) -> NDArray:
        """Return αₙ for the given state sₙ.

        Returns shape (1,) for global or (N_nodes,) for local.
        """
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_tensor = self._policy(state_tensor).squeeze(0)  # (action_dim,)

        if deterministic:
            return action_tensor.numpy()

        noise_std = 0.05 * self._policy.alpha_max
        noisy_action = action_tensor.numpy() + np.random.normal(
            0.0, noise_std, size=action_tensor.shape
        )
        return np.clip(noisy_action, 0.0, self._policy.alpha_max).astype(np.float32)

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
        ) = replay_buffer.sample(self._config.batch_size)

        # ---- Critic update ----
        with torch.no_grad():
            next_actions_batch = self._policy(next_states_batch)
            target_noise = (
                torch.randn_like(next_actions_batch) * 0.05 * self._policy.alpha_max
            )
            next_actions_batch = (next_actions_batch + target_noise).clamp(
                0.0, self._policy.alpha_max
            )
            q1_next_val, q2_next_val = self._critic_target(
                next_states_batch, next_actions_batch
            )
            q_next_min = torch.min(q1_next_val, q2_next_val)
            bellman_target = (
                rewards_batch + (1.0 - dones_batch) * self._config.gamma * q_next_min
            )

        q1_pred_val, q2_pred_val = self._critic(states_batch, actions_batch)
        critic_loss_val = nn.functional.mse_loss(
            q1_pred_val, bellman_target
        ) + nn.functional.mse_loss(q2_pred_val, bellman_target)

        self._critic_optimizer.zero_grad()
        critic_loss_val.backward()
        self._critic_optimizer.step()

        # ---- Actor update ----
        predicted_actions_batch = self._policy(states_batch)
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
        tau_val = self._config.tau
        for param_online, param_target in zip(
            self._critic.parameters(), self._critic_target.parameters()
        ):
            param_target.data.mul_(1.0 - tau_val)
            param_target.data.add_(tau_val * param_online.data)

        return (
            float(critic_loss_val.item()),
            float(actor_loss_val.item()),
            float(self.alpha_temp.item()),
        )


# ---------------------------------------------------------------------------
# Online trainer
# ---------------------------------------------------------------------------


class OnlineAVTrainer:
    """Online SAC training loop for the AV corrector.

    Drives BurgersAVCEnvironment episode-by-episode: collects transitions
    into a replay buffer and triggers SAC updates after each control step.

    Parameters
    ----------
    environment:
        The wrapped BurgersAVC solver environment.
    sac_agent:
        Initialised SACAgent (policy + critics).
    sac_config:
        Shared hyperparameter config.
    output_dir:
        Directory for policy checkpoints.
    """

    def __init__(
        self,
        environment: BurgersAVCEnvironment,
        sac_agent: SACAgent,
        sac_config: SACConfig,
        output_dir: Path,
    ) -> None:
        self._env = environment
        self._agent = sac_agent
        self._config = sac_config
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._replay_buffer = ReplayBuffer(capacity=sac_config.replay_capacity)
        self._stats = TrainingStats()

    def train(self, n_episodes: int, checkpoint_every: int = 50) -> TrainingStats:
        """Run n_episodes of online SAC training.

        Parameters
        ----------
        n_episodes:
            Total solver episodes (each resets to a fresh simulation).
        checkpoint_every:
            Save a policy checkpoint every this many episodes.

        Returns
        -------
        TrainingStats with per-episode and per-update diagnostics.
        """
        using_schedule = self._env._dns_reference_schedule is not None
        print(
            f"\nOnline SAC training — {n_episodes} episodes "
            f"| warmup: {self._config.warmup_steps} steps "
            f"| Nₛₖᵢₚ: {self._config.n_skip_steps} "
            f"| DNS ref: {'time-varying' if using_schedule else 'static terminal'}"
        )
        print("-" * 64)

        for episode_idx in range(n_episodes):
            state_current = self._env.reset()
            episode_reward_total = 0.0
            episode_step_count = 0
            done_flag = False

            while not done_flag:
                if self._stats.total_env_steps < self._config.warmup_steps:
                    random_scalar = float(
                        np.random.uniform(0.0, self._agent._policy.alpha_max)
                    )
                    action_dim = self._agent._policy.output_dim
                    alpha_action_val = np.full(
                        action_dim, random_scalar, dtype=np.float32
                    )
                else:
                    alpha_action_val = self._agent.select_action(state_current)

                next_state_array, reward_val, done_flag = self._env.step(
                    alpha_action_val
                )

                self._replay_buffer.push(
                    Transition(
                        state=state_current,
                        action=alpha_action_val,
                        reward=reward_val,
                        next_state=next_state_array,
                        done=done_flag,
                    )
                )

                state_current = next_state_array
                episode_reward_total += reward_val
                episode_step_count += 1
                self._stats.total_env_steps += 1

                enough_samples = len(self._replay_buffer) >= self._config.batch_size
                past_warmup = self._stats.total_env_steps >= self._config.warmup_steps
                update_due = (
                    self._stats.total_env_steps % self._config.update_every == 0
                )

                if enough_samples and past_warmup and update_due:
                    for _ in range(self._config.updates_per_step):
                        critic_loss_val, actor_loss_val, alpha_temp_val = (
                            self._agent.update(self._replay_buffer)
                        )
                        self._stats.critic_losses.append(critic_loss_val)
                        self._stats.actor_losses.append(actor_loss_val)
                        self._stats.alpha_temp_values.append(alpha_temp_val)

            self._stats.episode_rewards.append(episode_reward_total)
            self._stats.episode_lengths.append(episode_step_count)

            if episode_idx % 10 == 0 or episode_idx == n_episodes - 1:
                recent_mean = float(np.mean(self._stats.episode_rewards[-10:]))
                print(
                    f"Episode {episode_idx:04d} | "
                    f"Return: {episode_reward_total:+.2f} | "
                    f"Mean(10): {recent_mean:+.2f} | "
                    f"Control steps: {episode_step_count} | "
                    f"Buffer: {len(self._replay_buffer)}"
                )

            if (episode_idx + 1) % checkpoint_every == 0:
                self._save_checkpoint(episode_idx=episode_idx)

        self._save_checkpoint(episode_idx=n_episodes - 1, tag="final")
        print(f"\nTraining complete. Checkpoints saved to '{self._output_dir}'.")
        return self._stats

    def _save_checkpoint(self, episode_idx: int, tag: str | None = None) -> None:
        """Save policy weights to a .pt file."""
        suffix = tag if tag else f"ep{episode_idx:04d}"
        checkpoint_path = self._output_dir / f"av_corrector_{suffix}.pt"
        save_corrector(self._agent._policy, checkpoint_path)
        logger.info("Checkpoint saved to %s", checkpoint_path)
