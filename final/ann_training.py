import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from numpy.typing import NDArray

# Package imports
from final.ann_coupling.new_environment import EnvironmentForcingDNS
from final.ann_coupling.reference_scheduler import ReferenceTrajectory
from ml.tau_ann import (
    EPISODE_REWARD_CLIP,
    N_HIDDEN_UNITS,
    TauANN,
    TauANNConfig,
    save_tau_ann,
)
from ml.training.shared_assets import ReplayBuffer, TwinQCritic
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem

# =============================================================================
# Hyperparameters
# =============================================================================


@dataclass
class TD3Hyperparameters:
    """Hyperparameters for TD3 Agent training and environment interactions."""

    # Agent / Optimization Params
    lr: float = 1e-4
    discount: float = 0.99
    tau_polyak: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    max_action: float = 1.0

    # Training / Environment Setup Params
    total_episodes: int = 100
    stochastic_timesteps: int = 1200
    reduced_parameter_timesteps: int = 800
    batch_size: int = 64
    expl_noise: float = 0.1
    replay_buffer_max_size: int = int(1e5)

    @property
    def start_timesteps(self) -> int:
        return self.stochastic_timesteps + self.reduced_parameter_timesteps


# =============================================================================
# Helper Utilities
# =============================================================================


def _unpack_env_step(
    step_result: Tuple[Any, ...],
) -> Tuple[NDArray, float, bool, Dict[str, Any]]:
    """Unpack environment step outputs consistently for 3, 4, or 5 item tuples."""
    if len(step_result) == 5:
        next_state, reward, terminated, truncated, info = step_result
        return (
            np.asarray(next_state),
            float(reward),
            bool(terminated or truncated),
            info,
        )
    elif len(step_result) == 4:
        next_state, reward, done, info = step_result
        return np.asarray(next_state), float(reward), bool(done), info
    elif len(step_result) == 3:
        next_state, reward, done = step_result
        return np.asarray(next_state), float(reward), bool(done), {}
    else:
        raise ValueError(f"Unexpected step result format of length {len(step_result)}")


def _unpack_env_reset(
    reset_result: Union[NDArray, Tuple[NDArray, Dict[str, Any]]],
) -> NDArray:
    """Unpack environment reset outputs whether returning state or (state, info)."""
    if isinstance(reset_result, tuple):
        return np.asarray(reset_result[0])
    return np.asarray(reset_result)


# =============================================================================
# TD3 Agent
# =============================================================================


class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient Agent."""

    def __init__(
        self,
        ann_config: TauANNConfig,
        hp: Optional[TD3Hyperparameters] = None,
    ):
        self.hp = hp if hp is not None else TD3Hyperparameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = TauANN(config=ann_config).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.hp.lr)

        self.critic = TwinQCritic(
            state_dim=ann_config.state_dimension,
            action_dim=ann_config.action_dimension,
            hidden_dim=N_HIDDEN_UNITS,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.hp.lr
        )

        self.total_it = 0

    def select_action(self, state: NDArray, noise_std: float = 0.0) -> NDArray:
        """Select action with optional Gaussian noise for exploration."""
        state_tensor = torch.as_tensor(
            state.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        self.actor.train()

        if noise_std > 0.0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = (action + noise).clip(0.0, self.hp.max_action)

        return action

    def train(
        self, replay_buffer: ReplayBuffer, batch_size: Optional[int] = None
    ) -> Tuple[float, float]:
        if batch_size is None:
            batch_size = self.hp.batch_size
        self.total_it += 1

        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            next_state = torch.as_tensor(
                next_state, dtype=torch.float32, device=self.device
            )
            reward = torch.as_tensor(
                reward, dtype=torch.float32, device=self.device
            ).reshape(-1, 1)
            done = torch.as_tensor(
                done, dtype=torch.float32, device=self.device
            ).reshape(-1, 1)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.hp.policy_noise).clamp(
                -self.hp.noise_clip, self.hp.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                0.0, self.hp.max_action
            )

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.hp.discount * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.smooth_l1_loss(current_q1, target_q) + F.smooth_l1_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss_val = 0.0
        if self.total_it % self.hp.policy_freq == 0:
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            actor_loss_val = actor_loss.item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

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

        return critic_loss.item(), actor_loss_val


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
        reference_schedule: ReferenceTrajectory,
        hp: Optional[TD3Hyperparameters] = None,
    ):
        self.problem = problem
        self.disc_config = disc_config
        self.ann_config = ann_config
        self.master_path = Path(master_path)
        self.reference_schedule = reference_schedule

        self.hp = hp if hp is not None else TD3Hyperparameters()

        self.baseline_reward: Optional[float] = None
        self.end_of_random_episode: Optional[int] = None
        self.reward_history: List[float] = []

        self.best_historical_reward = -np.inf
        self.best_action_sequence: List[NDArray] = []
        self.best_actor_weights: Optional[Dict[str, torch.Tensor]] = None

    def run_td3_tau_ann_training(self) -> Tuple[TauANN, List[NDArray]]:
        """Main training loop connecting EnvironmentForcingDNS and TD3 Agent."""

        print(
            f"Architecture: {self.ann_config.state_dimension} -> "
            f"{self.ann_config.hidden_dimension} x 3 -> "
            f"{self.ann_config.action_dimension}"
        )

        env = EnvironmentForcingDNS(
            problem=self.problem,
            disc_config=self.disc_config,
            ann_config=self.ann_config,
            master_path=self.master_path,
            reference_schedule=self.reference_schedule,
        )

        agent = TD3Agent(
            ann_config=self.ann_config,
            hp=self.hp,
        )

        stochastic_mode = ""

        # Baseline evaluation
        state = _unpack_env_reset(env.reset())
        actual_state_dim = state.shape[0]
        baseline_episode_reward = 0.0
        baseline_action = np.ones(self.ann_config.action_dimension, dtype=np.float32)
        baseline_done = False

        replay_buffer = ReplayBuffer(
            state_dim=actual_state_dim,
            action_dim=self.ann_config.action_dimension,
            max_size=self.hp.replay_buffer_max_size,
        )

        while not baseline_done:
            step_res = env.step(action=baseline_action)
            _, reward, baseline_done, _ = _unpack_env_step(step_res)
            baseline_episode_reward += reward

        self.baseline_reward = baseline_episode_reward
        print(f"Computed Baseline Episode Reward: {self.baseline_reward:.2f}")

        total_steps = 0

        # Training Loop
        for episode in range(self.hp.total_episodes):
            state = _unpack_env_reset(env.reset())
            episode_reward = 0.0
            done = False
            episode_steps = 0
            actions = []

            while not done:
                total_steps += 1
                episode_steps += 1

                if total_steps < self.hp.stochastic_timesteps:
                    if total_steps < self.hp.stochastic_timesteps / 2:
                        stochastic_mode = "random narrow"
                        high_action = min(1.2, self.hp.max_action)
                        low_action = 0.8
                        action = np.random.uniform(
                            low_action,
                            high_action,
                            size=self.ann_config.action_dimension,
                        ).astype(np.float32)
                    else:
                        stochastic_mode = "random wide"
                        action = np.random.uniform(
                            0.0,
                            self.hp.max_action,
                            size=self.ann_config.action_dimension,
                        ).astype(np.float32)

                elif total_steps < self.hp.start_timesteps:
                    if self.ann_config.action_dimension > 2:
                        stochastic_mode = "reduced parameters"
                        action = np.random.uniform(
                            0.0,
                            self.hp.max_action,
                            size=self.ann_config.action_dimension,
                        ).astype(np.float32)
                        action[2:] = 0.0
                    else:
                        stochastic_mode = "random wide"
                        action = np.random.uniform(
                            0.0,
                            self.hp.max_action,
                            size=self.ann_config.action_dimension,
                        ).astype(np.float32)

                else:
                    if self.end_of_random_episode is None:
                        self.end_of_random_episode = episode
                    stochastic_mode = "deterministic"
                    action = agent.select_action(state, noise_std=self.hp.expl_noise)

                actions.append(action)
                step_res = env.step(action=action)
                next_state, reward, done, _ = _unpack_env_step(step_res)

                replay_buffer.add(state, action, next_state, reward, done)

                state = next_state
                episode_reward += reward

                if total_steps >= self.hp.start_timesteps:
                    agent.train(replay_buffer, self.hp.batch_size)

            episode_reward = float(np.clip(episode_reward, EPISODE_REWARD_CLIP, 0.0))
            self.reward_history.append(episode_reward)

            if episode_reward > self.best_historical_reward:
                self.best_historical_reward = episode_reward
                self.best_action_sequence = actions
                self.best_actor_weights = copy.deepcopy(agent.actor.state_dict())

            scope_val = getattr(
                self.ann_config.output_scope, "value", str(self.ann_config.output_scope)
            )
            n_groups = getattr(self.ann_config, "n_local_action_groups", 1)

            print(
                f"Episode: {episode + 1}/{self.hp.total_episodes} | "
                f"Steps in Ep: {episode_steps} | "
                f"Total Steps: {total_steps} | "
                f"Stochastic Mode: {stochastic_mode} | "
                f"Scope: {scope_val} ({n_groups}) | "
                f"Reward: {episode_reward:.4f} | "
                f"Baseline: {self.baseline_reward:.2f}"
            )

        assert self.ann_config.ann_path is not None
        if self.best_actor_weights is not None:
            agent.actor.load_state_dict(self.best_actor_weights)

        save_tau_ann(model=agent.actor, save_path=self.ann_config.ann_path)
        print(f"Successfully saved trained TauANN to {self.ann_config.ann_path}")

        return agent.actor, self.best_action_sequence

    def plot_reward_evolution(self, show_plot: bool = False) -> None:
        """Visualize the evolution of episode rewards over training with an inset zoom."""
        if not self.reward_history:
            print("No reward history found to plot.")
            return

        episodes = np.arange(len(self.reward_history))
        rewards = np.array(self.reward_history)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300, layout="constrained")

        if self.end_of_random_episode is not None:
            ax.axvspan(
                0,
                self.end_of_random_episode,
                color="#d9e2ec",
                alpha=0.5,
                label="Random Exploration Phase",
            )
            ax.axvline(
                x=self.end_of_random_episode,
                color="gray",
                linestyle=":",
                linewidth=1.2,
            )

        if self.baseline_reward is not None:
            ax.axhline(
                self.baseline_reward,
                color="crimson",
                linestyle="--",
                linewidth=1.5,
                label=f"Baseline ({self.baseline_reward:.2f})",
            )

        ax.axhline(
            EPISODE_REWARD_CLIP,
            color="black",
            linestyle="-.",
            linewidth=1.2,
            alpha=0.7,
            label=f"Penalty Clip ({EPISODE_REWARD_CLIP})",
        )

        ax.plot(
            episodes,
            rewards,
            color="tab:orange",
            alpha=0.3,
            linewidth=1.0,
            label="Episode Reward",
        )

        window_size = 10
        ma_episodes = np.array([])
        moving_avg = np.array([])
        if len(rewards) >= window_size:
            moving_avg = np.convolve(
                rewards, np.ones(window_size) / window_size, mode="valid"
            )
            ma_episodes = episodes[window_size - 1 :]

            ax.plot(
                ma_episodes,
                moving_avg,
                color="tab:orange",
                linewidth=2.2,
                label=f"Moving Avg ({window_size} ep)",
            )

        best_ep = int(np.argmax(rewards))
        best_reward = rewards[best_ep]
        ax.scatter(
            best_ep,
            best_reward,
            color="tab:blue",
            s=100,
            zorder=5,
            marker="*",
            label=f"Best Ep #{best_ep} ({best_reward:.2f})",
        )

        ax_inset = inset_axes(
            ax, width="42%", height="38%", loc="center right", borderpad=2.5
        )

        if self.baseline_reward is not None:
            ax_inset.axhline(
                self.baseline_reward, color="crimson", linestyle="--", linewidth=1.2
            )
        ax_inset.plot(episodes, rewards, color="tab:orange", alpha=0.3, linewidth=0.8)
        if len(rewards) >= window_size:
            ax_inset.plot(ma_episodes, moving_avg, color="tab:orange", linewidth=1.8)

        zoom_start = max(0, min(self.end_of_random_episode or 0, len(episodes) - 2))
        if len(episodes) > zoom_start:
            ax_inset.set_xlim(zoom_start, len(episodes) - 1)
            converged_rewards = rewards[zoom_start:]
            y_min, y_max = (
                np.percentile(converged_rewards, 2),
                np.percentile(converged_rewards, 98),
            )
            base_ref = self.baseline_reward if self.baseline_reward is not None else 0.0
            ax_inset.set_ylim(min(y_min, base_ref - 3), max(y_max, 0))

        ax_inset.grid(True, linestyle="--", alpha=0.3)
        ax_inset.set_title("Convergence Zoom", fontsize=9, fontweight="bold")
        ax_inset.tick_params(axis="both", which="major", labelsize=8)

        mark_inset(ax, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5", linestyle=":")

        ax.set_title("TD3 Training Reward Evolution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Episode [-]", fontsize=12)
        ax.set_ylabel("Reward [-]", fontsize=12)

        ax.set_xlim(0, max(1, self.hp.total_episodes - 1))
        ax.grid(True, linestyle="--", alpha=0.4)

        ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

        save_path = self.master_path / "training_reward_evolution.png"
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Reward evolution plot saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)
