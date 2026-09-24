import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from numpy.typing import NDArray
import torch.nn.functional as functional

from final.ml.environment import EnvironmentForcingDNS
from final.ml.hyperparameters import TD3Hyperparameters
from final.ml.reference_scheduler import ReferenceTrajectory
from final.ml.shared_assets import TwinQCritic, ReplayBuffer
from final.ml.tau_ann import TauANNConfig, TauANN, save_tau_ann
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem

EPISODE_PENALTY_CLIP = -1000


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
            action = (action + noise).clip(self.hp.min_action, self.hp.max_action)

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
        self.episode_reward_history: list = []

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
            f"Architecture Actor: {agent.actor.state_dim} -> {agent.actor.hidden_dim} x 3 -> {agent.actor.action_dim}"
        )
        print(
            f"Architecture Critic: {agent.critic.state_dim} -> {agent.critic.hidden_dim} x 3 -> {agent.critic.action_dim}"
        )

        total_steps = 0
        for episode in range(self.ann_config.n_training_episodes):
            print("episode:", episode)
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
                next_state, reward, done = env.step(action=action)
                episode_reward += reward
                replay_buffer.add(state, action, next_state, reward, done)

                if episode_steps % 100 == 0:
                    print(
                        f"step {episode_steps} | "
                        f"total steps {env._max_les_steps} | "
                        f"step reward {reward:.4f} | "  # Print instantaneous step reward
                        f"cumulative ep reward {episode_reward:.2f}"
                    )

            if total_steps >= self.hp.stochastic_timesteps:
                agent.train(replay_buffer, self.hp.batch_size)

            episode_reward = np.clip(
                episode_reward,
                a_min=EPISODE_PENALTY_CLIP,
                a_max=None,
            )
            self.episode_reward_history.append(episode_reward)

            self.plot_history_breakdown(env=env, show_plot=False)

            print(
                f"Episode: {episode + 1}/{self.hp.total_episodes} | "
                f"Steps in Ep: {episode_steps} | "
                f"Total Steps: {total_steps} | "
                f"Scope: {self.ann_config.output_scope.value} ({self.ann_config.n_local_action_groups}) | "
                f"Reward: {episode_reward:.4f} | "
            )

        assert self.ann_config.ann_path is not None
        save_tau_ann(model=agent.actor, save_path=self.ann_config.ann_path)
        print(f"Successfully saved trained TauANN to {self.ann_config.ann_path}")

        return agent.actor

    def plot_reward_evolution(self, show_plot: bool = False):
        """Visualize the evolution of episode rewards over training with a clean inset zoom."""
        episodes = np.arange(self.hp.total_episodes)
        rewards = np.array(self.episode_reward_history)

        # Use constrained_layout=True to prevent tight_layout warnings with inset_axes
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300, layout="constrained")

        # 1. Background shading for exploration vs. policy phase
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

        # 2. Reference lines (Baseline & Penalty Clip)
        assert self.baseline_reward is not None
        ax.axhline(
            self.baseline_reward,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline ({self.baseline_reward:.2f})",
        )

        ax.axhline(
            EPISODE_PENALTY_CLIP,
            color="black",
            linestyle="-.",
            linewidth=1.2,
            alpha=0.7,
            label=f"Penalty Clip ({EPISODE_PENALTY_CLIP})",
        )

        # 3. Raw rewards trajectory & Moving Average
        ax.plot(
            episodes,
            rewards,
            color="tab:orange",
            alpha=0.3,
            linewidth=1.0,
            label="Episode Reward",
        )

        window_size = 10
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

        # 4. Best Episode Marker
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

        # -------------------------------------------------------------
        # 5. INSET ZOOM PLOT
        # -------------------------------------------------------------
        # Position: lower right-center to leave space for legend in lower-right corner
        ax_inset = inset_axes(
            ax, width="42%", height="38%", loc="center right", borderpad=2.5
        )

        ax_inset.axhline(
            self.baseline_reward, color="crimson", linestyle="--", linewidth=1.2
        )
        ax_inset.plot(episodes, rewards, color="tab:orange", alpha=0.3, linewidth=0.8)
        if len(rewards) >= window_size:
            ax_inset.plot(ma_episodes, moving_avg, color="tab:orange", linewidth=1.8)

        zoom_start = max(100, self.end_of_random_episode or 0)
        ax_inset.set_xlim(zoom_start, len(episodes) - 1)

        converged_rewards = rewards[zoom_start:]
        y_min, y_max = (
            np.percentile(converged_rewards, 2),
            np.percentile(converged_rewards, 98),
        )
        ax_inset.set_ylim(min(y_min, self.baseline_reward - 3), max(y_max, 0))

        ax_inset.grid(True, linestyle="--", alpha=0.3)
        ax_inset.set_title("Convergence Zoom", fontsize=9, fontweight="bold")
        ax_inset.tick_params(axis="both", which="major", labelsize=8)

        # Clean connection lines (loc1=3 -> bottom-left, loc2=1 -> top-right)
        mark_inset(ax, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5", linestyle=":")
        # -------------------------------------------------------------

        # 6. Styling & Labels
        ax.set_title("TD3 Training Reward Evolution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Episode [-]", fontsize=12)
        ax.set_ylabel("Reward [-]", fontsize=12)

        ax.set_xlim(0, self.hp.total_episodes - 1)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Legend in lower right corner
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

        # Save handling (no plt.tight_layout() call!)
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

    def plot_history_breakdown(
        self, env: EnvironmentForcingDNS, show_plot: bool = False
    ) -> None:
        """Plot the evolution of all 5 reward/penalty histories over training steps."""
        histories = {
            "Total Reward": (env.total_reward_history, "tab:orange"),
            "Action Penalty": (env.action_penalty_history, "tab:red"),
            "Spectral Penalty": (env.spectral_penalty_history, "tab:purple"),
            "Distance Error": (env.distance_error_history, "tab:blue"),
            "Distance Improvement": (env.distance_improvement_history, "tab:green"),
        }

        fig, axes = plt.subplots(
            5, 1, figsize=(10, 12), sharex=True, layout="constrained", dpi=300
        )
        window_size = 100  # Smoothing window across environment steps

        for ax, (title, (data, color)) in zip(axes, histories.items()):
            if not data:
                ax.text(
                    0.5, 0.5, f"No data recorded for {title}", ha="center", va="center"
                )
                continue

            data_arr = np.array(data)
            steps = np.arange(len(data_arr))

            # Raw values in faint background line
            ax.plot(
                steps,
                data_arr,
                color=color,
                alpha=0.25,
                linewidth=0.8,
                label="Raw Step Value",
            )

            # Moving Average
            if len(data_arr) >= window_size:
                moving_avg = np.convolve(
                    data_arr, np.ones(window_size) / window_size, mode="valid"
                )
                ax.plot(
                    steps[window_size - 1 :],
                    moving_avg,
                    color=color,
                    linewidth=1.8,
                    label=f"Moving Avg ({window_size} steps)",
                )

            ax.set_ylabel(title, fontsize=10, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

        axes[-1].set_xlabel("Environment Step [-]", fontsize=11)
        fig.suptitle(
            "Environment Reward & Component Breakdown", fontsize=14, fontweight="bold"
        )

        save_path = self.master_path / "reward_components_evolution.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Reward components plot saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)
