import copy
import time
from dataclasses import fields
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
import torch.nn.functional as functional
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from numpy.typing import NDArray
from torch import nn, Tensor

from ml.environment import EnvironmentForcingDNS
from ml.reference_scheduler import ReferenceTrajectory
from ml.tau_ann import (
    TauANNConfig,
    TD3Hyperparameters,
    TauANN,
    ReplayBuffer,
    save_tau_ann,
)
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem

EPISODE_PENALTY_CLIP = -10000


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
    """Training wrapper for the TD3 training pipeline with plain-text logging."""

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
        self.master_path = Path(master_path)
        self.reference_trajectory = reference_trajectory
        self.hp = hp

        self.baseline_reward: float | None = None
        self.end_of_random_episode: int | None = None
        self.episode_reward_history: list = []

        self.best_historical_reward = -np.inf
        self.best_action_sequence: list = []

        # File logging setup
        self.master_path.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.master_path / "training_log.txt"
        self._init_txt_log()

    def _init_txt_log(self) -> None:
        """Initialize or overwrite the text log file with a session header."""
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write(
                "=========================================================================\n"
            )
            f.write(
                "                      TD3 TRAINING PIPELINE LOG                          \n"
            )
            f.write(f"  Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Directory : {self.master_path.resolve()}\n")
            f.write(
                "=========================================================================\n\n"
            )

    def _log(self, message: str = "", end: str = "\n") -> None:
        """Helper to print to console and write to the plain-text log file concurrently."""
        print(message, end=end)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(message + end)

    def run_training(self) -> TauANN:
        """Main training loop connecting the environment and TD3 agent."""

        self.print_title("Initializing Training Procedure")

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

        # PART 1: Print & log initial configuration
        self.print_configurations(agent)

        self.print_title("Starting Training Loop")

        total_steps = 0
        max_steps_per_ep = getattr(env, "_max_les_steps", 1000)
        training_start_time = time.time()

        # PART 2: In-Episode Training Loop
        for episode in range(self.ann_config.n_training_episodes):
            ep_start_time = time.time()
            state = env.reset()
            episode_reward = 0.0
            done = False
            episode_steps = 0
            actions = []

            # Header for in-episode step logging
            phase_str = (
                "EXPLORATION"
                if total_steps < self.hp.stochastic_timesteps
                else "POLICY TRAINING"
            )
            self._log(
                f"\n>>> Episode {episode + 1}/{self.ann_config.n_training_episodes} "
                f"[{phase_str}]"
            )
            self._log("-" * 75)

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

                # In-Episode Step Progress Logging
                if episode_steps % 100 == 0 or done:
                    pct = (episode_steps / max_steps_per_ep) * 100
                    sim_t = env.solver.time_elapsed if hasattr(env, "solver") else 0.0
                    self._log(
                        f"  Step {episode_steps:>4d}/{max_steps_per_ep} ({pct:>5.1f}%) | "
                        f"Step Rwd: {reward:>8.4f} | "
                        f"Ep Rwd: {episode_reward:>8.2f} | "
                        f"Sim Time: {sim_t:.4f}s"
                    )

            if total_steps >= self.hp.stochastic_timesteps:
                agent.train(replay_buffer, self.hp.batch_size)

            clipped_reward = np.clip(
                episode_reward,
                a_min=EPISODE_PENALTY_CLIP,
                a_max=None,
            )
            self.episode_reward_history.append(clipped_reward)

            # Execution of Post-Processing & Plots
            self._log("-" * 75)
            self._log("  [POST-PROCESSING & PLOTTING ARTIFACTS]")

            self.plot_history_breakdown(env=env, show_plot=False)
            self.plot_profile_comparison(env=env, episode=episode + 1, show_plot=False)

            if hasattr(env, "solver"):
                env.solver.post_processing()

            # End of Episode Summary Box
            ep_duration = time.time() - ep_start_time
            mean_action = np.mean(actions) if len(actions) > 0 else 0.0

            self._log("-" * 75)
            self._log(
                f"  [EPISODE {episode + 1} SUMMARY]\n"
                f"  Duration      : {ep_duration:.2f}s\n"
                f"  Total Steps   : {total_steps} (Ep Steps: {episode_steps})\n"
                f"  Reward        : {clipped_reward:.4f} (Raw: {episode_reward:.4f})\n"
                f"  Mean Action   : {mean_action:.4f}\n"
                f"  Replay Buffer : {replay_buffer.size}/{self.hp.replay_buffer_max_size}"
            )
            self._log("=" * 75)

        # PART 3: Post-Training Summary Dashboard
        total_duration = time.time() - training_start_time
        rewards = np.array(self.episode_reward_history)

        first_ep_reward = rewards[0]
        best_ep_reward = rewards.max()
        final_ep_reward = rewards[-1]

        # Calculate Improvement Metrics
        raw_delta_best = best_ep_reward - first_ep_reward
        pct_imp_best = (
            (raw_delta_best / abs(first_ep_reward)) * 100.0
            if first_ep_reward != 0
            else 0.0
        )

        raw_delta_final = final_ep_reward - first_ep_reward
        pct_imp_final = (
            (raw_delta_final / abs(first_ep_reward)) * 100.0
            if first_ep_reward != 0
            else 0.0
        )

        # Print Post-Training Summary Sections
        self.print_title("Training Summary")

        self.print_section("Overall Execution")
        self.print_row("Total Episodes Completed", self.ann_config.n_training_episodes)
        self.print_row("Total Steps Simulated", total_steps)
        self.print_row("Total Elapsed Time", f"{total_duration:.2f}s")
        self.print_row(
            "Avg Time per Episode",
            f"{total_duration / max(1, self.ann_config.n_training_episodes):.2f}s",
        )
        self.print_footer()

        self.print_section("Reward Performance")
        self.print_row("Initial Episode Reward", f"{first_ep_reward:.4f}")
        self.print_row("Best Episode Reward", f"{best_ep_reward:.4f}")
        self.print_row("Final Episode Reward", f"{final_ep_reward:.4f}")
        self.print_row("Mean Reward (All Ep)", f"{rewards.mean():.4f}")
        self.print_footer()

        self.print_section("Reward Improvement Metrics")
        self.print_row(
            "Initial -> Best Delta", f"{raw_delta_best:+.4f} ({pct_imp_best:+.2f}%)"
        )
        self.print_row(
            "Initial -> Final Delta",
            f"{raw_delta_final:+.4f} ({pct_imp_final:+.2f}%)",
        )
        self.print_footer()

        assert self.ann_config.ann_path is not None
        save_tau_ann(model=agent.actor, save_path=self.ann_config.ann_path)
        self._log(
            f"\n[SUCCESS] Saved trained TauANN model to: {self.ann_config.ann_path}\n"
        )

        return agent.actor

    def print_section(self, title: str, width: int = 70) -> None:
        self._log(f"\n+- {title} " + "-" * (width - len(title) - 3) + "+")

    def print_row(self, label: str, value: Any, include_brackets: bool = True) -> None:
        val_str = str(value)
        if include_brackets:
            self._log(f"|  {label:<28} : {val_str:<36} |")
        else:
            self._log(f"   {label:<28} : {val_str:<36} ")

    def print_footer(self, width: int = 70) -> None:
        self._log("+" + "-" * width + "+")

    def print_title(self, title: str, width: int = 90) -> None:
        """Printing routine for title section."""
        self._log(f"\n- {title} " + "-" * (width - len(title) - 3))

    def print_configurations(
        self, agent: "TD3Agent", print_hyperparameters: bool = True
    ) -> None:
        """Start of training logging showcasing internal settings, parameters and configurations."""
        w = 70

        self.print_row("Master Path", self.master_path, include_brackets=False)

        # 1. Neural Network Architectures
        self.print_section("Neural Network Architectures", w)
        self.print_row(
            "Actor Network",
            f"{agent.actor.state_dim} -> {agent.actor.hidden_dim}x3 -> {agent.actor.action_dim}",
        )
        self.print_row(
            "Critic Network",
            f"{agent.critic.state_dim + agent.critic.action_dim} -> {agent.critic.hidden_dim}x3 -> 1",
        )
        self.print_footer(w)

        # 2. ANN Configuration
        self.print_section("ANN Configuration Settings", w)
        self.print_row("Tau Model", self.ann_config.tau_model)

        ann_path_val = (
            self.ann_config.ann_path.name
            if hasattr(self.ann_config.ann_path, "name")
            else str(self.ann_config.ann_path)
        )
        self.print_row("ANN Model Path", ann_path_val)

        self.print_row("Skip Steps (N Skip)", self.ann_config.n_skip_steps)
        self.print_row("Input Scope", self.ann_config.input_scope)
        self.print_row("Output Scope", self.ann_config.output_scope)
        self.print_row(
            "Action Range",
            f"[{self.ann_config.min_action}, {self.ann_config.max_action}]",
        )
        self.print_row("Training Episodes", self.ann_config.n_training_episodes)

        if hasattr(self.ann_config, "output_scope") and str(
            self.ann_config.output_scope
        ) in ("Scope.HYBRID", "Scope.LOCAL"):
            self.print_row("Local Action Groups", self.ann_config.n_local_action_groups)
            self.print_row("Local Stencil Size", self.ann_config.local_stencil_size)

        self.print_footer(w)

        # 3. Hyperparameters Dataclass Unpacking
        if print_hyperparameters and hasattr(self, "hp"):
            self.print_section("TD3 Hyperparameters", w)

            if hasattr(self.hp, "__dataclass_fields__"):
                for field in fields(self.hp):
                    key = field.name
                    val = getattr(self.hp, key)
                    self.print_row(key, val)
            elif isinstance(self.hp, dict):
                for key, val in self.hp.items():
                    self.print_row(key, val)
            else:
                for key, val in vars(self.hp).items():
                    if not key.startswith("_"):
                        self.print_row(key, val)

            self.print_footer(w)

        # 4. Time-Blocking
        self.print_section("Time-Block", w)
        self.print_row(
            "Time Range",
            f"[{self.problem.t_start} - {self.problem.t_start + self.problem.domain_timespan}]",
        )
        self.print_footer(w)

    def plot_reward_evolution(self, show_plot: bool = False):
        """Visualize the evolution of episode rewards over training with a clean inset zoom."""
        episodes = np.arange(self.ann_config.n_training_episodes)
        rewards = np.array(self.episode_reward_history)

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
            EPISODE_PENALTY_CLIP,
            color="black",
            linestyle="-.",
            linewidth=1.2,
            alpha=0.7,
            label=f"Penalty Clip ({EPISODE_PENALTY_CLIP})",
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
        if len(rewards) >= window_size:
            moving_avg = np.convolve(
                rewards, np.ones(window_size) / window_size, mode="valid"
            )
            ma_episodes = episodes[window_size - 1:]

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

        # -------------------------------------------------------------
        # INSET ZOOM PLOT
        # -------------------------------------------------------------
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

        # Dynamically determine the zoom window starting point
        exploration_end = self.end_of_random_episode or 0
        total_eps = len(episodes)

        # Zooms in on the last 30% of training, but respects exploration phase bounds
        if total_eps > exploration_end + 5:
            zoom_start = max(exploration_end, int(total_eps * 0.7))
        else:
            zoom_start = 0

        ax_inset.set_xlim(zoom_start, max(1, total_eps - 1))

        # Safely compute y-axis zoom bounds
        converged_rewards = rewards[zoom_start:]
        if len(converged_rewards) > 0:
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

        ax.set_xlim(0, max(1, self.ann_config.n_training_episodes - 1))
        ax.grid(True, linestyle="--", alpha=0.4)

        ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

        save_path = self.master_path / "training_reward_evolution.png"
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        self._log(f"  * Saved reward evolution plot -> {save_path.name}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def plot_profile_comparison(
        self, env: EnvironmentForcingDNS, episode: int, show_plot: bool = False
    ) -> None:
        """Plot comparison of the mean velocity profile against DNS reference at episode end."""
        try:
            les_mean = env.solver.calculate_mean_profile()
        except ValueError as e:
            self._log(f"[Warning] Skipping profile plot for episode {episode}: {e}")
            return

        mean_profile_dns = env.reference_trajectory.target_profile

        if les_mean.shape != mean_profile_dns.shape:
            raise ValueError(
                f"Shape mismatch in profile comparison! "
                f"LES profile shape {les_mean.shape} vs DNS target shape {mean_profile_dns.shape}."
            )

        l2_error = float(np.linalg.norm(les_mean - mean_profile_dns))
        dns_norm = np.linalg.norm(mean_profile_dns)
        relative_l2_error = (l2_error / (dns_norm + 1e-12)) * 100.0

        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)

        ax.plot(
            self.disc_config.mesh_les,
            mean_profile_dns,
            color="royalblue",
            linestyle="-",
            linewidth=1.2,
            label="Projected DNS Reference",
            zorder=2,
        )

        ax.plot(
            self.disc_config.mesh_les,
            les_mean,
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
            marker="o",
            markevery=max(1, len(self.disc_config.mesh_les) // 16),
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=f"LES Model ({len(self.disc_config.mesh_les)} pts)",
            zorder=3,
        )

        all_data = np.concatenate([mean_profile_dns, les_mean])
        y_min, y_max = all_data.min(), all_data.max()
        y_range = y_max - y_min if y_max != y_min else 1.0
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        ax.set_title(
            r"Mean Velocity Profile Comparison $\langle w \rangle$ "
            + f"(Episode {episode})",
            fontsize=13,
            pad=10,
        )
        ax.set_xlabel(r"Domain Coordinate $z$", fontsize=11)
        ax.set_ylabel(r"Mean Velocity $\langle w \rangle$", fontsize=11)

        ax.set_xlim(min(self.disc_config.mesh_les), max(self.disc_config.mesh_les))
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", alpha=0.25)
        ax.minorticks_on()

        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)

        score_text = (
            rf"$\mathrm{{L}}_2$ Error: {l2_error:.4e}"
            + f"\nRel. Error: {relative_l2_error:.2f}%"
        )
        ax.text(
            0.03,
            0.05,
            score_text,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="gray",
                alpha=0.85,
            ),
            zorder=5,
        )

        plt.tight_layout()

        out_dir = self.master_path / "profile_comparisons"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_plot_path = out_dir / f"profile_comparison_ep{episode:03d}.png"
        plt.savefig(output_plot_path, dpi=300)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        self._log(f"  * Saved velocity profile comparison -> {output_plot_path.name}")

    def plot_history_breakdown(
        self, env: EnvironmentForcingDNS, show_plot: bool = False
    ) -> None:
        """Plot the evolution of raw and weighted reward components across environment steps."""
        window_size = 100
        burn_in_steps = getattr(self.hp, "burn_in_steps", 0)

        histories_raw = {
            "Action Penalty (raw)": (env.action_penalty_history_raw, "tab:red"),
            "Spectral Penalty (raw)": (env.spectral_penalty_history_raw, "tab:purple"),
            "Distance Error (raw)": (env.distance_error_history_raw, "tab:blue"),
            "Distance Improvement (raw)": (
                env.distance_improvement_history_raw,
                "tab:green",
            ),
        }

        # 1. FIGURE 1: RAW METRICS
        fig_raw, axes_raw = plt.subplots(
            4, 1, figsize=(10, 10), sharex=True, layout="constrained", dpi=300
        )

        for ax, (title, (data, color)) in zip(axes_raw, histories_raw.items()):
            if not data:
                ax.text(
                    0.5, 0.5, f"No data recorded for {title}", ha="center", va="center"
                )
                continue

            data_arr = np.array(data)
            steps = np.arange(len(data_arr))

            ax.plot(
                steps,
                data_arr,
                color=color,
                alpha=0.35,
                linewidth=1.0,
                label="Raw Step Value",
            )

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

            if burn_in_steps > 0 and (
                "Distance Error" in title or "Distance Improvement" in title
            ):
                ax.axvline(
                    x=burn_in_steps,
                    color="gray",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                    label=f"Burn-In End ({burn_in_steps} steps)",
                )

            ax.set_ylabel(title, fontsize=10, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

        axes_raw[-1].set_xlabel("Environment Step [-]", fontsize=11)
        fig_raw.suptitle(
            "Raw Physical Reward Components", fontsize=14, fontweight="bold"
        )

        save_path_raw = self.master_path / "reward_components_raw.png"
        plt.savefig(save_path_raw, dpi=300, bbox_inches="tight")
        self._log(f"  * Saved raw component metrics     -> {save_path_raw.name}")

        # 2. FIGURE 2: WEIGHTED METRICS & TOTAL REWARDS
        fig_weighted, axes_weighted = plt.subplots(
            5, 1, figsize=(10, 12), sharex=True, layout="constrained", dpi=300
        )

        ax_total = axes_weighted[0]

        if env.total_reward_history_unscaled:
            unscaled_arr = np.array(env.total_reward_history_unscaled)
            steps = np.arange(len(unscaled_arr))

            ax_total.plot(
                steps, unscaled_arr, color="tab:orange", alpha=0.2, linewidth=0.8
            )
            if len(unscaled_arr) >= window_size:
                ma_unscaled = np.convolve(
                    unscaled_arr, np.ones(window_size) / window_size, mode="valid"
                )
                ax_total.plot(
                    steps[window_size - 1 :],
                    ma_unscaled,
                    color="tab:orange",
                    linewidth=1.8,
                    label="Total Reward (Unscaled)",
                )

        if env.total_reward_history_scaled:
            scaled_arr = np.array(env.total_reward_history_scaled)
            steps = np.arange(len(scaled_arr))

            ax_total.plot(
                steps, scaled_arr, color="royalblue", alpha=0.2, linewidth=0.8
            )
            if len(scaled_arr) >= window_size:
                ma_scaled = np.convolve(
                    scaled_arr, np.ones(window_size) / window_size, mode="valid"
                )
                ax_total.plot(
                    steps[window_size - 1 :],
                    ma_scaled,
                    color="royalblue",
                    linewidth=1.8,
                    label="Total Reward (Scaled)",
                )

        ax_total.set_ylabel("Total Reward", fontsize=10, fontweight="bold")
        ax_total.grid(True, linestyle="--", alpha=0.4)
        ax_total.legend(loc="upper right", fontsize=8, framealpha=0.8)

        histories_weighted_components = {
            "Action Penalty (weighted)": (
                env.action_penalty_history_weighted,
                "tab:red",
            ),
            "Spectral Penalty (weighted)": (
                env.spectral_penalty_history_weighted,
                "tab:purple",
            ),
            "Distance Error (weighted)": (
                env.distance_error_history_weighted,
                "tab:blue",
            ),
            "Distance Improvement (weighted)": (
                env.distance_improvement_history_weighted,
                "tab:green",
            ),
        }

        for ax, (title, (data, color)) in zip(
            axes_weighted[1:], histories_weighted_components.items()
        ):
            if not data:
                ax.text(
                    0.5, 0.5, f"No data recorded for {title}", ha="center", va="center"
                )
                continue

            data_arr = np.array(data)
            steps = np.arange(len(data_arr))

            ax.plot(
                steps,
                data_arr,
                color=color,
                alpha=0.35,
                linewidth=1.0,
                label="Weighted Step Value",
            )

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

            if burn_in_steps > 0 and (
                "Distance Error" in title or "Distance Improvement" in title
            ):
                ax.axvline(
                    x=burn_in_steps,
                    color="gray",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                    label=f"Burn-In End ({burn_in_steps} steps)",
                )

            ax.set_ylabel(title, fontsize=10, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

        axes_weighted[-1].set_xlabel("Environment Step [-]", fontsize=11)
        fig_weighted.suptitle(
            "Weighted Reward Components & Total Rewards", fontsize=14, fontweight="bold"
        )

        save_path_weighted = self.master_path / "reward_components_weighted.png"
        plt.savefig(save_path_weighted, dpi=300, bbox_inches="tight")
        self._log(f"  * Saved weighted component metrics -> {save_path_weighted.name}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig_raw)
            plt.close(fig_weighted)
