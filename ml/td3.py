import copy
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
import torch.nn.functional as functional
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from numpy.typing import NDArray
from torch import nn, Tensor

from ml.ann import ReplayBuffer, TauANN, save_tau_ann
from ml.ann_config import TauANNConfig, TD3Hyperparameters
from ml.diagnostics import plot_diagnostic_metrics
from ml.environment import EnvironmentForcingDNS
from ml.les_caching import (
    LESCacheKey,
    resolve_les_baseline_cache,
    LESCacheStatus,
    write_les_parameters,
)
from ml.reference_scheduler import ReferenceTrajectory
from proofing.action_based.config import ANNConfigProof
from proofing.action_based.environment import EnvironmentProof
from proofing.action_based.solver import SolverForProofs
from proofing.single_action_testing.config import ANNSingleActionConfig
from proofing.single_action_testing.environment import EnvironmentSingleActionTraining
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SimulationMode, SolverBase
from solvers.solver_coupled import SolverCoupled

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
        self.device = torch.device("cpu")

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
        """Select physical action with Gaussian exploration noise scaled by action span."""
        state_tensor = torch.as_tensor(
            state.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        action = self.actor(state_tensor).cpu().data.numpy().flatten()

        if noise_std > 0.0:
            act_span = self.hp.max_action - self.hp.min_action
            noise = np.random.normal(0, noise_std * act_span, size=action.shape)
            action = (action + noise).clip(self.hp.min_action, self.hp.max_action)

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
            act_span = self.hp.max_action - self.hp.min_action
            noise = (torch.randn_like(action) * self.hp.policy_noise * act_span).clamp(
                -self.hp.noise_clip * act_span,
                self.hp.noise_clip * act_span,
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                self.hp.min_action, self.hp.max_action
            )

            # Clipped double Q-learning (Critic receives physical actions directly)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done.float()) * self.hp.discount * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = functional.smooth_l1_loss(
            current_q1, target_q
        ) + functional.smooth_l1_loss(current_q2, target_q)

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


# =============================================================================
# Main Training Pipeline
# =============================================================================


class TD3Trainer:
    """Training wrapper for the TD3 training pipeline with plain-text logging."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        ann_config: TauANNConfig | ANNConfigProof,
        master_path: Path,
        reference_trajectory: ReferenceTrajectory,
        hp: TD3Hyperparameters = TD3Hyperparameters(),
        baseline_path: Path | None = None,
    ):
        self.problem = problem
        self.disc_config = disc_config
        self.ann_config: TauANNConfig | ANNConfigProof | ANNSingleActionConfig = (
            ann_config
        )
        self.master_path = Path(master_path)
        self.reference_trajectory = reference_trajectory
        self.hp = hp
        self.baseline_dir = (
            Path(baseline_path)
            if baseline_path
            else self.master_path.parent / "baseline"
        )
        self.baseline_run_dir: Path | None = None

        self.episodes_ran: int = 0

        self.baseline_reward: float | None = None
        self.end_of_random_episode: int | None = None
        self.episode_reward_history: list = []

        self.best_historical_reward = -np.inf
        self.best_action_sequence: list = []

        self.solver: SolverCoupled | SolverBase | SolverForProofs | None = None
        self.env: EnvironmentForcingDNS | EnvironmentProof | None = None

        # File logging setup
        self.master_path.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.master_path / "training_log.txt"
        self._init_txt_log()

        self.mean_actions: list[float] = []
        self.action_deviation_history = []
        self.action_mean_history = []
        self.critic_loss_history = []
        self.actor_loss_history = []
        self.q_value_gradient_history = []
        self.q_sensitivity_history = []

    def run_training(self) -> TauANN:
        """Main training loop connecting the environment and TD3 agent."""

        self.print_title("Initializing Training Procedure")

        if isinstance(self.ann_config, ANNConfigProof):
            env = EnvironmentProof(
                problem=self.problem,
                disc_config=self.disc_config,
                ann_config=self.ann_config,
                hyperparameters=self.hp,
                reference_trajectory=self.reference_trajectory,
                master_path=self.master_path,
            )
        elif isinstance(self.ann_config, ANNSingleActionConfig):
            env = EnvironmentSingleActionTraining(
                problem=self.problem,
                disc_config=self.disc_config,
                ann_config=self.ann_config,
                hyperparameters=self.hp,
                reference_trajectory=self.reference_trajectory,
                master_path=self.master_path,
            )
        elif isinstance(self.ann_config, TauANNConfig):
            env = EnvironmentForcingDNS(
                problem=self.problem,
                disc_config=self.disc_config,
                ann_config=self.ann_config,
                hyperparameters=self.hp,
                reference_trajectory=self.reference_trajectory,
                master_path=self.master_path,
            )

        self.env = env
        agent = TD3Agent(ann_config=self.ann_config, hp=self.hp)

        replay_buffer = ReplayBuffer(
            state_dim=self.ann_config.state_dimension,
            action_dim=self.ann_config.action_dimension,
            max_size=self.hp.replay_buffer_max_size,
        )

        self.print_configurations(agent)

        self.print_title("Starting Baseline Evaluation Run")

        self.run_baseline_evaluation()

        self.print_title("Starting Training Loop")

        total_steps = 0
        max_steps_per_ep = getattr(self.ann_config, "n_agent_steps_per_episode", 1000)
        training_start_time = time.time()

        episode = 0
        stop_training = False

        if hasattr(self.ann_config, "n_random_episodes"):
            stochastic_steps = self.ann_config.n_random_episodes
        else:
            stochastic_steps = self.hp.stochastic_timesteps

        while episode < self.ann_config.n_training_episodes and not stop_training:
            ep_start_time = time.time()
            state = env.reset()
            initial_state = state.copy()  # Cache for diagnostic check
            episode_reward = 0.0
            done = False
            episode_steps = 0
            actions = []
            skip_episode = False

            self._log_episode_header(episode, total_steps)

            while not done:
                total_steps += 1
                episode_steps += 1

                # Action selection
                if total_steps < stochastic_steps:
                    min_action, max_action = self.get_action_bounds(steps=total_steps)
                    if isinstance(self.ann_config, ANNSingleActionConfig):
                        # Sample one random value and broadcast it across all action dimensions
                        rand_val = np.random.uniform(min_action, max_action)
                        action = np.full(self.ann_config.action_dimension, rand_val)
                    else:
                        action = np.random.uniform(
                            min_action,
                            max_action,
                            size=self.ann_config.action_dimension,
                        )
                else:
                    action = agent.select_action(state, noise_std=self.hp.expl_noise)
                    if isinstance(self.ann_config, ANNSingleActionConfig):
                        # Enforce exact identity across actions (if noise introduced minor floating-point differences)
                        action = np.full(self.ann_config.action_dimension, action[0])

                actions.append(action)
                # Execution & numeric safety checks
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        next_state, reward, done, info = env.step(
                            action=action,
                        )

                    if hasattr(env, "solver") and not np.all(
                        np.isfinite(env.solver.solution)
                    ):
                        raise FloatingPointError(
                            "Non-finite values found in solver solution"
                        )

                except (FloatingPointError, ZeroDivisionError, ArithmeticError) as e:
                    stop_training = self._handle_episode_abort(
                        f"Unhandled FloatingPointError ({e})"
                    )
                    skip_episode = True
                    break

                if info.get("crashed", False):
                    reason = info.get("crash_reason", "Solver Divergence")
                    stop_training = self._handle_episode_abort(
                        f"Episode {episode + 1} aborted: {reason}"
                    )
                    skip_episode = True
                    break

                # Buffer & training updates
                replay_buffer.add(state, action, next_state, reward, done)
                state = next_state
                episode_reward += reward

                # Step progress logging
                if episode_steps % 100 == 0 or done:
                    self._log_step_progress(
                        episode_steps, max_steps_per_ep, reward, episode_reward, env
                    )

                if total_steps >= stochastic_steps:
                    critic_loss, actor_loss = agent.train(
                        replay_buffer, self.hp.batch_size
                    )

                    self.critic_loss_history.append(critic_loss)
                    self.actor_loss_history.append(actor_loss)

            if skip_episode:
                episode += 1
                continue

            # Post-processing & logging
            clipped_reward = float(
                np.clip(episode_reward, a_min=EPISODE_PENALTY_CLIP, a_max=None)
            )
            self.episode_reward_history.append(clipped_reward)

            solver_post_processing = False
            if episode >= self.ann_config.n_training_episodes:
                solver_post_processing = True
            self._execute_post_processing(
                env, episode, do_solver_post_processing=solver_post_processing
            )

            # Convert action statistics explicitly to float scalars
            std_actions = float(np.std(actions))
            mean_actions_val = float(np.mean(actions))

            self.action_deviation_history.append(std_actions)
            self.mean_actions.append(mean_actions_val)

            self._log_episode_summary(
                episode,
                time.time() - ep_start_time,
                total_steps,
                episode_steps,
                clipped_reward,
                episode_reward,
                actions,
                replay_buffer,
            )

            # Diagnostic critic sensitivity check
            if total_steps >= stochastic_steps:
                a_default = np.ones(self.ann_config.action_dimension, dtype=np.float32)
                a_perturbed = np.full(
                    self.ann_config.action_dimension, 1.1, dtype=np.float32
                )

                q_default = agent.evaluate_q_value(initial_state, a_default)
                q_perturbed = agent.evaluate_q_value(initial_state, a_perturbed)
                delta_q = float(abs(q_default - q_perturbed))

                self.q_sensitivity_history.append(delta_q)
                self._log(
                    f"  [DIAGNOSTIC] Critic Q(s_0, a=1.0)={q_default:.4f} | "
                    f"Q(s_0, a=1.1)={q_perturbed:.4f} | Delta Q={delta_q:.4f}"
                )

            episode += 1
            self.episodes_ran += 1

        # Post-training summary dashboard
        self._log_training_summary(
            time.time() - training_start_time, total_steps, episode
        )

        assert self.ann_config.ann_path is not None
        save_tau_ann(model=agent.actor, save_path=self.ann_config.ann_path)
        self._log(
            f"\n[SUCCESS] Saved trained TauANN model to: {self.ann_config.ann_path}\n"
        )

        self.print_title("Starting Final Evaluation Run")

        if self.ann_config.run_final_evaluation:
            self.run_evaluation()

        return agent.actor

    def get_action_bounds(self, steps: int) -> tuple[float, float]:
        """Compute action bounds on stochastic action selection, follows linearly varying profile."""
        if steps > self.hp.stochastic_timesteps:
            return self.hp.min_action, self.hp.max_action

        progress = steps / max(1, self.hp.stochastic_timesteps)
        current_min = self.hp.init_min_action + progress * (
            self.hp.min_action - self.hp.init_min_action
        )
        current_max = self.hp.init_max_action + progress * (
            self.hp.max_action - self.hp.init_max_action
        )
        return current_min, current_max

    def run_baseline_evaluation(self) -> None:
        """Runs baseline evaluation with cache checking logic.

        If CACHE HIT: Does nothing and reuses existing artifacts.
        If CACHE MISS: Runs SolverBase, saves output to cache dir, and plots baseline profile.
        """
        cache_key = self._build_les_cache_key()
        cache_result = resolve_les_baseline_cache(self.baseline_dir, cache_key)

        if (
            cache_result.status == LESCacheStatus.HIT
            and cache_result.cache_dir is not None
        ):
            self._log(
                f"[CACHE HIT] Found existing baseline run at: {cache_result.cache_dir}. Skipping simulation."
            )
            self.baseline_run_dir = cache_result.cache_dir
            return

        # CACHE MISS: Execute solver run and cache the baseline
        target_cache_dir = self.baseline_dir / cache_key.dir_to_name()
        target_cache_dir.mkdir(parents=True, exist_ok=True)
        self._log(
            f"[CACHE MISS] Executing baseline run and caching to: {target_cache_dir}"
        )
        self.baseline_run_dir = target_cache_dir

        self.solver = SolverBase(
            problem=self.problem,
            disc_config=self.disc_config,
            master_path=target_cache_dir,
            tau_model=self.ann_config.tau_model,
            simulation_mode=SimulationMode.TAU_BASED,
        )
        self.solver.run_simulation()
        self.solver.post_processing()
        write_les_parameters(target_cache_dir, cache_key)

        self.plot_profile_comparison(
            env=None,
            solver=self.solver,
            episode=None,
            dns_trajectory=self.reference_trajectory.target_profile,
            save_dir=target_cache_dir,
        )

    def run_evaluation(self):
        self.solver = SolverCoupled(
            problem=self.problem,
            disc_config=self.disc_config,
            ann_config=replace(self.ann_config, training_mode=False),
            master_path=self.master_path / "evaluation",
            tau_model=self.ann_config.tau_model,
            simulation_mode=SimulationMode.TAU_BASED,
        )

        self.solver.run_simulation()
        self.solver.post_processing()

        self.plot_profile_comparison(
            env=None,
            solver=self.solver,
            episode=None,
            dns_trajectory=self.reference_trajectory.target_profile,
            evaluation_mode=True,
        )

    def run_diagnostic_plotting(self, save_path: Path | None = None) -> Path:
        """Trigger diagnostic plot generation for the trainer instance."""
        if save_path is None:
            save_path = self.master_path / "diagnostics"

        save_path.mkdir(parents=True, exist_ok=True)
        plot_diagnostic_metrics(trainer=self, save_dir=save_path)
        return save_path

    def _build_les_cache_key(self) -> LESCacheKey:
        """Construct LESCacheKey from problem and discretization configurations."""
        return LESCacheKey(
            problem_name=getattr(self.problem, "name", "unknown_problem"),
            domain_length=getattr(self.problem, "domain_length", 0.0),
            viscosity=getattr(self.problem, "viscosity", 0.0),
            bc_type=getattr(self.problem, "boundary_condition_type", "fixed"),
            bc_value=getattr(self.problem, "boundary_condition_value", 0),
            n_nodes=getattr(self.disc_config, "n_nodes_les", 0),
            dt=getattr(self.disc_config, "dt_les", 0.0),
            t_start=getattr(self.problem, "t_start", 0.0),
            t_end=getattr(self.problem, "t_end", 0.0),
            n_params=getattr(self.ann_config, "n_coefficients", 0),
        )

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

    def _log_episode_header(self, episode: int, total_steps: int) -> None:
        """Logs phase and episode start header."""
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

    def _log_step_progress(
        self,
        episode_steps: int,
        max_steps: int,
        reward: float,
        ep_reward: float,
        env: Any,
    ) -> None:
        """Logs individual in-episode step diagnostics."""
        pct = (episode_steps / max_steps) * 100
        sim_t = (
            getattr(env.solver, "time_elapsed", 0.0) if hasattr(env, "solver") else 0.0
        )
        self._log(
            f"  Step {episode_steps:>4d}/{max_steps} ({pct:>5.1f}%) | "
            f"Step Rwd: {reward:>8.4f} | "
            f"Ep Rwd: {ep_reward:>8.2f} | "
            f"Sim Time: {sim_t:.4f}s"
        )

    def _execute_post_processing(
        self, env: Any, episode: int, do_solver_post_processing: bool = False
    ) -> None:
        """Triggers plotting routines and solver post-processing."""
        self._log("-" * 75)
        self._log("  [POST-PROCESSING & PLOTTING ARTIFACTS]")
        self.plot_history_breakdown(env=env, show_plot=False)
        self.plot_profile_comparison(
            env=env, episode=episode, show_plot=False, solver=None
        )
        if hasattr(env, "solver") and do_solver_post_processing:
            env.solver.post_processing()

    def _log_episode_summary(
        self,
        episode: int,
        ep_duration: float,
        total_steps: int,
        episode_steps: int,
        reward: float,
        raw_reward: float,
        actions: list,
        replay_buffer: Any,
    ) -> None:
        """Logs end-of-episode summary block."""
        mean_action = float(np.mean(actions)) if len(actions) > 0 else 0.0
        self._log("-" * 75)
        self._log(
            f"  [EPISODE {episode + 1} SUMMARY]\n"
            f"  Duration      : {ep_duration:.2f}s\n"
            f"  Total Steps   : {total_steps} (Ep Steps: {episode_steps})\n"
            f"  Reward        : {reward:.4f} (Raw: {raw_reward:.4f})\n"
            f"  Mean Action   : {mean_action:.4f}\n"
            f"  Replay Buffer : {replay_buffer.size}/{self.hp.replay_buffer_max_size}"
        )
        self._log("=" * 75)

    def _log_training_summary(
        self, total_duration: float, total_steps: int, episode_count: int
    ) -> None:
        """Computes and prints final post-training statistics dashboard."""
        rewards = np.array(self.episode_reward_history)
        if len(rewards) == 0:
            return

        first_rwd, best_rwd, final_rwd = rewards[0], rewards.max(), rewards[-1]

        raw_delta_best = best_rwd - first_rwd
        pct_imp_best = (
            (raw_delta_best / abs(first_rwd) * 100.0) if first_rwd != 0 else 0.0
        )

        raw_delta_final = final_rwd - first_rwd
        pct_imp_final = (
            (raw_delta_final / abs(first_rwd) * 100.0) if first_rwd != 0 else 0.0
        )

        self.print_title("Training Summary")

        self.print_section("Overall Execution")
        self.print_row("Total Episodes Completed", episode_count)
        self.print_row("Total Steps Simulated", total_steps)
        self.print_row("Total Elapsed Time", f"{total_duration:.2f}s")
        self.print_row(
            "Avg Time per Episode", f"{total_duration / max(1, episode_count):.2f}s"
        )
        self.print_footer()

        self.print_section("Reward Performance")
        self.print_row("Initial Episode Reward", f"{first_rwd:.4f}")
        self.print_row("Best Episode Reward", f"{best_rwd:.4f}")
        self.print_row("Final Episode Reward", f"{final_rwd:.4f}")
        self.print_row("Mean Reward (All Ep)", f"{rewards.mean():.4f}")
        self.print_footer()

        self.print_section("Reward Improvement Metrics")
        self.print_row(
            "Initial -> Best Delta", f"{raw_delta_best:+.4f} ({pct_imp_best:+.2f}%)"
        )
        self.print_row(
            "Initial -> Final Delta", f"{raw_delta_final:+.4f} ({pct_imp_final:+.2f}%)"
        )
        self.print_footer()

    def _handle_episode_abort(self, reason: str) -> bool:
        """Helper to handle episode aborts, increment retries, or enforce safety caps.

        Returns:
            bool: True if training should terminate completely, False to just skip episode.
        """
        self._log(f"\n  [SOLVER ABORT] {reason}. Skipping Post-Processing.")

        if (
            self.ann_config.n_training_episodes
            < self.ann_config.n_total_allowed_episodes
        ):
            self.ann_config.n_training_episodes += 1
            self._log(
                f"  [RETRY QUEUED] Extended target episodes to: {self.ann_config.n_training_episodes}"
            )
            return False

        self._log(
            f"  [MAX ATTEMPTS REACHED] Reached safety cap of {self.ann_config.n_total_allowed_episodes} total episodes. "
            f"Terminating training early."
        )
        return True

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
        if len(self.episode_reward_history) == 0:
            print(
                "Skipping reward evolution across episodes. No episode rewards recorded to plot (episode_reward_history is unpopulated)."
            )
            return

        episodes = np.arange(self.episodes_ran)
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

        # ax.axhline(
        #     EPISODE_PENALTY_CLIP,
        #     color="black",
        #     linestyle="-.",
        #     linewidth=1.2,
        #     alpha=0.7,
        #     label=f"Penalty Clip ({EPISODE_PENALTY_CLIP})",
        # )

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
        if self.episodes_ran >= 50:
            ax_inset = inset_axes(
                ax, width="42%", height="38%", loc="center right", borderpad=2.5
            )

            if self.baseline_reward is not None:
                ax_inset.axhline(
                    self.baseline_reward, color="crimson", linestyle="--", linewidth=1.2
                )
            ax_inset.plot(
                episodes, rewards, color="tab:orange", alpha=0.3, linewidth=0.8
            )
            if len(rewards) >= window_size:
                ax_inset.plot(
                    ma_episodes, moving_avg, color="tab:orange", linewidth=1.8
                )

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
                base_ref = (
                    self.baseline_reward if self.baseline_reward is not None else 0.0
                )
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
        self,
        env: EnvironmentForcingDNS | None,
        solver: SolverCoupled | SolverBase | None,
        episode: int | None,
        show_plot: bool = False,
        dns_trajectory: NDArray | None = None,
        save_dir: Path | None = None,
        evaluation_mode: bool = False,
    ) -> None:
        """Plot comparison of the mean velocity profile against DNS reference.

        Evaluation mode: Use baseline data as a comparison."""

        # 1. Extract mean profile from active solver or environment
        try:
            if solver is not None:
                les_mean = solver.calculate_mean_profile()
            elif env is not None:
                les_mean = env.solver.calculate_mean_profile()
            else:
                self._log(
                    "[Error] Skipping profile plot: neither 'solver' nor 'env' was provided."
                )
                return
        except ValueError as e:
            self._log(f"[Warning] Skipping profile plot: {e}")
            return

        # 2. Extract reference DNS trajectory
        if env is not None:
            mean_profile_dns = env.reference_trajectory.target_profile
        elif dns_trajectory is not None:
            mean_profile_dns = dns_trajectory
        else:
            self._log(
                "[Error] Skipping profile plot: missing reference DNS trajectory."
            )
            return

        if les_mean.shape != mean_profile_dns.shape:
            raise ValueError(
                f"Shape mismatch in profile comparison! "
                f"LES profile shape {les_mean.shape} vs DNS target shape {mean_profile_dns.shape}."
            )

        # 3. Compute error metrics
        l2_error = float(np.linalg.norm(les_mean - mean_profile_dns))
        dns_norm = np.linalg.norm(mean_profile_dns)
        relative_l2_error = (l2_error / (dns_norm + 1e-12)) * 100.0

        # 4. Generate Plot
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

        if evaluation_mode:
            mean_profiles_dir = self.baseline_run_dir / "mean_profiles"
            baseline_file = mean_profiles_dir / f"mean_w_{self.problem.t_end}.npy"

            # Locate baseline file or fallback safely
            if not baseline_file.exists():
                npy_files = list(mean_profiles_dir.glob("*.npy"))
                if not npy_files:
                    self._log(
                        f"[Warning] Skipping baseline plot: No .npy files found in {mean_profiles_dir}"
                    )
                    baseline_file = None
                else:
                    baseline_file = npy_files[0]

            if baseline_file:
                baseline_mean = np.load(baseline_file)
                mesh = self.disc_config.mesh_les

                # Validate dimensions before plotting
                if mesh.shape != baseline_mean.shape:
                    self._log(
                        f"[Warning] Skipping baseline plot due to shape mismatch: "
                        f"mesh {mesh.shape} vs baseline {baseline_mean.shape}."
                    )
                else:
                    ax.plot(
                        mesh,
                        baseline_mean,
                        color="tab:green",
                        linestyle="--",
                        linewidth=1.0,
                        marker="x",
                        markevery=max(1, len(mesh) // 16),
                        markersize=4,
                        markerfacecolor="white",
                        markeredgewidth=1.2,
                        label=f"Baseline Model ({len(mesh)} pts)",
                        zorder=3,
                        alpha=0.8,
                    )

        all_data = np.concatenate([mean_profile_dns, les_mean])
        y_min, y_max = all_data.min(), all_data.max()
        y_range = y_max - y_min if y_max != y_min else 1.0
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # Dynamic Title
        if env is not None:
            title_tag = f"(Episode {episode if episode is not None else 0})"
        elif isinstance(solver, SolverCoupled):
            title_tag = "Evaluation"
        elif isinstance(solver, SolverBase):
            title_tag = "Baseline"
        else:
            title_tag = ""

        ax.set_title(
            rf"Mean Velocity Profile Comparison $\langle w \rangle$ {title_tag}".strip(),
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

        # 5. Resolve Output Directory & File Name
        if save_dir is not None:
            out_dir = save_dir
        elif env is not None:
            out_dir = self.master_path / "profile_comparisons"
        elif isinstance(solver, SolverCoupled):
            out_dir = self.master_path / "evaluation"
        elif isinstance(solver, SolverBase):
            out_dir = self.master_path / "baseline"
        else:
            out_dir = self.master_path

        out_dir.mkdir(parents=True, exist_ok=True)

        if env is not None:
            filename = (
                f"profile_comparison_ep{episode if episode is not None else 0:03d}.png"
            )
        elif isinstance(solver, SolverCoupled):
            filename = "profile_comparison_evaluation.png"
        elif isinstance(solver, SolverBase):
            filename = "profile_comparison_baseline.png"
        else:
            filename = "profile_comparison.png"

        output_plot_path = out_dir / filename

        plt.savefig(output_plot_path, dpi=300)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        self._log(f"  * Saved velocity profile comparison -> {output_plot_path}")

    def plot_history_breakdown(
        self, env: EnvironmentForcingDNS, show_plot: bool = False
    ) -> None:
        """Plot the evolution of raw and weighted reward components across environment steps."""

        # 1. Align all history lengths to avoid sharex distortion
        min_len = len(env.total_reward_history_scaled)
        if min_len == 0:
            self._log("No reward history available to plot.")
            return

        # Adapt moving average window size to the episode length
        window_size = min(20, max(1, min_len // 4))
        burn_in_steps = getattr(self.hp, "burn_in_steps", 0)

        # Clean array slices up to current episode step count
        steps = np.arange(min_len)

        # -------------------------------------------------------------
        # 1. RAW METRICS PLOT
        # -------------------------------------------------------------
        histories_raw = {
            "Action Penalty (raw)": (
                env.action_penalty_history_raw[:min_len],
                "tab:red",
            ),
            "Spectral Penalty (raw)": (
                env.spectral_penalty_history_raw[:min_len],
                "tab:purple",
            ),
            "Distance Error (raw)": (
                env.distance_error_history_raw[:min_len],
                "tab:blue",
            ),
            "Distance Improvement (raw)": (
                env.distance_improvement_history_raw[:min_len],
                "tab:green",
            ),
        }

        print(env.distance_error_history_raw)
        print(env.distance_improvement_history_raw)

        fig_raw, axes_raw = plt.subplots(
            4, 1, figsize=(10, 10), sharex=True, layout="constrained", dpi=300
        )

        for ax, (title, (data, color)) in zip(axes_raw, histories_raw.items()):
            if len(data) == 0:
                ax.text(
                    0.5, 0.5, f"No data recorded for {title}", ha="center", va="center"
                )
                continue

            data_arr = np.array(data)

            ax.plot(
                steps[: len(data_arr)],
                data_arr,
                color=color,
                alpha=0.35,
                linewidth=1.0,
                label="Raw Step Value",
            )

            if len(data_arr) >= window_size and window_size > 1:
                moving_avg = np.convolve(
                    data_arr, np.ones(window_size) / window_size, mode="valid"
                )
                ax.plot(
                    steps[window_size - 1 : len(data_arr)],
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

        # -------------------------------------------------------------
        # 2. WEIGHTED METRICS & TOTAL REWARDS PLOT
        # -------------------------------------------------------------
        fig_weighted, axes_weighted = plt.subplots(
            5, 1, figsize=(10, 12), sharex=True, layout="constrained", dpi=300
        )

        ax_total = axes_weighted[0]

        if env.total_reward_history_unscaled:
            unscaled_arr = np.array(env.total_reward_history_unscaled[:min_len])
            ax_total.plot(
                steps, unscaled_arr, color="tab:orange", alpha=0.2, linewidth=0.8
            )
            if len(unscaled_arr) >= window_size and window_size > 1:
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
            scaled_arr = np.array(env.total_reward_history_scaled[:min_len])
            ax_total.plot(
                steps, scaled_arr, color="royalblue", alpha=0.2, linewidth=0.8
            )
            if len(scaled_arr) >= window_size and window_size > 1:
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
                env.action_penalty_history_weighted[:min_len],
                "tab:red",
            ),
            "Spectral Penalty (weighted)": (
                env.spectral_penalty_history_weighted[:min_len],
                "tab:purple",
            ),
            "Distance Error (weighted)": (
                env.distance_error_history_weighted[:min_len],
                "tab:blue",
            ),
            "Distance Improvement (weighted)": (
                env.distance_improvement_history_weighted[:min_len],
                "tab:green",
            ),
        }

        for ax, (title, (data, color)) in zip(
            axes_weighted[1:], histories_weighted_components.items()
        ):
            if len(data) == 0:
                ax.text(
                    0.5, 0.5, f"No data recorded for {title}", ha="center", va="center"
                )
                continue

            data_arr = np.array(data)

            ax.plot(
                steps[: len(data_arr)],
                data_arr,
                color=color,
                alpha=0.35,
                linewidth=1.0,
                label="Weighted Step Value",
            )

            if len(data_arr) >= window_size and window_size > 1:
                moving_avg = np.convolve(
                    data_arr, np.ones(window_size) / window_size, mode="valid"
                )
                ax.plot(
                    steps[window_size - 1 : len(data_arr)],
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

        if show_plot:
            plt.show()
        else:
            plt.close(fig_raw)
            plt.close(fig_weighted)
