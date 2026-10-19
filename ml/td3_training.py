import random
import time
from dataclasses import fields, replace
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from ml.ann import ReplayBuffer, TauANN, save_tau_ann
from ml.ann_config import TauANNConfig, TD3Hyperparameters
from ml.diagnostics import plot_diagnostic_metrics
from ml.environment import EnvironmentForcingDNS
from ml.td3_agent import TD3Agent
from setup.constants import PATH_TO_LES_CACHE, N_NODES_DNS
from utils.les_caching import (
    LESCacheKey,
    resolve_les_baseline_cache,
    LESCacheStatus,
    write_les_parameters,
)
from ml.reference_scheduler import ReferenceTrajectory
from proofing.action_based.config import ANNConfigProof
from proofing.action_based.environment import EnvironmentProof
from proofing.action_based.solver import SolverForProofs
from proofing.contextual_bandit_reformulation.config import (
    ANNBanditConfig,
    TD3BanditHyperparameters,
)
from proofing.contextual_bandit_reformulation.environment import (
    EnvironmentBanditTraining,
)
from proofing.short_mdp.config import ShortMDPConfig, TD3ShortMDPHyperparameters
from proofing.short_mdp.environment import EnvironmentShortMDP
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SimulationMode, SolverBase
from solvers.solver_coupled import SolverCoupled
from utils.diagnostics import compute_error_metrics

EPISODE_PENALTY_CLIP = -10000

# --- Publication-Quality Global Settings ---
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
    }
)


class TD3Trainer:
    """Training wrapper for the TD3 training pipeline with plain-text logging."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        ann_config: TauANNConfig | ANNConfigProof | ANNBanditConfig | ShortMDPConfig,
        master_path: Path,
        reference_trajectory: ReferenceTrajectory,
        hp: TD3Hyperparameters
        | TD3BanditHyperparameters
        | TD3ShortMDPHyperparameters
        | None = None,
        baseline_path: Path | None = None,
        train_intervals: list[tuple[float, float]] | None = None,
        test_interval: tuple[float, float] | None = None,
    ):

        if hp is None:
            if isinstance(ann_config, ANNBanditConfig):
                self.hp = TD3BanditHyperparameters()
            elif isinstance(ann_config, ShortMDPConfig):
                self.hp = TD3ShortMDPHyperparameters()
            else:
                self.hp = TD3Hyperparameters()

        self.problem = replace(
            problem, initial_condition=reference_trajectory.initial_condition
        )
        self.disc_config = disc_config
        self.ann_config: (
            TauANNConfig | ANNConfigProof | ANNBanditConfig | ShortMDPConfig
        ) = ann_config
        self.master_path = Path(master_path)
        self.reference_trajectory = reference_trajectory
        self.baseline_dir = (
            Path(baseline_path) if baseline_path else Path(PATH_TO_LES_CACHE)
        )
        self.baseline_run_dir: Path | None = None
        self.baseline_test_dir: Path | None = None

        self.train_intervals = train_intervals
        self.test_interval = test_interval

        if self.train_intervals:
            self.interval_counts = np.zeros(len(self.train_intervals))
        else:
            self.interval_counts = np.array([])

        # Initialize reference trajectory interval if train intervals are provided
        if self.train_intervals and len(self.train_intervals) > 0:
            init_t_start, init_t_end = self.train_intervals[0]
            self.reference_trajectory.set_active_training_interval(
                init_t_start, init_t_end
            )
            self.problem = replace(
                self.problem,
                t_start=init_t_start,
                t_end=init_t_end,
                initial_condition=self.reference_trajectory.initial_condition,
            )

        self.episodes_ran: int = 0

        self.baseline_reward: float | None = None
        self.end_of_random_episode: int | None = None
        self.episode_reward_history: list = []

        self.best_historical_reward = -np.inf
        self.best_action_sequence: list = []

        if hasattr(self.ann_config, "n_random_episodes"):
            stochastic_episodes = self.ann_config.n_random_episodes
        else:
            stochastic_episodes = ceil(
                self.hp.stochastic_timesteps / self.ann_config.n_agent_steps_per_episode
            )
        self.n_stochastic_episodes = stochastic_episodes

        self.solver: SolverCoupled | SolverBase | SolverForProofs | None = None
        self.env: (
            EnvironmentForcingDNS | EnvironmentProof | EnvironmentBanditTraining | None
        ) = None

        # File logging setup
        self.master_path.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.master_path / "training_log.txt"
        self._init_txt_log()

        self.w_l2_error_history: list[float] = []
        self.w_rms_error_history: list[float] = []

        self.reynolds_l2_error_history: list[float] = []
        self.reynolds_rms_error_history: list[float] = []

        self.baseline_w_l2_error: float = 0
        self.baseline_w_rms_error: float = 0
        self.baseline_w_rel_error: float = 0

        self.baseline_reynolds_l2_error: float = 0
        self.baseline_reynolds_rms_error: float = 0
        self.baseline_reynolds_rel_error: float = 0

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
        elif isinstance(self.ann_config, ShortMDPConfig):
            env = EnvironmentShortMDP(
                problem=self.problem,
                disc_config=self.disc_config,
                ann_config=self.ann_config,
                hyperparameters=self.hp,
                reference_trajectory=self.reference_trajectory,
                master_path=self.master_path,
                baseline_path=None,
            )
        elif isinstance(self.ann_config, ANNBanditConfig):
            env = EnvironmentBanditTraining(
                problem=self.problem,
                disc_config=self.disc_config,
                ann_config=self.ann_config,
                hyperparameters=self.hp,
                reference_trajectory=self.reference_trajectory,
                master_path=self.master_path,
                baseline_path=None,
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
        self.env.reset()
        agent = TD3Agent(ann_config=self.ann_config, hp=self.hp)

        replay_buffer = ReplayBuffer(
            state_dim=self.ann_config.state_dimension,
            action_dim=self.ann_config.action_dimension,
            max_size=self.hp.replay_buffer_max_size,
        )

        self.print_configurations(agent)

        self.print_title("Starting Training Loop")

        total_steps = 0
        max_steps_per_ep = getattr(self.ann_config, "n_agent_steps_per_episode", 1000)
        training_start_time = time.time()

        episode = 0
        stop_training = False

        is_bandit = isinstance(self.ann_config, ANNBanditConfig)
        worst_valid_reward = None
        if is_bandit:
            edges = np.linspace(
                self.ann_config.min_action,
                self.ann_config.max_action,
                self.ann_config.n_random_episodes + 1,
            )
            random_values = np.random.permutation(
                np.random.uniform(edges[:-1], edges[1:])
            )

        while episode < self.ann_config.n_training_episodes and not stop_training:
            if self.train_intervals:
                idx = random.randrange(len(self.train_intervals))
                self.interval_counts[idx] += 1
                ep_t_start, ep_t_end = self.train_intervals[idx]

                self.reference_trajectory.set_active_training_interval(
                    ep_t_start, ep_t_end
                )
                self.problem = replace(
                    self.problem,
                    t_start=ep_t_start,
                    t_end=ep_t_end,
                    initial_condition=self.reference_trajectory.initial_condition,
                )
                self.disc_config = replace(self.disc_config, problem=self.problem)
                env.problem = self.problem

                self.run_baseline_evaluation()

                if isinstance(self.ann_config, ANNBanditConfig):
                    env.baseline_path = self.baseline_run_dir
                    self.env.compute_baseline_reward(baseline_path=env.baseline_path)

                self._log(
                    f"  [INTERVAL] Active Episode Interval: [{ep_t_start} - {ep_t_end}] (span: {ep_t_end - ep_t_start:.2f}s)"
                )

            ep_start_time = time.time()
            state = env.reset()
            initial_state = state.copy()
            episode_reward = 0.0
            done = False
            episode_steps = 0
            actions = []
            skip_episode = False

            is_random_action_episode = episode < self.n_stochastic_episodes

            self._log_episode_header(episode, total_steps)

            while not done:
                total_steps += 1
                episode_steps += 1

                if is_random_action_episode:
                    min_action, max_action = self.get_action_bounds(episode=episode)
                    if isinstance(self.ann_config, ANNBanditConfig) and getattr(
                        self.ann_config, "link_coefficients", False
                    ):
                        action = np.full(
                            self.ann_config.action_dimension, random_values[episode]
                        )
                    else:
                        action = np.random.uniform(
                            min_action,
                            max_action,
                            size=self.ann_config.action_dimension,
                        )
                else:
                    for _ in range(self.hp.updates_per_step):
                        critic_loss, actor_loss = agent.train(
                            replay_buffer, self.hp.batch_size
                        )
                        self.critic_loss_history.append(critic_loss)
                        self.actor_loss_history.append(actor_loss)

                    action = agent.select_action(state, noise_std=self.hp.expl_noise)
                    if isinstance(self.ann_config, ANNBanditConfig) and getattr(
                        self.ann_config, "link_coefficients", True
                    ):
                        action = np.full(self.ann_config.action_dimension, action[0])

                actions.append(action)
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        next_state, reward, done, info = env.step(action=action)

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
                    if is_bandit:
                        crash_reward = (
                            reward if worst_valid_reward is None else worst_valid_reward
                        )
                        replay_buffer.add(state, action, next_state, crash_reward, True)

                    reason = info.get("crash_reason", "Solver Divergence")
                    stop_training = self._handle_episode_abort(
                        f"Episode {episode + 1} aborted: {reason}"
                    )
                    skip_episode = True
                    break

                replay_buffer.add(state, action, next_state, reward, done)
                state = next_state
                episode_reward += reward
                worst_valid_reward = (
                    reward
                    if worst_valid_reward is None
                    else min(worst_valid_reward, reward)
                )

                if episode_steps % 100 == 0 or done:
                    self._log_step_progress(
                        episode_steps, max_steps_per_ep, reward, episode_reward, env
                    )

                if not is_random_action_episode:
                    critic_loss, actor_loss = agent.train(
                        replay_buffer, self.hp.batch_size
                    )
                    self.critic_loss_history.append(critic_loss)
                    self.actor_loss_history.append(actor_loss)

            if skip_episode:
                episode += 1
                continue

            clipped_reward = float(
                np.clip(episode_reward, a_min=EPISODE_PENALTY_CLIP, a_max=None)
            )
            self.episode_reward_history.append(clipped_reward)

            solver_post_processing = False
            if episode >= self.ann_config.n_training_episodes:
                solver_post_processing = True
            self._execute_episodic_post_processing(
                env, episode, do_solver_post_processing=solver_post_processing
            )

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

            if not is_random_action_episode:
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

        if self.test_interval is not None:
            self.print_title("Starting Testing Run")
            self.run_testing()

        self.plot_reward_evolution()
        self.plot_error_evaluation_evolution()
        self.plot_interval_count()
        self.run_diagnostic_plotting()

        return agent.actor

    def get_action_bounds(self, episode: int) -> tuple[float, float]:
        """Compute action bounds on stochastic action selection, follows linearly varying profile."""
        progress = episode / max(1, self.n_stochastic_episodes)
        current_min = self.ann_config.init_lower_bound_action + progress * (
            self.ann_config.min_action - self.ann_config.init_lower_bound_action
        )
        current_max = self.ann_config.init_upper_bound_action + progress * (
            self.ann_config.max_action - self.ann_config.init_upper_bound_action
        )
        return current_min, current_max

    def _load_baseline_metrics(self) -> None:
        """Helper to load baseline error metrics from cache files on cache hit or miss."""
        if self.baseline_run_dir is None:
            return
        mean_profiles_dir = self.baseline_run_dir / "mean_profiles"
        reynolds_profiles_dir = self.baseline_run_dir / "reynolds_profiles"
        baseline_mean_file = mean_profiles_dir / f"mean_w_{self.problem.t_end}.npy"
        baseline_ww_file = (
            reynolds_profiles_dir / f"reynolds_stress_profile_{self.problem.t_end}.npy"
        )

        mesh_les = self.disc_config.mesh_les
        mean_profile_dns = self.reference_trajectory.target_profile
        reynolds_dns = self.reference_trajectory.reynolds_profile

        if baseline_mean_file.exists():
            baseline_mean = np.load(baseline_mean_file)
            if mesh_les.shape == baseline_mean.shape:
                l2, rms, rel = compute_error_metrics(baseline_mean, mean_profile_dns)
                self.baseline_w_l2_error = l2
                self.baseline_w_rms_error = rms
                self.baseline_w_rel_error = rel

        if baseline_ww_file.exists():
            baseline_ww = np.load(baseline_ww_file)
            if mesh_les.shape == baseline_ww.shape:
                l2, rms, rel = compute_error_metrics(baseline_ww, reynolds_dns)
                self.baseline_reynolds_l2_error = l2
                self.baseline_reynolds_rms_error = rms
                self.baseline_reynolds_rel_error = rel

    def run_baseline_evaluation(self) -> None:
        """Runs baseline evaluation with cache checking logic."""
        cache_key = self._build_les_cache_key()
        cache_result = resolve_les_baseline_cache(self.baseline_dir, cache_key)

        if (
            cache_result.status == LESCacheStatus.HIT
            and cache_result.cache_dir is not None
        ):
            self._log(
                f"[CACHE HIT] Found existing baseline run for time-span [{
                    self.problem.t_start
                } - {self.problem.t_end}] at: {
                    cache_result.cache_dir
                }. Skipping simulation."
            )
            self.baseline_run_dir = cache_result.cache_dir
            self._load_baseline_metrics()
            return

        target_cache_dir = self.baseline_dir / cache_key.dir_to_name()
        target_cache_dir.mkdir(parents=True, exist_ok=True)
        self._log(
            f"[CACHE MISS] Executing baseline run for time-span [{
                self.problem.t_start
            } - {self.problem.t_end}] and caching to: {target_cache_dir}"
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

        (
            baseline_w_l2,
            baseline_w_rms,
            _,
            _,
            baseline_reynolds_l2,
            baseline_reynolds_rms,
            _,
            _,
        ) = self.plot_combined_profile_comparison(
            env=None,
            solver=self.solver,
            episode=None,
            dns_trajectory=self.reference_trajectory.target_profile,
            dns_reynolds_profile=self.reference_trajectory.reynolds_profile,
            raw_dns_trajectory=self.reference_trajectory._raw_target_profile,
            raw_reynolds_profile=self.reference_trajectory._raw_reynolds_profile,
            save_dir=target_cache_dir,
        )

        self._load_baseline_metrics()

    def run_evaluation(self):
        self.solver = SolverCoupled(
            problem=self.problem,
            disc_config=self.disc_config,
            ann_config=replace(self.ann_config, training_mode=False),
            master_path=self.master_path / "evaluation",
            tau_model=self.ann_config.tau_model,
            simulation_mode=SimulationMode.TAU_BASED,
            prescribed_action_trajectory=None,
        )

        self.solver.run_simulation()
        self.solver.post_processing()

        (
            eval_w_l2,
            eval_w_rms,
            baseline_w_l2,
            baseline_w_rms,
            eval_reynolds_l2,
            eval_reynolds_rms,
            baseline_reynolds_l2,
            baseline_reynolds_rms,
        ) = self.plot_combined_profile_comparison(
            env=None,
            solver=self.solver,
            episode=None,
            dns_trajectory=self.reference_trajectory.target_profile,
            dns_reynolds_profile=self.reference_trajectory.reynolds_profile,
            raw_dns_trajectory=self.reference_trajectory._raw_target_profile,
            raw_reynolds_profile=self.reference_trajectory._raw_reynolds_profile,
            evaluation_mode=True,
        )

        self._load_baseline_metrics()

    def run_testing(self) -> None:
        """Runs testing evaluation over the specified test interval using the trained model."""
        if self.test_interval is None:
            self._log("[Warning] No test interval provided. Skipping testing run.")
            return

        t_start, t_end = self.test_interval
        self._log(f"\n[TESTING] Setting test interval to [{t_start} - {t_end}]")

        self.reference_trajectory.set_testing_interval(t_start, t_end)

        self.problem = replace(
            self.problem,
            t_start=t_start,
            t_end=t_end,
            initial_condition=self.reference_trajectory.initial_condition,
        )
        self.disc_config = replace(self.disc_config, problem=self.problem)

        self.run_baseline_evaluation()

        self.solver = SolverCoupled(
            problem=self.problem,
            disc_config=self.disc_config,
            ann_config=replace(self.ann_config, training_mode=False),
            master_path=self.master_path / "testing",
            tau_model=self.ann_config.tau_model,
            simulation_mode=SimulationMode.TAU_BASED,
            prescribed_action_trajectory=None,
        )

        self.solver.run_simulation()
        self.solver.post_processing()

        self.plot_combined_profile_comparison(
            env=None,
            solver=self.solver,
            episode=None,
            dns_trajectory=self.reference_trajectory.target_profile,
            dns_reynolds_profile=self.reference_trajectory.reynolds_profile,
            raw_dns_trajectory=self.reference_trajectory._raw_target_profile,
            raw_reynolds_profile=self.reference_trajectory._raw_reynolds_profile,
            evaluation_mode=True,
            is_testing_plot=True,
        )

        metadata_path = self.master_path / "metadata.txt"
        with open(metadata_path, mode="w") as f:
            f.write(f"Baseline cache key: {self.baseline_run_dir}")

    def plot_combined_profile_comparison(
        self,
        env: EnvironmentForcingDNS | EnvironmentBanditTraining | None,
        solver: SolverCoupled | SolverBase | None,
        episode: int | None,
        show_plot: bool = False,
        dns_trajectory: NDArray | None = None,
        dns_reynolds_profile: NDArray | None = None,
        raw_dns_trajectory: NDArray | None = None,
        raw_reynolds_profile: NDArray | None = None,
        save_dir: Path | None = None,
        evaluation_mode: bool = False,
        is_testing_plot: bool = False,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        """Plot comparison of both mean velocity profile and Reynolds stress profile in a 2x1 vertical subplot grid."""

        # 1. Extract profiles from solver or environment
        try:
            if solver is not None:
                les_mean = solver.calculate_mean_profile()
                les_reynolds = solver.compute_reynolds_stress()
            elif env is not None:
                les_mean = env.solver.calculate_mean_profile()
                les_reynolds = env.solver.compute_reynolds_stress()
            else:
                self._log(
                    "[Error] Skipping profile plot: neither 'solver' nor 'env' was provided."
                )
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        except ValueError as e:
            self._log(f"[Warning] Skipping profile plot: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # 2. Extract reference DNS profiles
        raw_profile_dns = None
        raw_reynolds_dns = None
        if env is not None:
            mean_profile_dns = env.reference_trajectory.target_profile
            reynolds_dns = env.reference_trajectory.reynolds_profile
            if (
                env.reference_trajectory._raw_target_profile is not None
                and env.reference_trajectory._raw_reynolds_profile is not None
            ):
                raw_profile_dns = env.reference_trajectory._raw_target_profile
                raw_reynolds_dns = env.reference_trajectory._raw_reynolds_profile
        elif dns_trajectory is not None and dns_reynolds_profile is not None:
            mean_profile_dns = dns_trajectory
            reynolds_dns = dns_reynolds_profile
            raw_profile_dns = raw_dns_trajectory
            raw_reynolds_dns = raw_reynolds_profile
        else:
            self._log(
                "[Error] Skipping profile plot: missing reference DNS trajectories."
            )
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if (
            les_mean.shape != mean_profile_dns.shape
            or les_reynolds.shape != reynolds_dns.shape
        ):
            raise ValueError(
                "Shape mismatch between LES profiles and DNS reference targets."
            )

        # 3. Compute error metrics
        mean_l2, mean_rmse, mean_rel = compute_error_metrics(les_mean, mean_profile_dns)
        ww_l2, ww_rmse, ww_rel = compute_error_metrics(les_reynolds, reynolds_dns)

        # 4. Generate 2x1 Figure Layout (Top: Mean Velocity, Bottom: Reynolds Stress)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 9), dpi=300, constrained_layout=True
        )

        for ax in (ax1, ax2):
            ax.axhline(
                0, color="black", linestyle="--", linewidth=0.5, alpha=0.4, zorder=1
            )
            ax.minorticks_on()
            ax.tick_params(
                which="both", direction="in", top=True, right=True, length=4, width=0.8
            )
            ax.tick_params(which="minor", length=2, width=0.5)

        # Professional Color Palette (Colorblind-safe & publication-refined)
        color_dns_1, color_les_1 = "#1b4965", "#e76f51"  # Mean Profile
        color_dns_2, color_les_2 = "#2a9d8f", "#b7094c"  # Reynolds Stress
        color_baseline = "#8b5cf6"  # Vibrant Purple (Standout Baseline)

        mesh_les = self.disc_config.mesh_les

        # --- Top Subplot: Mean Velocity Profile ---
        if raw_profile_dns is not None:
            ax1.plot(
                self.disc_config.mesh_dns,
                raw_profile_dns,
                color="grey",
                linestyle="-",
                linewidth=1.0,
                label=f"DNS ({N_NODES_DNS} pts)",
                zorder=2,
                alpha=0.8,
            )
        ax1.plot(
            mesh_les,
            mean_profile_dns,
            color=color_dns_1,
            linestyle="--",
            linewidth=1.6,
            marker="s",
            markevery=max(1, len(mesh_les) // 16),
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=f"Proj. DNS ({len(mesh_les)} pts)",
            zorder=2,
        )
        ax1.plot(
            mesh_les,
            les_mean,
            color=color_les_1,
            linestyle="--",
            linewidth=1.3,
            marker="o",
            markevery=max(1, len(mesh_les) // 16),
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=f"LES Model ({len(mesh_les)} pts)",
            zorder=3,
        )

        # Baseline comparison handling
        baseline_mean_file, baseline_ww_file = None, None
        baseline_mean_l2, baseline_mean_rms, baseline_ww_l2, baseline_ww_rms = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        has_baseline_mean = False
        has_baseline_ww = False

        if evaluation_mode and getattr(self, "baseline_run_dir", None) is not None:
            mean_profiles_dir = self.baseline_run_dir / "mean_profiles"
            reynolds_profiles_dir = self.baseline_run_dir / "reynolds_profiles"
            baseline_mean_file = mean_profiles_dir / f"mean_w_{self.problem.t_end}.npy"
            baseline_ww_file = (
                reynolds_profiles_dir
                / f"reynolds_stress_profile_{self.problem.t_end}.npy"
            )

            if baseline_mean_file.exists():
                baseline_mean = np.load(baseline_mean_file)
                if mesh_les.shape == baseline_mean.shape:
                    baseline_mean_l2, baseline_mean_rms, baseline_mean_rel = (
                        compute_error_metrics(baseline_mean, mean_profile_dns)
                    )
                    has_baseline_mean = True
                    ax1.plot(
                        mesh_les,
                        baseline_mean,
                        color=color_baseline,
                        linestyle="--",
                        linewidth=1.4,
                        marker="^",
                        markevery=max(1, len(mesh_les) // 16),
                        markersize=4.5,
                        markerfacecolor="white",
                        markeredgewidth=1.1,
                        label=f"Baseline Model ({len(mesh_les)} pts)",
                        zorder=3,
                    )

            if baseline_ww_file.exists():
                baseline_ww = np.load(baseline_ww_file)
                if mesh_les.shape == baseline_ww.shape:
                    baseline_ww_l2, baseline_ww_rms, baseline_ww_rel = (
                        compute_error_metrics(baseline_ww, reynolds_dns)
                    )
                    has_baseline_ww = True
                    ax2.plot(
                        mesh_les,
                        baseline_ww,
                        color=color_baseline,
                        linestyle="--",
                        linewidth=1.4,
                        marker="^",
                        markevery=max(1, len(mesh_les) // 16),
                        markersize=4.5,
                        markerfacecolor="white",
                        markeredgewidth=1.1,
                        label=f"Baseline Model ({len(mesh_les)} pts)",
                        zorder=3,
                    )

        # --- Time Interval Label Integration ---
        time_str = f"$t \\in [{self.problem.t_start:.2f}, {self.problem.t_end:.2f}]$"
        title_tag = (
            f"(Episode {episode}) | {time_str}"
            if episode is not None
            else ("Evaluation" if evaluation_mode else f"Baseline | {time_str}")
        )

        ax1.set_title(
            rf"(a) Mean Velocity Profile $\langle w \rangle$ | {title_tag}",
            fontsize=11,
            pad=8,
            loc="left",
        )
        ax1.set_xlabel(r"Wall-normal coordinate $z$", fontsize=11)
        ax1.set_ylabel(r"$\langle w \rangle$", fontsize=11)
        ax1.set_xlim(min(mesh_les), max(mesh_les))
        ax1.legend(
            loc="upper right",
            frameon=True,
            edgecolor="none",
            facecolor="white",
            framealpha=0.9,
        )

        if has_baseline_mean:
            score_text1 = (
                f"$L_2$: {mean_l2:.2e} | RMSE: {mean_rmse:.2e} | "
                f"Rel. Err: {mean_rel:.1f}% (vs. baseline: {baseline_mean_rel:.1f}%)"
            )
        else:
            score_text1 = f"$L_2$: {mean_l2:.2e} | RMSE: {mean_rmse:.2e} | Rel. Err: {mean_rel:.1f}%"

        ax1.text(
            0.5,
            0.03,
            score_text1,
            transform=ax1.transAxes,
            fontsize=8.5,
            horizontalalignment="center",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="square,pad=0.4",
                facecolor="white",
                edgecolor="0.85",
                linewidth=0.6,
            ),
            zorder=5,
        )

        # --- Bottom Subplot: Reynolds Stress Profile ---
        if raw_reynolds_dns is not None:
            ax2.plot(
                self.disc_config.mesh_dns,
                raw_reynolds_dns,
                color="grey",
                linestyle="-",
                linewidth=1,
                label=f"DNS ({N_NODES_DNS} pts)",
                zorder=2,
                alpha=0.8,
            )
        ax2.plot(
            mesh_les,
            reynolds_dns,
            color=color_dns_2,
            linestyle="--",
            linewidth=1.6,
            marker="s",
            markevery=max(1, len(mesh_les) // 16),
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=f"Proj. DNS ({len(mesh_les)} pts)",
            zorder=2,
        )
        ax2.plot(
            mesh_les,
            les_reynolds,
            color=color_les_2,
            linestyle="--",
            linewidth=1.3,
            marker="o",
            markevery=max(1, len(mesh_les) // 16),
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=f"LES Model ({len(mesh_les)} pts)",
            zorder=3,
        )

        ax2.set_title(
            rf"(b) Reynolds Stress Profile $\langle w'w' \rangle$ | {title_tag}",
            fontsize=11,
            pad=8,
            loc="left",
        )
        ax2.set_xlabel(r"Wall-normal coordinate $z$", fontsize=11)
        ax2.set_ylabel(r"$\langle w'w' \rangle$", fontsize=11)
        ax2.set_xlim(min(mesh_les), max(mesh_les))
        ax2.legend(
            loc="upper right",
            frameon=True,
            edgecolor="none",
            facecolor="white",
            framealpha=0.9,
        )

        if has_baseline_ww:
            score_text2 = (
                f"$L_2$: {ww_l2:.2e} | RMSE: {ww_rmse:.2e} | "
                f"Rel. Err: {ww_rel:.1f}% (vs. baseline: {baseline_ww_rel:.1f}%)"
            )
        else:
            score_text2 = (
                f"$L_2$: {ww_l2:.2e} | RMSE: {ww_rmse:.2e} | Rel. Err: {ww_rel:.1f}%"
            )

        ax2.text(
            0.5,
            0.03,
            score_text2,
            transform=ax2.transAxes,
            fontsize=8.5,
            horizontalalignment="center",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="square,pad=0.4",
                facecolor="white",
                edgecolor="0.85",
                linewidth=0.6,
            ),
            zorder=5,
        )

        # 5. Resolve Output Directory & File Name
        if save_dir is not None:
            out_dir = save_dir / "combined_comparisons"
        elif is_testing_plot:
            out_dir = self.master_path / "testing"
        elif env is not None:
            out_dir = self.master_path / "profile_comparisons"
        elif isinstance(solver, SolverCoupled):
            out_dir = self.master_path / "evaluation"
        elif isinstance(solver, SolverBase):
            out_dir = self.master_path / "baseline"
        else:
            out_dir = self.master_path

        out_dir.mkdir(parents=True, exist_ok=True)

        if episode is not None:
            filename = f"profile_comparison_ep{episode:03d}.png"
        elif is_testing_plot:
            filename = "profile_comparison_testing.png"
        elif isinstance(solver, SolverCoupled):
            filename = "profile_comparison_evaluation.png"
        elif isinstance(solver, SolverBase):
            filename = "profile_comparison_baseline.png"
        else:
            filename = "profile_comparison.png"

        output_plot_path = out_dir / filename
        plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        self._log(f"  * Saved 2x1 profile comparison -> {output_plot_path}")

        return (
            mean_l2,
            mean_rmse,
            baseline_mean_l2,
            baseline_mean_rms,
            ww_l2,
            ww_rmse,
            baseline_ww_l2,
            baseline_ww_rms,
        )

    def _execute_episodic_post_processing(
        self, env: Any, episode: int, do_solver_post_processing: bool = False
    ) -> None:
        """Triggers plotting routines and solver post-processing."""
        self._log("-" * 75)
        self._log("  [POST-PROCESSING & PLOTTING ARTIFACTS]")
        self.plot_history_breakdown(env=env, show_plot=False)

        (
            w_l2,
            w_rms,
            _,
            _,
            reynolds_l2,
            reynolds_rms,
            _,
            _,
        ) = self.plot_combined_profile_comparison(
            env=env,
            solver=None,
            episode=episode,
            show_plot=False,
            evaluation_mode=False,
        )

        self.w_l2_error_history.append(w_l2)
        self.w_rms_error_history.append(w_rms)
        self.reynolds_l2_error_history.append(reynolds_l2)
        self.reynolds_rms_error_history.append(reynolds_rms)

        if hasattr(env, "solver") and do_solver_post_processing:
            env.solver.post_processing()

    def run_diagnostic_plotting(self, save_path: Path | None = None) -> Path:
        """Trigger diagnostic plot generation for the trainer instance."""
        if save_path is None:
            save_path = self.master_path / "diagnostics"

        save_path.mkdir(parents=True, exist_ok=True)
        plot_diagnostic_metrics(trainer=self, save_dir=save_path)
        return save_path

    def _build_les_cache_key(self, problem: Problem | None = None) -> LESCacheKey:
        if problem is None:
            problem = self.problem
        return LESCacheKey(
            problem_name=getattr(problem, "name", "unknown_problem"),
            domain_length=getattr(problem, "domain_length", 0.0),
            viscosity=getattr(problem, "viscosity", 0.0),
            bc_type=getattr(problem, "boundary_condition_type", "fixed"),
            bc_value=getattr(problem, "boundary_condition_value", 0),
            n_nodes=getattr(self.disc_config, "n_nodes_les", 0),
            dt=getattr(self.disc_config, "dt_les", 0.0),
            t_start=getattr(problem, "t_start", 0.0),
            t_end=getattr(problem, "t_end", 0.0),
            n_params=getattr(self.ann_config, "n_coefficients", 0),
        )

    def _init_txt_log(self) -> None:
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
        print(message, end=end)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(message + end)

    def _log_episode_header(self, episode: int, total_steps: int) -> None:
        if hasattr(self.ann_config, "n_random_episodes"):
            stochastic_steps = self.ann_config.n_random_episodes
        else:
            stochastic_steps = self.hp.stochastic_timesteps

        phase_str = (
            "EXPLORATION" if total_steps < stochastic_steps else "POLICY TRAINING"
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
        self._log(f"\n- {title} " + "-" * (width - len(title) - 3))

    def print_configurations(
        self, agent: "TD3Agent", print_hyperparameters: bool = True
    ) -> None:
        w = 70
        self.print_row("Master Path", self.master_path, include_brackets=False)

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
        self.print_row("N Agent Calls", self.env.solver.n_agent_calls)
        self.print_row("Agent Interval", self.env.solver.agent_eval_interval)
        self.print_footer(w)

        if print_hyperparameters and hasattr(self, "hp"):
            self.print_section("TD3 Hyperparameters", w)
            if hasattr(self.hp, "__dataclass_fields__"):
                for field in fields(self.hp):
                    key = field.name
                    val = getattr(self.hp, key)
                    self.print_row(key, val)
            self.print_footer(w)

        self.print_section("Time-Block", w)
        self.print_row(
            "Training Range",
            f"[{self.problem.t_start} - {self.problem.t_end}]",
        )
        if self.test_interval:
            self.print_row(
                "Test Range",
                f"[{self.test_interval[0]} - {self.test_interval[1]}]",
            )
        self.print_footer(w)

        if self.train_intervals:
            self.print_section("Stochastic Training Intervals Pool", w)
            for idx, interval in enumerate(self.train_intervals):
                span = interval[1] - interval[0]
                self.print_row(
                    f"Interval {idx + 1}",
                    f"[{interval[0]} - {interval[1]}] ({span:.2f}s)",
                )
            self.print_footer(w)

    def plot_error_evaluation_evolution(self, show_plot: bool = False) -> None:
        if len(self.w_l2_error_history) == 0:
            return

        episodes = np.arange(self.episodes_ran)
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=300, sharex=True)
        color_velocity, color_reynolds = "#2563eb", "#7c3aed"

        axs[0, 0].plot(
            episodes,
            self.w_l2_error_history,
            color=color_velocity,
            linewidth=1.8,
            label="Velocity L2 Error",
        )
        axs[0, 0].axhline(
            self.baseline_w_l2_error,
            color=color_velocity,
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
        )
        axs[0, 0].set_title(
            "Velocity Field - L2 Error", fontsize=11, fontweight="bold", pad=8
        )
        axs[0, 0].grid(True, linestyle=":", alpha=0.6)
        axs[0, 0].legend(frameon=True, facecolor="white", fontsize=8.5)

        axs[0, 1].plot(
            episodes,
            self.reynolds_l2_error_history,
            color=color_reynolds,
            linewidth=1.8,
            label="Reynolds L2 Error",
        )
        axs[0, 1].axhline(
            self.baseline_reynolds_l2_error,
            color=color_reynolds,
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
        )
        axs[0, 1].set_title(
            "Reynolds - L2 Error", fontsize=11, fontweight="bold", pad=8
        )
        axs[0, 1].grid(True, linestyle=":", alpha=0.6)
        axs[0, 1].legend(frameon=True, facecolor="white", fontsize=8.5)

        axs[1, 0].plot(
            episodes,
            self.w_rms_error_history,
            color=color_velocity,
            linewidth=1.8,
            label="Velocity RMS Error",
        )
        axs[1, 0].axhline(
            self.baseline_w_rms_error,
            color=color_velocity,
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
        )
        axs[1, 0].set_title(
            "Velocity Field - RMS Error", fontsize=11, fontweight="bold", pad=8
        )
        axs[1, 0].grid(True, linestyle=":", alpha=0.6)
        axs[1, 0].legend(frameon=True, facecolor="white", fontsize=8.5)

        axs[1, 1].plot(
            episodes,
            self.reynolds_rms_error_history,
            color=color_reynolds,
            linewidth=1.8,
            label="Reynolds RMS Error",
        )
        axs[1, 1].axhline(
            self.baseline_reynolds_rms_error,
            color=color_reynolds,
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
        )
        axs[1, 1].set_title(
            "Reynolds - RMS Error", fontsize=11, fontweight="bold", pad=8
        )
        axs[1, 1].grid(True, linestyle=":", alpha=0.6)
        axs[1, 1].legend(frameon=True, facecolor="white", fontsize=8.5)

        plt.tight_layout()
        save_path = self.master_path / "training_error_evolution.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        self._log(f"  * Saved error evolution plot -> {save_path.name}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def plot_reward_evolution(self, show_plot: bool = False):
        if len(self.episode_reward_history) == 0:
            return

        episodes = np.arange(self.episodes_ran)
        rewards = np.array(self.episode_reward_history)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300, layout="constrained")

        if self.baseline_reward is not None:
            ax.axhline(
                self.baseline_reward, color="crimson", linestyle="--", linewidth=1.5
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
            ax.plot(
                episodes[window_size - 1 :],
                moving_avg,
                color="tab:orange",
                linewidth=2.2,
                label=f"Moving Avg ({window_size} ep)",
            )

        ax.set_title("TD3 Training Reward Evolution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Episode [-]", fontsize=12)
        ax.set_ylabel("Reward [-]", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

        save_path = self.master_path / "training_reward_evolution.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        self._log(f"  * Saved reward evolution plot -> {save_path.name}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def plot_history_breakdown(
        self, env: EnvironmentForcingDNS, show_plot: bool = False
    ) -> None:
        min_len = len(env.total_reward_history_scaled)
        if min_len == 0:
            return
        steps = np.arange(min_len)
        fig_weighted, axes_weighted = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
        axes_weighted.plot(
            steps,
            env.total_reward_history_scaled[:min_len],
            color="royalblue",
            label="Total Reward (Scaled)",
        )
        axes_weighted.set_xlabel("Environment Step [-]")
        axes_weighted.set_ylabel("Reward")
        axes_weighted.grid(True, linestyle="--", alpha=0.4)
        axes_weighted.legend()
        save_path = self.master_path / "reward_components_weighted.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig_weighted)

    def plot_interval_count(self) -> None:
        """Bar chart of training intervals on a timeline with the test interval highlighted."""
        if not self.train_intervals and not self.test_interval:
            raise ValueError("No intervals provided.")

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300, constrained_layout=True)

        # Professional color scheme
        train_color = "#2a6f97"  # Steel Blue
        test_color = "#e07a5f"  # Muted Terracotta

        # 1. Plot training intervals as true timeline bars (width = interval duration)
        if self.train_intervals:
            for i, (start, end) in enumerate(self.train_intervals):
                count = self.interval_counts[i] if i < len(self.interval_counts) else 0
                width = end - start
                ax.bar(
                    start,
                    count,
                    width=width,
                    align="edge",
                    color=train_color,
                    edgecolor="#1d3557",
                    linewidth=0.8,
                    zorder=3,
                    label="Training Interval" if i == 0 else "",
                )

        # 2. Highlight the test interval using a shaded vertical span right where it belongs
        if self.test_interval is not None:
            test_start, test_end = self.test_interval
            ax.axvspan(
                test_start,
                test_end,
                color=test_color,
                alpha=0.35,
                zorder=2,
                label="Test Interval",
            )
            ax.axvline(
                test_start, color=test_color, linestyle="--", linewidth=1.2, zorder=4
            )
            ax.axvline(
                test_end, color=test_color, linestyle="--", linewidth=1.2, zorder=4
            )

        # Major ticks per 1 unit for both X (time) and Y (counts)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))

        # Style consistency with profile comparisons
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.4, zorder=1)
        ax.minorticks_on()
        ax.tick_params(
            which="both", direction="in", top=True, right=True, length=4, width=0.8
        )
        ax.tick_params(which="minor", length=2, width=0.5)

        ax.set_xlabel(r"Time / Simulation Domain ($t$) [s]", fontsize=11)
        ax.set_ylabel("Selection Count [-]", fontsize=11)

        # Grid lines explicitly on major ticks (per 1) for both axes
        ax.grid(True, which="major", linestyle=":", alpha=0.6, zorder=0)

        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            fontsize=9.5,
        )

        # Save to master path
        plot_path = self.master_path / "interval_counts.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self._log(f"  * Saved interval count plot -> {plot_path.name}")
