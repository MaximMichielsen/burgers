import random
import time
from enum import Enum
from dataclasses import replace, fields
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from ml.bandit.ann import (
    BanditANN,
    BanditConfig,
    BanditHyperparameters,
    save_tau_ann,
)
from ml.bandit.environment import BanditEnvironment
from ml.reference_trajectory import (
    ReferenceTrajectory,
)
from ml.diagnostics import plot_diagnostic_metrics
from ml.mdp.ann import MDPConfig, MDPHyperparameters
from ml.mdp.environment import MDPEnvironment
from ml.original.training import set_cache_writable
from ml.replay_buffer import ReplayBuffer
from ml.td3_agent import TD3Agent
from setup.config_discretization import DiscretizationConfig
from setup.constants import PATH_TO_LES_CACHE, N_NODES_DNS
from setup.problems import Problem
from solvers.solver_base import SimulationMode
from solvers.solver_coupled import SolverCoupled
from utils.diagnostics import compute_error_metrics
from utils.les_caching import (
    resolve_les_baseline_cache,
    LESCacheStatus,
    write_les_parameters,
    LESCacheKey,
)

# Reward clip is mode dependent (taken from the latest MDP and the bandit scripts).
REWARD_CLIP_MDP = -1.0
REWARD_CLIP_BANDIT = -0.2
REWARD_CLIP = REWARD_CLIP_MDP  # kept so external imports of REWARD_CLIP still resolve


class TrainingMode(Enum):
    MDP = "mdp"
    BANDIT = "bandit"


def resolve_training_mode(ann_config: Any) -> TrainingMode:
    """Select the training logic from the type of the ANN config."""
    is_mdp = isinstance(ann_config, MDPConfig)
    is_bandit = isinstance(ann_config, BanditConfig)

    if is_mdp and is_bandit:
        # One config class inherits from the other: the most specific class wins.
        mro = type(ann_config).__mro__
        return (
            TrainingMode.MDP
            if mro.index(MDPConfig) < mro.index(BanditConfig)
            else TrainingMode.BANDIT
        )
    if is_mdp:
        return TrainingMode.MDP
    if is_bandit:
        return TrainingMode.BANDIT
    raise TypeError(
        "ann_config must be an MDPConfig or a BanditConfig, "
        f"got {type(ann_config).__name__}."
    )

STYLE_DNS_RAW = dict(color="0.65", linestyle="-", linewidth=1.0, zorder=1)
STYLE_DNS_PROJ = dict(color="black", linestyle=":", linewidth=1.6, zorder=2)
STYLE_BASELINE = dict(
    color="#0072B2",
    linestyle="-.",
    linewidth=1.5,
    marker="^",
    markersize=3.5,
    markerfacecolor="white",
    markeredgewidth=0.9,
    zorder=3,
)
STYLE_IMPROVED = dict(
    color="#D55E00",
    linestyle="--",
    linewidth=1.8,
    marker="o",
    markersize=3.5,
    markerfacecolor="white",
    markeredgewidth=0.9,
    zorder=4,
)


# Set publication style defaults
plt.rcParams.update(
    {
        "font.family": "serif",  # Match your thesis font (serif or sans-serif)
        "mathtext.fontset": "cm",  # Times-compatible math font set
        "font.size": 10,  # Base font size
        "axes.labelsize": 11,  # Axis label size
        "axes.titlesize": 11,  # Title size (if using titles)
        "xtick.labelsize": 9,  # Tick label size
        "ytick.labelsize": 9,  # Tick label size
        "legend.fontsize": 9,  # Legend font size
        "xtick.direction": "in",  # Ticks pointing inward
        "ytick.direction": "in",
        "xtick.top": True,  # Ticks on all four sides for a clean box frame
        "ytick.right": True,
        "axes.linewidth": 0.8,  # Frame line width
        "figure.autolayout": True,  # Tight layout adjustment
    }
)


class Trainer:
    """TD3 trainer. Applies MDP or bandit logic based on the type of `ann_config`."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        ann_config: MDPConfig | BanditConfig,
        master_path: Path,
        reference_trajectory: ReferenceTrajectory,
        train_intervals: list[tuple[float, float]],
        test_interval: tuple[float, float],
        hp: MDPHyperparameters | BanditHyperparameters | None = None,
        baseline_path: Path | None = None,
    ):
        self.mode: TrainingMode = resolve_training_mode(ann_config)
        self.reward_clip: float = (
            REWARD_CLIP_MDP if self.is_mdp else REWARD_CLIP_BANDIT
        )

        hp_cls = MDPHyperparameters if self.is_mdp else BanditHyperparameters
        if hp is None:
            hp = hp_cls()
        elif not isinstance(hp, hp_cls):
            raise TypeError(
                f"{self.mode.value} mode requires {hp_cls.__name__}, "
                f"got {type(hp).__name__}."
            )
        self.hp: MDPHyperparameters | BanditHyperparameters = hp

        self.disc_config = disc_config
        self.ann_config = ann_config
        self.master_path = Path(master_path)
        self.reference_trajectory = reference_trajectory
        self.baseline_dir = (
            Path(baseline_path) if baseline_path else Path(PATH_TO_LES_CACHE)
        )
        self.baseline_run_dir: Path | None = None
        self.baseline_test_dir: Path | None = None
        self.train_intervals = train_intervals
        self.test_interval = test_interval
        self.interval_counts = np.zeros(len(self.train_intervals))
        init_t_start, init_t_end = self.train_intervals[0]
        self.reference_trajectory.set_active_training_interval(init_t_start, init_t_end)
        self.problem = replace(
            problem,
            t_start=init_t_start,
            t_end=init_t_end,
            initial_condition=self.reference_trajectory.initial_condition,
        )
        self.episodes_ran: int = 0
        self.reward_history = []
        self.random_episodes = self.ann_config.n_random_episodes
        self.env: MDPEnvironment | BanditEnvironment | None = None
        self.solver: SolverCoupled | None = None

        self.master_path.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.master_path / "training_log.txt"
        self._init_txt_log()

        self.action_history = []
        self.std_actions = []
        self.mean_actions = []

        self.critic_loss_history = []
        self.actor_loss_history = []
        self.q_value_gradient_history = []
        self.q_sensitivity_history = []

        self.master_plot_path = (
            self.master_path / f"plots_e{self.ann_config.n_training_episodes}"
        )
        self.master_plot_path.mkdir(parents=True, exist_ok=True)

    @property
    def is_mdp(self) -> bool:
        return self.mode is TrainingMode.MDP

    def _build_environment(self) -> MDPEnvironment | BanditEnvironment:
        env_cls = MDPEnvironment if self.is_mdp else BanditEnvironment
        return env_cls(
            problem=self.problem,
            disc_config=self.disc_config,
            ann_config=self.ann_config,
            hyperparameters=self.hp,
            reference_trajectory=self.reference_trajectory,
            master_path=self.master_path,
            baseline_path=None,
        )

    def _env_step(
        self,
        env: MDPEnvironment | BanditEnvironment,
        action: NDArray,
        agent_step: int,
        is_baseline_run: bool,
    ):
        """MDP env.step takes `agent_step`; the bandit env.step does not."""
        if self.is_mdp:
            return env.step(
                action=action,
                agent_step=agent_step,
                is_baseline_run=is_baseline_run,
            )
        return env.step(action=action, is_baseline_run=is_baseline_run)

    def run_training(self) -> BanditANN:
        """Main training loop connecting the environment and TD3 agent."""

        self.print_title("Initializing Training Procedure")
        self._log(f"[MODE] Training logic: {self.mode.value.upper()}")

        # initializing
        env = self._build_environment()
        env.reset()
        self.env = env
        agent = TD3Agent(self.ann_config)

        replay_buffer = ReplayBuffer(
            state_dim=self.ann_config.state_dimension,
            action_dim=self.ann_config.action_dimension,
            max_size=self.hp.replay_buffer_max_size,
        )

        self.print_configurations(agent)
        self.print_title("Starting Training Loop")
        training_start_time = time.time()

        episode = 0

        # Latin Hypercube Sampling of random actions
        edges = np.linspace(
            self.ann_config.min_action,
            self.ann_config.max_action,
            self.random_episodes + 1,
        )
        low, high = edges[:-1], edges[1:]
        random_actions = np.random.uniform(
            low[:, None],
            high[:, None],
            size=(self.random_episodes, self.ann_config.action_dimension),
        )
        for j in range(self.ann_config.action_dimension):
            np.random.shuffle(random_actions[:, j])

        # Training Start
        while episode < self.ann_config.n_training_episodes:
            interval_idx = random.randrange(len(self.train_intervals))
            self.interval_counts[interval_idx] += 1
            ep_t_start, ep_t_end = self.train_intervals[interval_idx]
            self.reference_trajectory.set_active_training_interval(ep_t_start, ep_t_end)
            self.problem = replace(
                self.problem,
                t_start=ep_t_start,
                t_end=ep_t_end,
                initial_condition=self.reference_trajectory.initial_condition,
            )
            self.disc_config = replace(self.disc_config, problem=self.problem)
            env.problem = self.problem
            env.disc_config = self.disc_config

            self.run_baseline_evaluation()

            self._log(
                f"  [INTERVAL] Active Episode Interval: [{ep_t_start} - {ep_t_end}] (span: {ep_t_end - ep_t_start:.2f}s)"
            )

            ep_start_time = time.time()
            state = np.asarray(env.reset())
            is_random_action_episode = episode < self.random_episodes

            done = False
            self._log_episode_header(episode)
            # Episode Start
            episode_steps = 0
            episode_return = 0
            while not done:
                episode_steps += 1

                if is_random_action_episode:
                    action = np.asarray(random_actions[episode])

                else:
                    action = agent.select_action(
                        np.asarray(state), noise_std=self.hp.expl_noise
                    )

                self.action_history.append(action)

                next_state, reward, done = self._env_step(
                    env,
                    action=action,
                    agent_step=episode_steps - 1,
                    is_baseline_run=False,
                )
                episode_return += reward
                if self.is_mdp:
                    print("step: ", episode_steps)
                    print("solver time: ", self.env.solver.time)
                    print(reward)
                scalar_reward = (
                    float(np.squeeze(reward))
                    if hasattr(reward, "__len__")
                    else float(reward)
                )
                clipped_reward = np.clip(
                    scalar_reward, a_min=self.reward_clip, a_max=None
                )
                self.reward_history.append(clipped_reward)

                replay_buffer.add(
                    np.asarray(state),
                    action,
                    np.asarray(next_state),
                    clipped_reward,
                    done,
                )

                state = next_state

                if not is_random_action_episode:
                    for _ in range(self.hp.n_updates_post_episode):
                        critic_loss, actor_loss = agent.train(
                            replay_buffer, self.hp.batch_size
                        )
                        self.critic_loss_history.append(critic_loss)
                        self.actor_loss_history.append(actor_loss)

            solver_post_processing = False
            if episode >= self.ann_config.n_training_episodes:
                solver_post_processing = True
            self._execute_episodic_post_processing(
                env, episode, do_solver_post_processing=solver_post_processing
            )

            self.std_actions = np.std(self.action_history, axis=1)
            self.mean_actions = np.mean(self.action_history, axis=1)

            self._log_episode_summary(
                episode=episode,
                ep_duration=time.time() - ep_start_time,
                reward=episode_return if self.is_mdp else clipped_reward,
                actions=action,
                replay_buffer=replay_buffer,
            )

            if not is_random_action_episode:
                a_default = np.ones(self.ann_config.action_dimension, dtype=np.float32)
                a_perturbed = np.full(
                    self.ann_config.action_dimension, 1.1, dtype=np.float32
                )

                if self.env is not None:
                    state = np.asarray(self.env.solver.create_input_stencil())

                q_default = agent.evaluate_q_value(np.asarray(state), a_default)
                q_perturbed = agent.evaluate_q_value(np.asarray(state), a_perturbed)
                delta_q = float(abs(q_default - q_perturbed))

                self.q_sensitivity_history.append(delta_q)
                self._log(
                    f"  [DIAGNOSTIC] Critic Q(s_0, a=1.0)={q_default:.4f} | "
                    f"Q(s_0, a=1.1)={q_perturbed:.4f} | Delta Q={delta_q:.4f}"
                )

            episode += 1
            self.episodes_ran += 1
            # End of Episode

        # Post-Training Logging and evaluation
        self._log_training_summary(time.time() - training_start_time, episode)

        assert self.ann_config.ann_path is not None
        save_tau_ann(model=agent.actor, save_path=self.ann_config.ann_path)
        self._log(
            f"\n[SUCCESS] Saved trained TauANN model to: {self.ann_config.ann_path}\n"
        )

        if self.test_interval is not None:
            self.print_title("Starting Testing Run")
            self.run_testing()

        self.plot_reward_history()
        self.plot_interval_count()
        self.run_diagnostic_plotting()

        return agent.actor

    # TESTING ---------------------------------------------------------------------------

    def run_testing(self) -> None:
        """Runs testing evaluation over the specified test interval using the trained model."""
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

        # Reset the reference trajectory state back to the test interval endpoint before plotting
        self.reference_trajectory.set_testing_interval(t_start, t_end)

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

    # BASELINE --------------------------------------------------------------------------

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
            set_cache_writable(self.baseline_run_dir, False)
            if self.env is not None:
                self.env.baseline_path = self.baseline_run_dir
            self._load_baseline_metrics()
            return

        target_cache_dir = self.baseline_dir / cache_key.dir_to_name()
        target_cache_dir.mkdir(parents=True, exist_ok=True)
        self._log(
            f"[CACHE MISS] Executing baseline run for time-span [{
                self.problem.t_start
            } - {self.problem.t_end}] and caching to: {target_cache_dir}"
        )
        set_cache_writable(target_cache_dir, True)
        self.baseline_run_dir = target_cache_dir

        self.solver = SolverCoupled(
            problem=self.problem,
            disc_config=self.disc_config,
            master_path=target_cache_dir,
            tau_model=self.ann_config.tau_model,
            simulation_mode=SimulationMode.TAU_BASED,
            ann_config=replace(self.ann_config, ann_path=None, training_mode=False),
            baseline_mode=True,
        )
        if self.env is None:
            raise RuntimeError("Initialize Environment!")

        self.env.reset()
        self.env.solver = self.solver

        self.env.baseline_path = self.baseline_run_dir
        baseline_rewards = []
        done = False
        step = 0
        while not done:
            next_state, reward, done = self._env_step(
                self.env,
                action=np.ones(self.ann_config.action_dimension),
                agent_step=step,
                is_baseline_run=True,
            )
            reward = np.clip(reward, min=self.reward_clip, max=None)
            baseline_rewards.append(reward)
            step += 1

        baseline_rewards = np.asarray(baseline_rewards)
        np.save(self.baseline_run_dir / "reward_history.npy", baseline_rewards)
        with open(self.baseline_run_dir / "reward_history.txt", mode="w") as f:
            for reward in baseline_rewards:
                f.write(str(reward))

        write_les_parameters(target_cache_dir, cache_key)

        self.plot_combined_profile_comparison(
            env=None,
            solver=self.solver,
            episode=None,
            dns_trajectory=self.reference_trajectory.target_profile,
            dns_reynolds_profile=self.reference_trajectory.reynolds_profile,
            raw_dns_trajectory=self.reference_trajectory._raw_target_profile,
            raw_reynolds_profile=self.reference_trajectory._raw_reynolds_profile,
            save_dir=target_cache_dir,
            is_baseline_run=True,
        )

        self._load_baseline_metrics()
        set_cache_writable(self.baseline_run_dir, False)

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

    def _build_les_cache_key(self, problem: Problem | None = None) -> LESCacheKey:
        if problem is None:
            problem = self.problem
        extra_key_fields: dict[str, Any] = {}
        if self.is_mdp:
            extra_key_fields["with_improvement_signal"] = float(
                getattr(self.ann_config, "with_improvement_signal", False)
            )
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
            n_agent_steps_per_episode=getattr(
                self.ann_config, "n_agent_steps_per_episode", 0
            ),
            use_reynolds_signal=self.ann_config.include_reynolds_signal,
            **extra_key_fields,
        )

    # PLOTTING --------------------------------------------------------------------------

    def plot_combined_profile_comparison(
        self,
        env: MDPEnvironment | BanditEnvironment | None,
        solver: SolverCoupled | None,
        episode: int | None,
        show_plot: bool = False,
        dns_trajectory: NDArray | None = None,
        dns_reynolds_profile: NDArray | None = None,
        raw_dns_trajectory: NDArray | None = None,
        raw_reynolds_profile: NDArray | None = None,
        save_dir: Path | None = None,
        evaluation_mode: bool = False,
        is_testing_plot: bool = False,
        is_baseline_run: bool = False,
    ) -> tuple[float, float, float, float, float, float, float, float]:

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

        # 3. Compute error metrics
        mean_l2, mean_rmse, mean_rel = compute_error_metrics(les_mean, mean_profile_dns)
        ww_l2, ww_rmse, ww_rel = compute_error_metrics(les_reynolds, reynolds_dns)

        # 4. Generate 1x2 figure layout (left: mean velocity, right: Reynolds stress)
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10, 3.5), dpi=300, constrained_layout=True
        )

        for ax in (ax1, ax2):
            ax.axhline(
                0, color="black", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0
            )
            ax.minorticks_on()
            ax.tick_params(
                which="both", direction="in", top=True, right=True, length=4, width=0.8
            )
            ax.tick_params(which="minor", length=2, width=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

        mesh_les = self.disc_config.mesh_les
        mark_every = max(1, len(mesh_les) // 16)

        # --- Left Subplot: Mean Velocity Profile ---
        if raw_profile_dns is not None:
            ax1.plot(
                self.disc_config.mesh_dns,
                raw_profile_dns,
                label=f"DNS ({N_NODES_DNS} pts)",
                **STYLE_DNS_RAW,
            )
        ax1.plot(
            mesh_les,
            mean_profile_dns,
            label=f"Proj. DNS ({len(mesh_les)} pts)",
            **STYLE_DNS_PROJ,
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
                        markevery=mark_every,
                        label=f"Baseline Model ({len(mesh_les)} pts)",
                        **STYLE_BASELINE,
                    )
        style = STYLE_IMPROVED if not is_baseline_run else STYLE_BASELINE
        ax1.plot(
            mesh_les,
            les_mean,
            markevery=mark_every,
            label=f"Corrected Model ({len(mesh_les)} pts)" if not is_baseline_run else f"Base Model ({len(mesh_les)} pts)",
            **style,
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
            fontsize=10,
            pad=6,
            loc="left",
        )
        ax1.set_ylabel(r"$\langle w \rangle$", fontsize=10)
        ax1.set_xlim(min(mesh_les), max(mesh_les))
        ax1.set_xlabel("Wall-normal coordinate $z$", fontsize=9)

        if has_baseline_mean:
            score_text1 = (
                f"RMSE: {mean_rmse:.2e} | "
                f"Rel. Err: {mean_rel:.1f}% (vs. {baseline_mean_rel:.1f}%)"
            )
        else:
            score_text1 = f"$L_2$: {mean_l2:.2e} | RMSE: {mean_rmse:.2e} | Rel. Err: {mean_rel:.1f}%"

        # Professional text box at the bottom of ax1
        ax1.text(
            0.5,
            0.06,
            score_text1,
            transform=ax1.transAxes,
            fontsize=8,
            ha="center",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="0.8",
                alpha=0.9,
                linewidth=0.8,
            ),
        )

        # --- Right Subplot: Reynolds Stress Profile ---
        if raw_reynolds_dns is not None:
            ax2.plot(
                self.disc_config.mesh_dns,
                raw_reynolds_dns,
                label=f"DNS ({N_NODES_DNS} pts)",
                **STYLE_DNS_RAW,
            )
        ax2.plot(
            mesh_les,
            reynolds_dns,
            label=f"Proj. DNS ({len(mesh_les)} pts)",
            **STYLE_DNS_PROJ,
        )
        if evaluation_mode and getattr(self, "baseline_run_dir", None) is not None:
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
                        markevery=mark_every,
                        label=f"Baseline Model ({len(mesh_les)} pts)",
                        **STYLE_BASELINE,
                    )

        ax2.plot(
            mesh_les,
            les_reynolds,
            markevery=mark_every,
            label=f"Corrected Model ({len(mesh_les)} pts)",
            **style,
        )

        ax2.set_title(
            rf"(b) Reynolds Stress Profile $\langle w'w' \rangle$ | {title_tag}",
            fontsize=10,
            pad=6,
            loc="left",
        )
        ax2.set_ylabel(r"$\langle w'w' \rangle$", fontsize=10)
        ax2.set_xlim(min(mesh_les), max(mesh_les))
        ax2.set_xlabel("Wall-normal coordinate $z$", fontsize=9)

        if has_baseline_ww:
            score_text2 = (
                f"RMSE: {ww_rmse:.2e} | "
                f"Rel. Err: {ww_rel:.1f}% (vs. {baseline_ww_rel:.1f}%)"
            )
        else:
            score_text2 = f"RMSE: {ww_rmse:.2e} | Rel. Err: {ww_rel:.1f}%"

        # Professional text box at the bottom of ax2
        ax2.text(
            0.5,
            0.06,
            score_text2,
            transform=ax2.transAxes,
            fontsize=8,
            ha="center",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="0.8",
                alpha=0.9,
                linewidth=0.8,
            ),
        )

        # Shared legend below both panels
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = (h1, l1) if len(h1) >= len(h2) else (h2, l2)
        fig.legend(
            handles,
            labels,
            loc="outside lower center",  # requires matplotlib >= 3.7
            ncol=len(handles),
            fontsize=9,
            frameon=False,
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
        else:
            out_dir = self.master_path

        out_dir.mkdir(parents=True, exist_ok=True)

        if episode is not None:
            filename = f"profile_comparison_ep{episode:03d}.png"
        elif is_testing_plot:
            filename = f"profiles_e{self.ann_config.n_training_episodes}.png"
        elif isinstance(solver, SolverCoupled):
            filename = "profile_comparison_evaluation.png"
        else:
            filename = "profile_comparison.png"

        output_plot_path = out_dir / filename
        plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")

        if is_testing_plot:
            plt.savefig(self.master_plot_path / filename, dpi=300, bbox_inches="tight")
            plt.savefig(
                self.master_plot_path / "profiles.pdf", dpi=300, bbox_inches="tight"
            )

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        self._log(f"  * Saved 1x2 profile comparison -> {output_plot_path}")

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

    def plot_reward_history(self):
        if len(self.reward_history) == 0:
            return

        rewards = np.array(self.reward_history)
        steps = np.arange(len(rewards))

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300, layout="constrained")

        ax.plot(
            steps,
            rewards,
            color="royalblue",
            alpha=0.3,
            linewidth=1.0,
            label="Stepwise Reward",
        )
        window_size = 10
        if len(rewards) >= window_size:
            moving_avg = np.convolve(
                rewards, np.ones(window_size) / window_size, mode="valid"
            )
            ax.plot(
                steps[window_size - 1 :],
                moving_avg,
                color="royalblue",
                linewidth=2.2,
                label=f"Moving Avg ({window_size} ep)",
            )

        ax.set_title("Stepwise rewards")
        ax.set_xlabel("Step [-]")
        ax.set_ylabel("Reward [-]")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(framealpha=0.9, fontsize=9.5)

        save_path = self.master_path / "step_rewards.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.savefig(
            self.master_plot_path / "step_rewards.png", dpi=300, bbox_inches="tight"
        )
        self._log(f"  * Saved reward evolution plot -> {save_path.name}")
        plt.close(fig)

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
        fig.savefig(
            self.master_plot_path / "interval_counts.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        self._log(f"  * Saved interval count plot -> {plot_path.name}")

    def run_diagnostic_plotting(self, save_path: Path | None = None) -> Path:
        """Trigger diagnostic plot generation for the trainer instance."""
        if save_path is None:
            save_path = (
                self.master_path / f"diagnostics_e{self.ann_config.n_training_episodes}"
            )

        save_path.mkdir(parents=True, exist_ok=True)
        plot_diagnostic_metrics(trainer=self, save_dir=save_path)
        return save_path

    def _execute_episodic_post_processing(
        self, env: Any, episode: int, do_solver_post_processing: bool = False
    ) -> None:
        """Triggers plotting routines and solver post-processing."""
        self._log("-" * 75)
        self._log("  [POST-PROCESSING & PLOTTING ARTIFACTS]")
        self.plot_combined_profile_comparison(
            env=env,
            solver=None,
            episode=episode,
            show_plot=False,
            evaluation_mode=False,
        )

        if hasattr(env, "solver") and do_solver_post_processing:
            env.solver.post_processing()

    # PRINTING --------------------------------------------------------------------------

    def print_title(self, title: str, width: int = 90) -> None:
        self._log(f"\n- {title} " + "-" * (width - len(title) - 3))

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

    def _log_episode_header(self, episode: int) -> None:
        phase_str = (
            "EXPLORATION" if episode < self.random_episodes else "POLICY TRAINING"
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
        reward: float,
        actions: list | NDArray,
        replay_buffer: Any,
    ) -> None:
        mean_action = float(np.mean(actions)) if len(actions) > 0 else 0.0

        self._log("-" * 75)
        self._log(
            f"  [EPISODE {episode + 1} SUMMARY]\n"
            f"  Duration      : {ep_duration:.2f}s\n"
            f"  Mean Action   : {mean_action:.4f}\n"
            f"  Reward        : {float(reward):.4f}\n"
            f"  Replay Buffer : {replay_buffer.size}/{self.hp.replay_buffer_max_size}"
        )
        self._log("=" * 75)

    def _log_training_summary(self, total_duration: float, episode_count: int) -> None:
        rewards = np.array(self.reward_history)
        if len(rewards) == 0:
            return

        first_rwd, best_rwd, final_rwd = rewards[0], rewards.max(), rewards[-1]

        raw_delta_best = float(best_rwd) - float(first_rwd)
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
        self.print_row("Total Elapsed Time", f"{total_duration:.2f}s")
        self.print_row(
            "Avg Time per Episode", f"{total_duration / max(1, episode_count):.2f}s"
        )
        self.print_footer()

        self.print_section("Reward Performance")
        self.print_row("Initial Episode Reward", f"{float(first_rwd):.4f}")
        self.print_row("Best Episode Reward", f"{float(best_rwd):.4f}")
        self.print_row("Final Episode Reward", f"{final_rwd:.4f}")
        self.print_row("Mean Reward (All Ep)", f"{rewards.mean():.4f}")
        self.print_footer()

        self.print_section("Reward Improvement Metrics")
        self.print_row(
            "Initial -> Best Delta",
            f"{raw_delta_best:+.4f} ({float(pct_imp_best):+.2f}%)",
        )
        self.print_row(
            "Initial -> Final Delta",
            f"{raw_delta_final:+.4f} ({float(pct_imp_final):+.2f}%)",
        )
        self.print_footer()

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
        self.print_row(
            "Action Range",
            f"[{self.ann_config.min_action}, {self.ann_config.max_action}]",
        )
        self.print_row("Training Episodes", self.ann_config.n_training_episodes)
        if self.is_mdp:
            self.print_row("Agent Calls", self.ann_config.n_agent_steps_per_episode)
            self.print_row("Agent n_skip", self.env.solver_steps_per_agent_call)
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
            "Training Intervals",
            "",
        )
        for idx, interval in enumerate(self.train_intervals):
            span = interval[1] - interval[0]
            self.print_row(
                f"Interval {idx + 1}",
                f"[{interval[0]} - {interval[1]}] ({span:.2f}s)",
            )
        if self.test_interval:
            self.print_row(
                "Test Range",
                f"[{self.test_interval[0]} - {self.test_interval[1]}]",
            )
        self.print_footer(w)


# Backwards-compatible aliases (mode is chosen from the ANN config type)
MDPTrainer = Trainer
BanditTrainer = Trainer