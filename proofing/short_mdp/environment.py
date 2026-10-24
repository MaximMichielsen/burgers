from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ml.environment import EnvironmentForcingDNS, CRASH_PENALTY
from ml.reference_scheduler import ReferenceTrajectory
from proofing.short_mdp.config import ShortMDPConfig, TD3ShortMDPHyperparameters
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem

# TODO: could follow from bandit env instead


class EnvironmentShortMDP(EnvironmentForcingDNS):
    """Adjusted environment for testing single action based sweeping training."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        ann_config: ShortMDPConfig,
        hyperparameters: TD3ShortMDPHyperparameters,
        reference_trajectory: ReferenceTrajectory,
        master_path: Path,
        baseline_path: Path,
    ):
        super().__init__(
            problem,
            disc_config,
            ann_config,
            hyperparameters,
            reference_trajectory,
            master_path,
        )

        self.action_history: list[NDArray] = []
        self.baseline_reward: float | None = None
        self.baseline_path = baseline_path
        self.previous_action = np.ones(self.ann_config.action_dimension)

    def step(
        self, action: NDArray, is_baseline_run: bool = False, agent_step_idx: int = 0
    ) -> tuple[NDArray, float, bool, dict]:
        """Set αₙ, advance Nₛₖᵢₚ LES steps, return (sₙ₊₁, rₙ, done, info)."""
        if self.solver is None:
            raise RuntimeError("Call reset() before step().")

        self.solver.correction_coefficients = action
        self.action_history.append(action)
        last_valid_state = self.solver.create_input_stencil(
            mean_profile=self.running_mean_solution
        )
        info = {"crashed": False}

        n_agents = self.ann_config.n_agent_steps_per_episode
        n_total = self.solver.disc_config.n_timesteps  # the solver's own config
        is_last = agent_step_idx + 1 >= n_agents
        steps_this_call = (
            n_total - self._total_les_steps if is_last else n_total // n_agents
        )

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for _ in range(steps_this_call):
                    if (
                        self.solver.simulation_done
                        or self._total_les_steps >= self._max_les_steps
                    ):
                        break

                    self.solver.advance_time_step()
                    self._total_les_steps += 1

            self.reference_trajectory.set_step_index(self._total_les_steps)

            reward_val = float(
                self.compute_reward(
                    action, self.previous_action, is_baseline_run, agent_step_idx
                )
            )
            self.previous_action = action

            done_flag = (
                self._total_les_steps >= self._max_les_steps
                or self.solver.simulation_done
            )
            next_state_array = self.solver.create_input_stencil(
                mean_profile=self.running_mean_solution
            )

            if not np.all(np.isfinite(next_state_array)):
                raise FloatingPointError("NaN/Inf detected in state stencil.")

            info["reward_components"] = getattr(self, "last_reward_components", {})

            return next_state_array, reward_val, done_flag, info

        except (FloatingPointError, ZeroDivisionError, ArithmeticError) as e:
            info["crashed"] = True
            info["crash_reason"] = str(e)

            reward_val = float(CRASH_PENALTY)
            done_flag = True

            fallback_state = np.nan_to_num(
                last_valid_state, nan=0.0, posinf=1.0, neginf=-1.0
            )
            return fallback_state, reward_val, done_flag, info

    def compute_reward(
        self,
        action: NDArray,
        previous_action: NDArray,
        is_baseline_run: bool,
        agent_step_idx: int,
    ) -> float:
        """Compute balanced differential reward with smoothness penalty."""

        # 1. Calculate errors
        mean_error = self.compute_distance_error(
            field=self.solver.mean_solution,
            target=self.reference_trajectory.target_profile,
        )

        reynolds_error = 0.0
        if self.ann_config.include_reynolds_stress_reward_signal:
            reynolds_profile = self.solver.compute_reynolds_stress()
            reynolds_error = self.compute_distance_error(
                field=reynolds_profile,
                target=self.reference_trajectory.reynolds_profile,
            )

        # Weighted combination of errors (guardrail + primary goal)
        total_ann_error = (1.0 * mean_error) + (
            5.0 * reynolds_error
        )  # weights tuned to scale

        # 2. Smoothness penalty to stop erratic coefficient jumps across the 100 steps
        smoothness_penalty = (
            0.1 * np.linalg.norm(action - previous_action) ** 2
            if agent_step_idx > 0
            else 0
        )

        # 3. Baseline comparison (Differential reward)
        if not is_baseline_run:
            baseline_error = self.compute_baseline_reward(
                baseline_path=self.baseline_path, agent_step=agent_step_idx
            )
            # Improvement over baseline (positive means ANN did better than uncorrected model)
            improvement = baseline_error - total_ann_error

            n_agents = self.ann_config.n_agent_steps_per_episode
            if agent_step_idx + 1 >= n_agents:
                # Multiply the final step's improvement by a heavy factor (e.g., 3.0 or 5.0)
                # This gives extra credit for outperforming the baseline right at the finish line!
                improvement *= 3.0  # or improvement + (3.0 * improvement)

            return improvement - smoothness_penalty

        # If it's a baseline run, just return the negative error minus smoothness
        return -total_ann_error - smoothness_penalty

    def compute_baseline_reward(self, baseline_path: Path, agent_step: int) -> None:
        """Compute mean target deviation with baseline."""
        reward_history = np.load(baseline_path / "reward_history.npy")
        return reward_history[agent_step]
