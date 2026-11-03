import dataclasses
import math
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ml.reference_trajectory import ReferenceTrajectory
from ml.mdp.ann import MDPHyperparameters, MDPConfig, RewardType
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SimulationMode
from solvers.solver_coupled import SolverCoupled
from utils.diagnostics import compute_error_metrics


class MDPEnvironment:
    """Environment for Bandit training."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        ann_config: MDPConfig,
        hyperparameters: MDPHyperparameters,
        reference_trajectory: ReferenceTrajectory,
        master_path: Path,
        baseline_path: Path | None,
    ) -> None:

        self.problem = problem
        self.disc_config = disc_config
        self.ann_config = ann_config
        self.hp = hyperparameters
        self.reference_trajectory = reference_trajectory
        self.master_path = master_path

        self.solver: SolverCoupled | None = None
        self.solver_step_idx: int | None = None
        self.baseline_path = baseline_path

        self.solver_steps_per_agent_call = (
            self.disc_config.n_timesteps // self.ann_config.n_agent_steps_per_episode
        )
        self.previous_action: NDArray | None = None
        self.previous_mean: NDArray | None = None

    def reset(self) -> float:
        self.solver = SolverCoupled(
            problem=self.problem,
            disc_config=dataclasses.replace(
                self.disc_config, suppress_file_logging=True
            ),
            ann_config=dataclasses.replace(
                self.ann_config,
                training_mode=True,
                ann_path=None,
            ),
            master_path=self.master_path,
            simulation_mode=SimulationMode.TAU_BASED,
            tau_model=self.ann_config.tau_model,
            baseline_mode=False,
        )
        self.previous_action: NDArray | None = None
        self.previous_mean: NDArray | None = None
        self.solver_step_idx = self.reference_trajectory._start_idx
        return self.solver.create_input_stencil()

    def step(
        self, action: NDArray, agent_step: int, is_baseline_run: bool = False
    ) -> tuple[float, float, Literal[True]]:
        if self.solver is None:
            raise RuntimeError("Call reset() before step().")

        self.solver.correction_coefficients = action
        for _ in range(self.solver_steps_per_agent_call):
            self.solver.advance_time_step()
            self.solver_step_idx += 1

            if not np.all(np.isfinite(self.solver.solution)):
                raise FloatingPointError(
                    "NaN/Inf detected in intermediate solver step."
                )

            if self.solver.simulation_done:
                break
        is_final_action = self.solver.simulation_done
        reward_val = self.compute_reward(
            action, agent_step, is_final_action, is_baseline_run
        )
        done_flag = self.solver.simulation_done
        next_state = self.solver.create_input_stencil()
        return next_state, reward_val, done_flag

    def compute_reward(self, action, agent_step, is_final_action, is_baseline_run):
        # Normalize reward_type to handle raw strings, shorthands, or enums safely
        r_type = self.ann_config.reward_type
        if isinstance(r_type, str):
            try:
                # Try parsing via the RewardType enum (using our _missing_ handler if configured)
                r_type = RewardType(r_type.lower())
            except ValueError:
                # Fallback manual mapping for shorthands if needed
                mapping = {
                    "s": RewardType.SPARSE,
                    "d": RewardType.DENSE,
                    "l": RewardType.LEGACY,
                    "st": RewardType.SPECTRAL,
                }
                r_type = mapping.get(r_type.lower(), r_type)

        reward_fn = {
            RewardType.SPECTRAL: self._compute_reward_spectral,
            RewardType.DENSE: self._compute_reward_dense,
            RewardType.SPARSE: self._compute_reward_sparse,
        }[r_type]

        return reward_fn(action, agent_step, is_final_action, is_baseline_run)

    def _compute_reward_spectral(
        self, action, agent_step, is_final_action, is_baseline_run
    ):
        if self.solver is None:
            raise RuntimeError("Solver not initialized.")

        instant_solution = self.solver.solution
        instant_reference = self.reference_trajectory.query(
            step_idx=int(self.solver_step_idx)
        )

        _, spectral_solution = self.solver.compute_energy_spectrum_(
            solution=instant_solution
        )
        _, spectral_reference = self.solver.compute_energy_spectrum_(
            solution=instant_reference
        )

        instant_profile_error = self.compute_distance_error(
            field=spectral_solution, target=spectral_reference
        )

        exponential_decay = 1.0
        if self.ann_config.with_exponential_decay:
            exponential_decay = math.exp(
                -2 * agent_step / self.ann_config.n_agent_steps_per_episode
            )

        if is_final_action:

            if not self.ann_config.with_terminal_bonus:
                return -self.ann_config.n_agent_steps_per_episode * self.hp.reward_scale * (self.hp.weight_exponential * exponential_decay * instant_profile_error)

            mean_profile = np.mean(self.solver.snapshots_solution, axis=0)
            mean_reference = self.reference_trajectory.target_profile

            mean_profile_error = self.compute_distance_error(
                mean_profile, mean_reference
            )

            if self.ann_config.include_reynolds_signal:
                reynolds_profile = self.solver.compute_reynolds_stress()
                reynolds_reference = self.reference_trajectory.reynolds_profile

                reynolds_profile_error = self.compute_distance_error(
                    reynolds_profile, reynolds_reference
                )

            if is_baseline_run:
                if self.ann_config.include_reynolds_signal:
                    return mean_profile_error + reynolds_profile_error

                return mean_profile_error

            baseline_profile_error = np.load(self.baseline_path / "reward_history.npy")[
                -1
            ]

            if self.ann_config.include_reynolds_signal:
                return self.hp.reward_scale * (
                    baseline_profile_error - mean_profile_error - reynolds_profile_error
                )

            return self.hp.reward_scale * (baseline_profile_error - mean_profile_error)



        return -self.ann_config.n_agent_steps_per_episode * self.hp.reward_scale * (self.hp.weight_exponential * exponential_decay * instant_profile_error)

    def _compute_reward_sparse(
        self, action, agent_step, is_final_action, is_baseline_run
    ) -> float:
        """Zero reward until the final step; then reward relative to the baseline.

        Baseline runs return the (weighted) profile error, which is stored in
        reward_history.npy. Training runs return
        reward_scale * (baseline_error - error) at the final step.
        """
        if self.solver is None:
            raise RuntimeError("Solver not initialized.")

        if not is_final_action:
            return 0.0

        mean_profile = np.mean(self.solver.snapshots_solution, axis=0)
        error = self.hp.weight_mean_profile * self.compute_distance_error(
            mean_profile, self.reference_trajectory.target_profile
        )

        if self.ann_config.include_reynolds_signal:
            reynolds_profile = self.solver.compute_reynolds_stress()
            error += self.hp.weight_reynolds * self.compute_distance_error(
                reynolds_profile, self.reference_trajectory.reynolds_profile
            )

        if is_baseline_run:
            return error

        baseline_error = np.load(self.baseline_path / "reward_history.npy")[-1]
        return self.hp.reward_scale * (baseline_error - error)

    def _compute_reward_dense(
        self,
        action: NDArray,
        agent_step: int,
        is_final_action: bool,
        is_baseline_run: bool,
    ) -> float:
        if self.solver is None:
            raise RuntimeError("Solver not initialized.")

        # 1. Physical Tracking Error (Evaluated at EVERY step, not just the end)
        current_mean = np.mean(self.solver.snapshots_solution, axis=0)
        current_distance = self.compute_distance_error(
            current_mean, self.reference_trajectory.target_profile
        )

        reynolds_error = 0.0
        if self.ann_config.include_reynolds_signal:
            reynolds_stress = self.solver.compute_reynolds_stress()
            target_reynolds = self.reference_trajectory.reynolds_profile
            reynolds_error = self.compute_distance_error(
                reynolds_stress, target_reynolds
            )

        physics_penalty = (
            self.hp.weight_mean_profile * current_distance
            + self.hp.weight_reynolds * reynolds_error
        )

        # 4. Dense Step-Wise Reward (Negative total cost)
        # We penalize both physical mismatch and erratic/extreme actions at every step
        step_reward = -physics_penalty

        if is_final_action and not is_baseline_run:

            if not self.ann_config.with_terminal_bonus:
                return step_reward

            # Optional bonus/penalty relative to baseline at the terminal step
            try:
                baseline_rewards = np.load(self.baseline_path / "reward_history.npy")
                if len(baseline_rewards) > 0:
                    penalty_baseline = baseline_rewards[-1]
                    terminal_bonus = 5.0 * (-penalty_baseline - physics_penalty)
                    return step_reward + terminal_bonus
            except Exception:
                pass

        return step_reward

    def _compute_reward_legacy(
        self,
        action: NDArray,
        agent_step: int,
        is_final_action: bool,
        is_baseline_run: bool,
    ) -> float:
        if self.solver is None:
            raise RuntimeError("Solver not initialized.")

        base_action = np.ones(self.ann_config.action_dimension)
        regularization = self.compute_distance_error(action, base_action)

        current_mean = np.mean(self.solver.snapshots_solution, axis=0)
        current_distance = self.compute_distance_error(
            current_mean, self.reference_trajectory.target_profile
        )
        if self.previous_mean is not None:
            previous_mean = self.previous_mean
        else:
            self.previous_mean = previous_mean = current_mean

        previous_distance = self.compute_distance_error(
            previous_mean, self.reference_trajectory.target_profile
        )

        distance_error = (
            previous_distance - current_distance
        )  # prev distance is always (prob) going to be greater than current distance as we move closer to the mean

        if self.previous_action is None:
            self.previous_action = action
            smoothness_regularization = 0
        else:
            smoothness_regularization = self.compute_distance_error(
                action, self.previous_action
            )
        action_penalty = float(regularization + smoothness_regularization)

        if self.ann_config.with_improvement_signal:
            # 1. Base tracking cost (we want to minimize distance to target)
            tracking_cost = self.compute_distance_error(
                current_mean, self.reference_trajectory.target_profile
            )

            # 2. Smoothness cost (penalize large action changes between steps)
            if self.previous_action is None:
                smoothness_cost = 0.0
            else:
                smoothness_cost = self.compute_distance_error(
                    action, self.previous_action
                )

            # 3. Effort cost
            effort_cost = self.compute_distance_error(action, base_action)

            # Combined scalar reward (Negative cost = Reward)
            reward = -(
                3.0 * tracking_cost + 0.01 * smoothness_cost + 0.01 * effort_cost
            )
            return reward

        reynolds_error = 0
        if self.ann_config.include_reynolds_signal:
            reynolds_stress = self.solver.compute_reynolds_stress()
            target_reynolds = self.reference_trajectory.reynolds_profile
            reynolds_error = self.compute_distance_error(
                reynolds_stress, target_reynolds
            )

        penalty = (
            self.hp.weight_mean_profile * current_distance
            + self.hp.weight_reynolds * reynolds_error
        )

        if is_final_action:
            if not is_baseline_run:
                penalty_baseline = np.load(self.baseline_path / "reward_history.npy")[
                    -1
                ]
                return 10 * (penalty_baseline - penalty) - (action_penalty * 0.01)

            return penalty
        else:
            return -action_penalty * 0.01

    @staticmethod
    def compute_distance_error(field: NDArray, target: NDArray) -> float:
        """Compute spatial L1 or RMS error (non-squared) between field and target vectors."""
        _, rms_error, _ = compute_error_metrics(field, target)
        return rms_error
