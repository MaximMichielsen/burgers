import dataclasses
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ml.ann_config import TauANNHyperparameters, TD3Hyperparameters
from ml.environment import EnvironmentForcingDNS, CRASH_PENALTY
from ml.reference_scheduler import ReferenceTrajectory
from proofing.ann_config import ANNConfigProof
from proofing.solver import SolverForProofs
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SimulationMode


class EnvironmentProof(EnvironmentForcingDNS):
    """Extends existing environment to adapt target rewards."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        ann_config: ANNConfigProof,
        hyperparameters: TauANNHyperparameters | TD3Hyperparameters,
        reference_trajectory: ReferenceTrajectory,
        master_path: Path,
    ):
        super().__init__(
            problem,
            disc_config,
            ann_config,
            hyperparameters,
            reference_trajectory,
            master_path,
        )

        self.ann_config: ANNConfigProof = ann_config

        self.target_actions_current: NDArray | None = None
        self.target_action_current_fixed: NDArray | None = None
        self.target_actions_mean: NDArray | None = None
        self.current_action_mean: NDArray | None = None

        self.action_current_reward_history: list[float] = []
        self.action_mean_reward_history: list[float] = []

        self.distance_history: list[float] = []

        self.target_actions_current_history: list[NDArray] = []
        self.applied_actions_history: list[NDArray] = []
        self.action_mean_history: list[NDArray] = []

    def reset_solver(self):
        self.solver = SolverForProofs(
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
        )

    def step(self, action: NDArray) -> tuple[NDArray, float, bool, dict]:
        """Set αₙ, advance Nₛₖᵢₚ LES steps, return (sₙ₊₁, rₙ, done, info)."""
        if self.solver is None:
            raise RuntimeError("Call reset() before step().")

        self.applied_actions_history.append(action)
        self.solver.correction_coefficients = action
        last_valid_state = self.solver.create_input_stencil(
            mean_profile=self.running_mean_solution
        )
        info = {"crashed": False}

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for _ in range(self.ann_config.n_skip_steps):
                    if self.ann_config.proof_mode == "c":
                        self.apply_action_target_perturbation(time=self.solver.time)

                    self.solver.advance_time_step()
                    self._total_les_steps += 1

                    # 1. Early check for divergence to avoid running extra corrupted steps
                    current_solution = self.solver.solution  # or relevant state
                    if not np.all(np.isfinite(current_solution)):
                        raise FloatingPointError(
                            "NaN/Inf detected in intermediate solver step."
                        )

                    if self.solver.simulation_done:
                        break

            self.reference_trajectory.set_step_index(self._total_les_steps)

            reward_val = self.compute_reward(action)

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

    def initialize_randomized_target_action(self) -> NDArray:
        """Returns a randomly valued array, the size of the action dimension."""
        return np.random.uniform(
            self.ann_config.lower_bound,
            self.ann_config.upper_bound,
            size=self.ann_config.action_dimension,
        )

    def apply_action_target_perturbation(self, time: float) -> None:
        """Apply a change to the action targets for proof of concept."""
        func = np.sin
        if self.target_action_current_fixed is None:
            self.target_action_current_fixed = (
                self.initialize_randomized_target_action()
            )

        perturb_a = (
            func(self.ann_config.frequency * self.ann_config.omega * time)
            * self.ann_config.amplitude
        )
        perturb_b = (
            func(self.ann_config.frequency * (time + self.ann_config.phase))
            * self.ann_config.amplitude
        )

        perturbations = np.array([perturb_a, perturb_b])
        self.target_actions_current = self.target_action_current_fixed + perturbations
        self.target_actions_current_history.append(self.target_actions_current)

    def target_action_mean(self):
        """Target mean for concept 'd'."""
        return np.array([self.ann_config.target_mean_1, self.ann_config.target_mean_2])

    def compute_reward(self, action: NDArray) -> float:
        """
        Compute a scalar RL reward signal.

        This reward signal serves the function of testing whether the agent actually learn anything from its actions.
        The reward is purely an offset from a randomly initialized action set a_ref_proof (a_1, a_2) where a = {0.8, 1.2}
         but not "near" 1 (or rather the initialized value).
        """
        if self.ann_config.proof_mode == "d":
            self.update_running_action_mean(action=action)
            reward = self.compute_reward_long_horizon()
            self.action_mean_reward_history.append(reward)
            return reward

        elif self.ann_config.proof_mode in ("b", "c"):
            if self.target_actions_current is None:
                self.target_actions_current = self.initialize_randomized_target_action()

            penalty_action = self.compute_distance_error(
                field=action, target=self.target_actions_current
            )
            self.action_current_reward_history.append(penalty_action)
            return -penalty_action

        return None

    def update_running_action_mean(self, action: NDArray) -> None:
        """Update the running action mean with an exponential moving average."""
        action = np.asarray(action, dtype=np.float64)
        smoothing_factor = float(
            getattr(self.ann_config, "smoothing_factor", self.hp.smoothing_factor)
        )
        if self.current_action_mean is None:
            self.current_action_mean = action.copy()
        else:
            self.current_action_mean = (
                1.0 - smoothing_factor
            ) * self.current_action_mean + smoothing_factor * action
        self.action_mean_history.append(self.current_action_mean.copy())

    def compute_reward_long_horizon(self) -> float:
        """Compute reward signal based on improvement and distance from action mean."""
        distance = self.compute_distance_error(
            field=self.current_action_mean, target=self.target_action_mean()
        )

        if not self.distance_history:
            prev_distance = distance
        else:
            prev_distance = self.distance_history[-1]

        self.distance_history.append(distance)
        raw_improvement = float(np.clip(prev_distance - distance, -1.0, 1.0))
        return (
            self.hp.weight_improvement * raw_improvement
            - self.hp.weight_absolute_error * distance
        )
