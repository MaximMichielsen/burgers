from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ml.ann_config import TauANNHyperparameters, TD3Hyperparameters
from ml.environment import EnvironmentForcingDNS, CRASH_PENALTY
from ml.reference_scheduler import ReferenceTrajectory
from proofing.contextual_bandit_reformulation.config import ANNBanditConfig
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem


class EnvironmentBanditTraining(EnvironmentForcingDNS):
    """Adjusted environment for testing single action based sweeping training."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        ann_config: ANNBanditConfig,
        hyperparameters: TauANNHyperparameters | TD3Hyperparameters,
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
        self.baseline_w_reward: float | None = None
        self.baseline_path = baseline_path

        self.baseline_reynolds_reward: float | None = None

    def step(self, action: NDArray) -> tuple[NDArray, float, bool, dict]:
        """Set αₙ, advance Nₛₖᵢₚ LES steps, return (sₙ₊₁, rₙ, done, info)."""
        if self.solver is None:
            raise RuntimeError("Call reset() before step().")

        self.solver.correction_coefficients = action
        self.action_history.append(action)
        last_valid_state = self.solver.create_input_stencil(
            mean_profile=self.running_mean_solution
        )
        info = {"crashed": False}

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for _ in range(self.disc_config.n_timesteps):
                    self.solver.advance_time_step()
                    self._total_les_steps += 1

            self.reference_trajectory.set_step_index(self._total_les_steps)

            reward_val = float(self.compute_reward(action))

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

    def compute_reward(self, action: NDArray) -> float:
        """Compute mean target deviation reward."""
        mean_solution = np.mean(self.solver.snapshots_solution, axis=0)
        penalty = self.compute_distance_error(
            field=mean_solution, target=self.reference_trajectory.target_profile
        )
        if self.baseline_w_reward is None:
            self.compute_baseline_reward(baseline_path=self.baseline_path)

        normalized_penalty = self.baseline_w_reward - penalty

        if self.ann_config.include_reynolds_stress_reward_signal:
            reynolds_profile = self.solver.compute_reynolds_stress()
            penalty_reynolds = self.compute_distance_error(
                field=reynolds_profile,
                target=self.reference_trajectory.reynolds_profile,
            )
            normalized_reynolds_penalty = (
                self.baseline_reynolds_reward - penalty_reynolds
            )
            return (
                self.ann_config.w_field_reward_weight * normalized_penalty
                + self.ann_config.reynolds_field_reward_weight
                * normalized_reynolds_penalty
            )

        return normalized_penalty * 100

    def compute_baseline_reward(self, baseline_path: Path) -> None:
        """Compute mean target deviation with baseline."""
        mean_profile = np.load(
            baseline_path / "mean_profiles" / f"mean_w_{self.problem.t_end}.npy"
        )
        reynold_profile = np.load(
            baseline_path
            / "reynolds_profiles"
            / f"reynolds_stress_profile_{self.problem.t_end}.npy"
        )
        penalty_w = self.compute_distance_error(
            field=mean_profile, target=self.reference_trajectory.target_profile
        )
        penalty_reynolds = self.compute_distance_error(
            field=reynold_profile, target=self.reference_trajectory.reynolds_profile
        )
        self.baseline_w_reward = penalty_w
        self.baseline_reynolds_reward = penalty_reynolds
