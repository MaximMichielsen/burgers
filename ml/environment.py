"""Environment for the TauANN coupled with DNS forced solvers."""

import dataclasses
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from final.ml.hyperparameters import TD3Hyperparameters
from final.ml.reference_scheduler import ReferenceTrajectory
from final.ml.tau_ann import TauANNConfig, TauANNHyperparameters
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SimulationMode
from solvers.solver_coupled import SolverCoupled


CRASH_PENALTY = -100


class EnvironmentForcingDNS:
    """MDP wrapper around SolverCoupled with DNS forcing for TauANN training."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        ann_config: TauANNConfig,
        hyperparameters: TauANNHyperparameters | TD3Hyperparameters,
        reference_trajectory: ReferenceTrajectory,
        master_path: Path,
    ) -> None:
        self.problem = problem
        self.disc_config = disc_config
        self.ann_config = ann_config
        self.hp = hyperparameters
        self.master_path = master_path
        self.reference_trajectory = reference_trajectory

        self.solver: SolverCoupled | None = None
        self.running_mean_solution: NDArray | None = None

        self._max_les_steps: int = self.disc_config.n_timesteps
        self._total_les_steps: int = 0

        self.distance_history: list[float] = []

        self.total_reward_history_unscaled: list[float] = []
        self.total_reward_history_scaled: list[float] = []

        self.distance_improvement_history_raw: list[float] = []
        self.distance_error_history_raw: list[float] = []
        self.spectral_penalty_history_raw: list[float] = []
        self.action_penalty_history_raw: list[float] = []

        self.distance_improvement_history_weighted: list[float] = []
        self.distance_error_history_weighted: list[float] = []
        self.spectral_penalty_history_weighted: list[float] = []
        self.action_penalty_history_weighted: list[float] = []

    def reset(self) -> NDArray:
        """Instantiate a fresh solver and return initial state s₀."""
        self.solver = SolverCoupled(
            training_mode=True,
            problem=self.problem,
            disc_config=dataclasses.replace(
                self.disc_config, suppress_file_logging=True
            ),
            ann_config=self.ann_config,
            master_path=self.master_path,
            simulation_mode=SimulationMode.TAU_BASED,
            tau_model=self.ann_config.tau_model,
            ann_path=None,
        )
        self._total_les_steps = 0

        # --- Clear State Tracking Histories ---
        self.distance_history.clear()

        # --- Clear Reward Totals ---
        self.total_reward_history_unscaled.clear()
        self.total_reward_history_scaled.clear()

        # --- Clear Raw Diagnostic Histories ---
        self.distance_improvement_history_raw.clear()
        self.distance_error_history_raw.clear()
        self.spectral_penalty_history_raw.clear()
        self.action_penalty_history_raw.clear()

        # --- Clear Weighted Diagnostic Histories ---
        self.distance_improvement_history_weighted.clear()
        self.distance_error_history_weighted.clear()
        self.spectral_penalty_history_weighted.clear()
        self.action_penalty_history_weighted.clear()

        self.reference_trajectory.reset()
        self.running_mean_solution = self.solver.solution.copy()
        return self.solver.create_input_stencil(mean_profile=self.running_mean_solution)

    def step(self, action: NDArray) -> tuple[NDArray, float, bool]:
        """Set αₙ, advance Nₛₖᵢₚ LES steps, return (sₙ₊₁, rₙ, done)."""
        if self.solver is None:
            raise RuntimeError("Call reset() before step().")

        self.solver.correction_coefficients = action
        last_valid_state = self.solver.create_input_stencil(
            mean_profile=self.running_mean_solution
        )

        try:
            for _ in range(self.ann_config.n_skip_steps):
                self.solver.advance_time_step()
                self._total_les_steps += 1

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

            return next_state_array, reward_val, done_flag

        except (FloatingPointError, ZeroDivisionError, ArithmeticError):
            # Catch solver divergence, apply soft crash penalty, return sanitized state
            reward_val = CRASH_PENALTY
            done_flag = True

            fallback_state = np.nan_to_num(
                last_valid_state, nan=0.0, posinf=1.0, neginf=-1.0
            )
            self.total_reward_history_unscaled.append(reward_val)
            return fallback_state, reward_val, done_flag

    def compute_reward(self, action: NDArray) -> float:
        """Compute the scalar RL reward for the current step."""
        if self.solver is None or self.running_mean_solution is None:
            raise RuntimeError("Solver or running mean solution is not initialized.")

        # --- 1. EWMA Running Profile Update & Distance Computation ---
        self.running_mean_solution = (
            1.0 - self.hp.smoothing_factor
        ) * self.running_mean_solution + self.hp.smoothing_factor * self.solver.solution

        target_profile = self.reference_trajectory.target_profile

        if self._total_les_steps > self.hp.burn_in_steps:
            distance = self.compute_distance_error(
                self.running_mean_solution, target_profile
            )
            if self._total_les_steps == self.hp.burn_in_steps + 1:
                prev_distance = distance
            else:
                prev_distance = (
                    self.distance_history[-1] if self.distance_history else 0.0
                )
        else:
            distance = 0.0
            prev_distance = 0.0

        # Raw physical delta improvement (unweighted)
        raw_improvement = prev_distance - distance

        raw_distance_error = distance**2

        # --- 2. Instantaneous Energy Spectrum Metrics ---
        _, spectrum_les = self.solver.compute_energy_spectrum_(self.solver.solution)
        spectrum_k = spectrum_les.astype(np.float64)

        projected_solution_field = self.reference_trajectory.projected_solution
        _, spectrum_proj = self.solver.compute_energy_spectrum_(
            projected_solution_field
        )
        proj_spectrum_k = spectrum_proj.astype(np.float64)

        if len(spectrum_k) != len(proj_spectrum_k):
            raise ValueError(
                f"Spectrum length mismatch: LES ({len(spectrum_k)}) vs "
                f"Reference ({len(proj_spectrum_k)})."
            )

        if not np.all(np.isfinite(spectrum_k)):
            raise FloatingPointError("NaN/Inf detected in energy spectrum.")

        wavenumber_indices = np.arange(1, len(spectrum_k) + 1, dtype=np.float64)
        normalized_k = wavenumber_indices / len(spectrum_k)
        norm_factor = (np.sum(np.abs(proj_spectrum_k)) ** 2) + 1e-12

        spectral_error = ((spectrum_k - proj_spectrum_k) ** 2) / norm_factor
        unweighted_spectral_error = (normalized_k**self.hp.gamma) * spectral_error
        raw_spectral_error = float(np.sum(unweighted_spectral_error))

        # --- 3. Action Regularization Metric ---
        a_ref = np.ones_like(action)
        raw_action_deviation = self.compute_distance_error(action, a_ref)

        # --- 4. Log Unweighted Physical Metrics (for Diagnostics / Plots) ---
        self.distance_history.append(distance)
        self.distance_improvement_history_raw.append(raw_improvement)
        self.distance_error_history_raw.append(raw_distance_error)
        self.spectral_penalty_history_raw.append(raw_spectral_error)
        self.action_penalty_history_raw.append(raw_action_deviation)

        # --- 5. Apply Reward Weights for TD3 Agent ---
        reward_improvement = self.hp.weight_improvement * raw_improvement
        penalty_absolute_distance = self.hp.weight_absolute_error * (distance**2)
        penalty_spectral = self.hp.weight_spectral * raw_spectral_error
        penalty_action = self.hp.weight_action * raw_action_deviation

        # Assemble composite rewards
        total_penalty = penalty_absolute_distance + penalty_spectral + penalty_action
        reward_total = reward_improvement - total_penalty

        # Scaled reward for policy backpropagation
        scaled_reward = reward_improvement - np.log1p(total_penalty)

        # --- 6. Log Weighted Physical Metrics (for Diagnostics / Plots) ---
        self.distance_improvement_history_weighted.append(reward_improvement)
        self.distance_error_history_weighted.append(penalty_absolute_distance)
        self.spectral_penalty_history_weighted.append(penalty_spectral)
        self.action_penalty_history_weighted.append(penalty_action)

        self.total_reward_history_unscaled.append(reward_total)
        self.total_reward_history_scaled.append(scaled_reward)

        return float(scaled_reward)

    @staticmethod
    def compute_distance_error(field: NDArray, target: NDArray) -> float:
        """Compute spatial L2 (MSE) error between field and target vectors."""
        if field.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: field {field.shape} vs target {target.shape}"
            )

        return float(np.mean((field - target) ** 2))
