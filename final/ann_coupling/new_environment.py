"""Environment for the TauANN coupled with DNS forced solvers."""

import dataclasses
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from final.ann_coupling.reference_scheduler import ReferenceTrajectory
from ml_old_old_old.tau_ann import TauANNConfig
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SimulationMode
from solvers.solver_coupled import SolverCoupled


class EnvironmentForcingDNS:
    """MDP wrapper around SolverCoupled with DNS forcing for TauANN training."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        ann_config: TauANNConfig,
        master_path: Path,
        reference_schedule: ReferenceTrajectory,
    ) -> None:
        self.problem = problem
        self.disc_config = disc_config
        self.ann_config = ann_config
        self.master_path = master_path
        self.reference_schedule = reference_schedule

        self.solver: SolverCoupled | None = None
        self.running_mean_solution: NDArray | None = None

        # TODO: Move _max_les_steps into DiscretizationConfig
        self._max_les_steps: int = self.disc_config.n_timesteps
        self._total_les_steps: int = 0

        self.total_reward_history: list[float] = []
        self.distance_improvement_history: list[float] = []
        self.distance_error_history: list[float] = []
        self.spectral_penalty_history: list[float] = []
        self.action_penalty_history: list[float] = []

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
        self.total_reward_history.clear()
        self.distance_error_history.clear()
        self.reference_schedule.reset()
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

            self.reference_schedule.set_step_index(self._total_les_steps)

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
            reward_val = -100.0
            done_flag = True

            fallback_state = np.nan_to_num(
                last_valid_state, nan=0.0, posinf=1.0, neginf=-1.0
            )
            self.total_reward_history.append(reward_val)
            return fallback_state, reward_val, done_flag

    def compute_reward(self, action: NDArray) -> float:
        """Compute the scalar RL reward for the current step."""
        if self.solver is None or self.running_mean_solution is None:
            raise RuntimeError("Solver or running mean solution is not initialized.")

        # --- Hyperparameters ---
        smoothing_factor = 0.002
        burn_in_steps = 300
        weight_improvement = 1e3
        weight_absolute_error = 1.0
        weight_spectral = 1.0
        weight_action = 0.1
        gamma = 5.0 / 3.0

        # --- 1. EWMA Running Profile Update & Distance Computation ---
        self.running_mean_solution = (
            1.0 - smoothing_factor
        ) * self.running_mean_solution + smoothing_factor * self.solver.solution

        target_profile = self.reference_schedule.target_profile

        # Retrieve previous raw distance error
        raw_prev_distance_error = (
            self.distance_error_history[-1] if self.distance_error_history else 0.0
        )

        if self._total_les_steps > burn_in_steps:
            raw_distance_error = self.compute_distance_error(
                self.running_mean_solution, target_profile
            )
        else:
            raw_distance_error = 0.0

        # Raw physical delta improvement (unweighted)
        raw_improvement = raw_prev_distance_error - raw_distance_error

        # --- 2. Instantaneous Energy Spectrum Metrics ---
        _, spectrum_les = self.solver.compute_energy_spectrum_(self.solver.solution)
        spectrum_k = spectrum_les.astype(np.float64)

        projected_solution_field = self.reference_schedule.projected_solution
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
        norm_factor = np.sum(np.abs(proj_spectrum_k)) + 1e-12

        spectral_error = ((spectrum_k - proj_spectrum_k) ** 2) / norm_factor
        unweighted_spectral_error = (wavenumber_indices**gamma) * spectral_error
        raw_spectral_error = float(np.sum(unweighted_spectral_error))

        # --- 3. Action Regularization Metric ---
        a_ref = np.ones_like(action)
        raw_action_deviation = self.compute_distance_error(action, a_ref)

        # --- 4. Log Unweighted Physical Metrics (for Diagnostics / Plots) ---
        self.distance_error_history.append(raw_distance_error)
        self.distance_improvement_history.append(raw_improvement)
        self.spectral_penalty_history.append(raw_spectral_error)
        self.action_penalty_history.append(raw_action_deviation)

        # --- 5. Apply Reward Weights for TD3 Agent ---
        reward_improvement = weight_improvement * raw_improvement
        penalty_absolute_distance = weight_absolute_error * (raw_distance_error**2)
        penalty_spectral = weight_spectral * raw_spectral_error
        penalty_action = weight_action * raw_action_deviation

        # Assemble composite rewards
        total_penalty = penalty_absolute_distance + penalty_spectral + penalty_action
        reward_total = reward_improvement - total_penalty

        # Scaled reward for policy backpropagation
        scaled_reward = reward_improvement - np.log1p(total_penalty)
        self.total_reward_history.append(reward_total)

        return float(scaled_reward)

    @staticmethod
    def compute_distance_error(field: NDArray, target: NDArray) -> float:
        """Compute spatial L2 (MSE) error between field and target vectors."""
        if field.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: field {field.shape} vs target {target.shape}"
            )

        return float(np.mean((field - target) ** 2))
