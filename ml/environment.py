"""Environment for the TauANN coupled with DNS forced solvers."""

import dataclasses
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ml.reference_scheduler import ReferenceTrajectory
from ml.tau_ann import TauANNConfig, TauANNHyperparameters, TD3Hyperparameters
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SimulationMode
from solvers.solver_coupled import SolverCoupled

REWARD_CLIP = 50

CRASH_PENALTY = -10000


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

        self.target_actions_proof: NDArray | None = None
        self.target_actions_mean: NDArray | None = None
        self.penalty_action_deviation_proof: list[float] | None = None
        self.target_actions_proof_history: list[NDArray] | None = None

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

        self.distance_history.clear()
        self.distance_improvement_history_raw.clear()
        self.distance_error_history_raw.clear()
        self.spectral_penalty_history_raw.clear()
        self.action_penalty_history_raw.clear()

        self.distance_improvement_history_weighted.clear()
        self.distance_error_history_weighted.clear()
        self.spectral_penalty_history_weighted.clear()
        self.action_penalty_history_weighted.clear()

        self.total_reward_history_unscaled.clear()
        self.total_reward_history_scaled.clear()

        self.reference_trajectory.reset()
        self.running_mean_solution = self.solver.solution.copy()
        return self.solver.create_input_stencil(mean_profile=self.running_mean_solution)

    def step(
        self, action: NDArray, proof_of_concept_mode: bool = False
    ) -> tuple[NDArray, float, bool, dict]:
        """Set αₙ, advance Nₛₖᵢₚ LES steps, return (sₙ₊₁, rₙ, done, info)."""
        if self.solver is None:
            raise RuntimeError("Call reset() before step().")

        self.solver.correction_coefficients = action
        last_valid_state = self.solver.create_input_stencil(
            mean_profile=self.running_mean_solution
        )
        info = {"crashed": False}

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for _ in range(self.ann_config.n_skip_steps):
                    if self.ann_config.proof_mode:
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

            if not proof_of_concept_mode:
                reward_val = float(self.compute_reward(action))
            else:
                reward_val = self.compute_reward_proof_b(action)

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
        action_dimension = self.ann_config.action_dimension
        high = 1.3
        low = 1.1
        upper_bound = 1.2
        lower_bound = 1.1

        values = np.random.uniform(low, high, size=action_dimension)
        invalid_mask = (values >= lower_bound) & (values <= upper_bound)

        while np.any(invalid_mask):
            values[invalid_mask] = np.random.uniform(
                low, high, size=np.count_nonzero(invalid_mask)
            )
            invalid_mask = (values >= lower_bound) & (values <= upper_bound)

        return values

    def apply_action_target_perturbation(self, time: float) -> None:
        """Apply a change to the action targets for proof of concept."""
        func = np.sin
        phase = 0.5
        omega = 0.8
        if self.target_actions_mean is None:
            self.target_actions_mean = self.initialize_randomized_target_action()
            self.target_actions_proof_history = []
            self.penalty_action_deviation_proof = []

        perturb_a = func(2 * omega * time) * 0.1
        perturb_b = func(2 * (time + phase)) * 0.1

        perturbations = np.array([perturb_a, perturb_b])
        self.target_actions_proof = self.target_actions_mean + perturbations
        self.target_actions_proof_history.append(self.target_actions_proof)

    def compute_reward_proof_b(self, action: NDArray) -> float:
        """
        Compute a scalar RL reward signal.

        This reward signal serves the function of testing whether the agent actually learn anything from its actions.
        The reward is purely an offset from a randomly initialized action set a_ref_proof (a_1, a_2) where a = {0.8, 1.2}
         but not "near" 1 (or rather the initialized value).
        """
        if self.target_actions_proof is None:
            self.target_actions_proof = self.initialize_randomized_target_action()
            self.penalty_action_deviation_proof = []

        assert (
            self.target_actions_proof is not None
            and self.penalty_action_deviation_proof is not None
        )
        penalty = self.compute_distance_error(
            field=action, target=self.target_actions_proof
        )
        self.penalty_action_deviation_proof.append(penalty)
        return -penalty

    def compute_reward(self, action: NDArray) -> float:
        """Compute the scalar RL reward for the current step."""
        if self.solver is None or self.running_mean_solution is None:
            raise RuntimeError("Solver or running mean solution is not initialized.")

        # --- 1. EWMA Running Profile Update & Distance Computation ---
        self.running_mean_solution = self.solver.update_running_mean()

        target_profile = self.reference_trajectory.target_profile

        # L2 spatial distance error (non-squared Euclidean norm / RMSE for linear gradient response)
        distance = self.compute_distance_error(
            self.running_mean_solution, target_profile
        )

        # Handle initial step initialization cleanly
        if not self.distance_history:
            prev_distance = distance
        else:
            prev_distance = self.distance_history[-1]

        # Raw physical delta improvement (unweighted)
        raw_improvement = prev_distance - distance
        raw_improvement = np.clip(prev_distance - distance, -1.0, 1.0)

        raw_distance_error = distance

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
        norm_factor = np.sum(np.abs(proj_spectrum_k)) + 1e-12

        # Linear high-wavenumber spectral error
        spectral_error = np.abs(spectrum_k - proj_spectrum_k) / norm_factor
        unweighted_spectral_error = (normalized_k**self.hp.gamma) * spectral_error
        raw_spectral_error = float(np.sum(unweighted_spectral_error))

        # --- 3. Action Regularization Metric ---
        a_ref = np.ones_like(action)
        raw_action_deviation = self.compute_distance_error(action, a_ref)

        # --- 4. Log Unweighted Physical Metrics ---
        self.distance_history.append(distance)
        self.distance_improvement_history_raw.append(raw_improvement)
        self.distance_error_history_raw.append(raw_distance_error)
        self.spectral_penalty_history_raw.append(raw_spectral_error)
        self.action_penalty_history_raw.append(raw_action_deviation)

        # --- 5. Apply Reward Weights for TD3 Agent ---
        reward_improvement = self.hp.weight_improvement * raw_improvement
        penalty_absolute_distance = self.hp.weight_absolute_error * distance
        penalty_spectral = self.hp.weight_spectral * raw_spectral_error
        penalty_action = self.hp.weight_action * raw_action_deviation

        reward_total = (
            reward_improvement
            - penalty_absolute_distance
            - penalty_spectral
            - penalty_action
        )
        scaled_reward = float(np.clip(reward_total, -REWARD_CLIP, REWARD_CLIP))

        # --- 6. Log Weighted Physical Metrics ---
        self.distance_improvement_history_weighted.append(reward_improvement)
        self.distance_error_history_weighted.append(penalty_absolute_distance)
        self.spectral_penalty_history_weighted.append(penalty_spectral)
        self.action_penalty_history_weighted.append(penalty_action)

        self.total_reward_history_unscaled.append(reward_total)
        self.total_reward_history_scaled.append(scaled_reward)

        self.last_reward_components = {
            "improvement_term": float(reward_improvement),
            "absolute_distance_term": float(penalty_absolute_distance),
            "action_penalty_term": float(penalty_action),
            "spectral_penalty_term": float(penalty_spectral),
        }

        if distance < 1e-3 or raw_improvement > 1.0:
            print(f"[DEBUG STEP {len(self.distance_history)}] SPIKE DETECTED!")
            print(f"  -> distance: {distance}")
            print(f"  -> prev_distance: {prev_distance}")
            print(f"  -> raw_improvement: {raw_improvement}")
            print(
                f"  -> running_mean min/max: {self.running_mean_solution.min()}, {self.running_mean_solution.max()}"
            )
            print(
                f"  -> target_profile min/max: {target_profile.min()}, {target_profile.max()}"
            )

        return scaled_reward

    @staticmethod
    def compute_distance_error(field: NDArray, target: NDArray) -> float:
        """Compute spatial L1 or RMS error (non-squared) between field and target vectors."""
        if field.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: field {field.shape} vs target {target.shape}"
            )
        # Root Mean Square Error (RMSE) provides a linear gradient standard
        return float(np.sqrt(np.mean((field - target) ** 2)))
