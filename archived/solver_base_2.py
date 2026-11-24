import csv
import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence, Generator

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

from ml.reference_trajectory import ReferenceTrajectory
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from utils.diagnostics import (
    compute_energy,
    compute_dissipation,
    compute_energy_spectrum,
)

TOLERANCE_RESIDUAL: float = 1e-2
FIXED_HOMOGENEOUS_BCS: tuple[float, float] = (0, 0)
NR_RELAXATION = 1


class TauModel(str, Enum):
    TWO_PARAMS_BASE = "two_base"
    TWO_PARAMS_TEMPORAL = "two_temporal"

    THREE_PARAMS_BASE = "three_base"
    THREE_PARAMS_TEMPORAL = "three_temporal"

    @property
    def output_dimensions(self) -> int:
        match self:
            case TauModel.TWO_PARAMS_BASE:
                return 2
            case TauModel.TWO_PARAMS_TEMPORAL:
                return 3
            case TauModel.THREE_PARAMS_BASE:
                return 3
            case TauModel.THREE_PARAMS_TEMPORAL:
                return 4


class TauMode(str, Enum):
    FROZEN = "frozen"
    HOT = "hot"


class SimulationMode(str, Enum):
    DNS = "dns"
    NO_MODEL = "no_model"
    VMS = "vms"


class SolverBase2:
    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        reference_trajectory: ReferenceTrajectory,
        simulation_mode: SimulationMode,
        master_path: Path,
        tau_model: TauModel | None = None,
        snapshot_factor: int = 1,
        external_correction_coefficients: NDArray | None = None,
        tau_mode: TauMode | str = TauMode.FROZEN,
        include_nonlinear_vms: bool = True,
    ) -> None:

        try:
            self.simulation_mode = SimulationMode(simulation_mode)
        except ValueError:
            raise ValueError(
                f"Invalid simulation_mode {simulation_mode!r}. "
                f"Must be one of {[e.value for e in SimulationMode]}"
            ) from None

        if tau_model is not None:
            try:
                self.tau_model = TauModel(tau_model)
            except ValueError:
                raise ValueError(
                    f"Invalid tau_model {tau_model!r}. "
                    f"Must be one of {[e.value for e in TauModel]}"
                ) from None
        else:
            self.tau_model = None

        try:
            self.tau_mode = TauMode(tau_mode)
        except ValueError:
            raise ValueError(
                f"Invalid tau_mode {tau_mode!r}. Must be one of {[m.value for m in TauMode]}"
            ) from None

        self.problem = problem
        self.disc_config = disc_config
        self.reference_trajectory = reference_trajectory
        self.master_path = master_path
        self.snapshot_factor = snapshot_factor
        self.forced_corrections = external_correction_coefficients
        self.include_nonlinear_vms = bool(include_nonlinear_vms)

        self.viscosity = problem.viscosity

        self.convergence_tolerance = TOLERANCE_RESIDUAL
        self.nr_relaxation: float = NR_RELAXATION

        # continuation scale for the quadratic VMS term, set per timestep
        self._nonlinear_vms_scale: float = 1.0

        # temporal parameters
        self.time: float = problem.t_start
        self.time_elapsed: float = 0.0
        self.prev_elapsed: float = 0.0
        self.current_solver_step: int = 0
        self.simulation_done: bool = False
        self.domain_timespan: float = problem.domain_timespan
        self.dt: float = (
            disc_config.dt_dns if simulation_mode == "dns" else disc_config.dt_les
        )
        self._n_time_steps, self._step_dts = self._compute_time_stepping(
            self.dt, self.domain_timespan
        )
        self.time_steps: NDArray = problem.t_start + np.concatenate(
            ([0.0], np.cumsum(self._step_dts))
        )

        # spatial parameters
        self.domain_length = problem.domain_length
        self.n_nodes: int = (
            disc_config.n_nodes_dns
            if simulation_mode == "dns"
            else disc_config.n_nodes_les
        )
        self.n_elements: int = self.n_nodes - 1
        self.nodes: NDArray = np.arange(0, self.n_nodes)
        self.boundary_nodes: set[int] = {int(self.nodes[0]), int(self.nodes[-1])}
        self.mesh: NDArray = np.linspace(0, self.domain_length, self.n_nodes)
        self.elements: NDArray = self.initialize_elements()
        self.element_size: float = self.domain_length / (self.n_nodes - 1)

        self.boundary_values: tuple[float, float] = FIXED_HOMOGENEOUS_BCS

        # initial condition
        assert reference_trajectory.initial_condition is not None
        self.initial_condition: NDArray = reference_trajectory.initial_condition.copy()

        # solution fields
        self.solution: NDArray = self.initial_condition.copy()
        self.solution_previous: NDArray = np.zeros_like(self.solution)
        self.mean_solution: NDArray | None = None
        self.reynolds_stress: NDArray | None = None

        # forcing field
        assert reference_trajectory.initial_forcing is not None
        self.forcing_current: NDArray = reference_trajectory.initial_forcing.copy()

        # output
        self.snapshot_factor = snapshot_factor
        self._snapshot_step_indices: frozenset[int] = frozenset(
            range(0, self._n_time_steps, snapshot_factor)
        )
        self.requested_snapshots: NDArray = self.time_steps[
            sorted(self._snapshot_step_indices)
        ]
        self.is_written_at_times: list[bool] | None = (
            [False] * len(self.requested_snapshots)
            if self.requested_snapshots is not None
            else None
        )
        self.master_path: Path = master_path
        self.master_path.mkdir(parents=True, exist_ok=True)

        # snapshots
        self.snapshots_solution: list[NDArray] = []
        self.snapshots_forcing: list[NDArray] = []

        # Benchmarking
        self.run_id: str = datetime.now().strftime("%m%d_%H%M%S")
        self.timings_performance: dict = {}
        self.residual_history: list = []
        self.update_history: list = []
        self.energy_history: list = []
        self.dissipation_history: list = []
        self.logger = self._setup_logger(
            suppress_file_logging=disc_config.suppress_file_logging
        )

        self.tau_exponential_decay = 0
        self.current_nr_iteration = 0

        # VMS parameter tracking
        self.tau_advective_history = []
        self.tau_diffusive_history = []
        self.tau_temporal_history = []
        self.strong_residual_history = []

        self.frozen_tau_history = []
        self.hot_tau_history = []

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def set_tau_exponential_decay(self):
        self.tau_exponential_decay = 1  # - np.exp(-10 * self.current_solver_step)

    def initialize_elements(self) -> NDArray:
        """Build element connectivity array [[0,1], [1,2], …]."""
        return np.column_stack((self.nodes[:-1], self.nodes[1:]))

    # ------------------------------------------------------------------ #
    #  Core
    # ------------------------------------------------------------------ #

    def run_simulation(self) -> None:
        """Run the full simulation and writes output."""
        self.resolve_current_forcing()  # add IC to snapshots
        self._extract_snapshot()

        with self.timer("total_simulation"):
            with tqdm(
                total=self._n_time_steps,
                desc=f"Eating Burgers | {self.throbber(0)}",
                file=sys.stdout,
            ) as pbar:
                for time_step in range(self._n_time_steps):
                    step_start = perf_counter()

                    try:
                        self.advance_time_step()
                    except Exception:
                        self.write_strong_residual_history()
                        raise

                    if (time_step + 1) in self._snapshot_step_indices:
                        self._extract_snapshot()

                    pbar.set_description(f"Eating Burgers | {self.throbber(time_step)}")
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "t": f"{self.time:.3f}",
                            "dt": f"{self.dt:.3f}",
                            "step_time": f"{perf_counter() - step_start:.3f}s",
                        }
                    )

            self.write_config_to_json()
            self.write_solution_to_csv()

            # final flush
            self.mean_solution = self.calculate_mean_profile()
            self.reynolds_stress = self.compute_reynolds_stress()
            self.write_mean_solution_to_csv()
            self.write_reynolds_profile()
            self.write_strong_residual_history()
            self.write_tau_history()

    def advance_time_step(self) -> None:
        """Advance one timestep, committing only converged solutions."""
        self.reference_trajectory.set_step_index(self.current_solver_step + 1)
        self.set_tau_exponential_decay()
        self.resolve_current_forcing()

        # Previous solution and time history remain untouched on failure.
        new_solution = self.nr_iteration(
            self.solution,
            self.solution_previous,
        )

        old_solution = self.solution.copy()
        self.solution_previous = old_solution
        self.solution = new_solution

        self.energy_history.append(self.compute_energy_(self.solution))
        self.dissipation_history.append(self.compute_dissipation_(self.solution))

        self.time += self.dt
        self.time_elapsed += self.dt
        self.current_solver_step += 1

        if int(self.time_elapsed) > int(self.prev_elapsed):
            self.mean_solution = self.calculate_mean_profile()
            self.write_mean_solution_to_csv()

        self.prev_elapsed = self.time_elapsed

        if self.time_elapsed >= self.domain_timespan:
            self.simulation_done = True

    def nr_iteration(
        self,
        solution: NDArray,
        solution_prev: NDArray,
        max_iterations: int = 100,
    ) -> NDArray:
        """Newton solve with selectable frozen or lagged-hot element tau."""
        solution_nm1 = solution_prev.copy()
        solution_n = solution.copy()
        solution_k = solution.copy()

        residual_history_loop = []
        update_history_loop = []

        if self._use_vms:
            frozen_taus = self._evaluate_element_tau(solution_n, record_components=True)
        else:
            frozen_taus = [None] * self.n_elements
        self.frozen_tau_history.append(np.asarray(frozen_taus, dtype=float))
        self.hot_tau_history.append([])

        # ---- CHEAP FIX: continuation on the nonlinear VMS term ----
        self._nonlinear_vms_scale = 1.0
        ramp_step = 0.25
        # -----------------------------------------------------------

        converged = False
        for iteration in range(max_iterations + 1):
            self.current_nr_iteration = iteration

            if self._use_vms and self.tau_mode == TauMode.HOT:
                current_taus = self._evaluate_element_tau(
                    solution_k, record_components=False
                )
            else:
                current_taus = frozen_taus

            if self._use_vms:
                self.hot_tau_history[-1].append(
                    np.asarray(self._evaluate_element_tau(solution_k), dtype=float)
                )

            global_residual, global_jacobian = self.assemble_global_system(
                solution_k, solution_n, solution_nm1, current_taus
            )
            residual_norm = float(np.linalg.norm(global_residual))
            residual_history_loop.append(residual_norm)

            if self.is_residual_converged(residual_norm, self.convergence_tolerance):
                converged = True
                break

            if iteration == max_iterations:
                break

            try:
                delta_u = np.linalg.solve(global_jacobian, -global_residual)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    f"Singular Newton Jacobian at timestep "
                    f"{self.current_solver_step + 1}, iteration {iteration + 1}."
                ) from exc

            update_history_loop.append(float(np.linalg.norm(delta_u)))

            alpha = 1.0
            min_alpha = 1e-6
            accepted = False
            trial_history = []

            while alpha >= min_alpha:
                candidate = solution_k + alpha * self.nr_relaxation * delta_u
                candidate_taus = (
                    self._evaluate_element_tau(candidate, record_components=False)
                    if self._use_vms and self.tau_mode == TauMode.HOT
                    else current_taus
                )
                candidate_residual, _ = self.assemble_global_system(
                    candidate, solution_n, solution_nm1, candidate_taus
                )
                candidate_norm = float(np.linalg.norm(candidate_residual))
                trial_history.append((alpha, candidate_norm))

                if np.isfinite(candidate_norm) and candidate_norm < residual_norm:
                    solution_k = candidate
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                self.write_config_to_json()
                self.write_solution_to_csv()
                self.write_tau_history()
                self.mean_solution = self.calculate_mean_profile()
                self.reynolds_stress = self.compute_reynolds_stress()
                self.write_mean_solution_to_csv()
                self.write_reynolds_profile()
                self.post_processing()

                print(f"\nNewton failure: timestep {self.current_solver_step + 1}")
                print(f"Iteration: {iteration + 1}")
                print(f"Current residual: {residual_norm:.8e}")
                print(f"Newton update norm: {np.linalg.norm(delta_u):.8e}")
                print(f"Nonlinear VMS scale: {self._nonlinear_vms_scale:.3f}")
                for alpha_trial, norm_trial in trial_history:
                    print(f"alpha={alpha_trial:.8e}, residual={norm_trial:.8e}")
                raise RuntimeError("Newton line search failed.")

            # ---- CHEAP FIX: ramp the nonlinear term after each accepted step ----
            if self.include_nonlinear_vms:
                self._nonlinear_vms_scale = min(
                    1.0, self._nonlinear_vms_scale + ramp_step
                )
            # -------------------------------------------------------------------

        self.residual_history.append(residual_history_loop)
        self.update_history.append(update_history_loop)

        if not converged:
            raise RuntimeError(
                f"Newton failed to converge at timestep {self.current_solver_step + 1} "
                f"after {max_iterations} iterations. Final residual: "
                f"{residual_history_loop[-1]:.6e}"
            )

        return solution_k

    def _evaluate_element_tau(
        self, solution: NDArray, record_components: bool = False
    ) -> list[float | None]:
        """Evaluate one tau value per element for the supplied global solution."""
        taus: list[float | None] = []
        gradient_basis = self.gradient_basis_functions()
        for element in self.elements:
            if not self._use_vms:
                taus.append(None)
                continue
            u_e = solution[element]
            du_e = gradient_basis @ u_e
            tau_e = self.tau_exponential_decay * self.compute_tau(
                u_e,
                du_e,
                self.forced_corrections,
                record_components=record_components,
            )
            taus.append(float(tau_e))
        return taus

    def write_tau_history(self) -> None:
        """Write frozen/hot tau histories and a spatial-temporal comparison plot."""
        if not self._use_vms or not self.frozen_tau_history:
            return

        output_dir = self.master_path / "tau_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)

        frozen = np.asarray(self.frozen_tau_history, dtype=float)
        hot_final = np.full_like(frozen, np.nan, dtype=float)
        csv_path = output_dir / "frozen_vs_hot_tau.csv"

        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timestep",
                    "time",
                    "newton_iteration",
                    "element",
                    "tau_frozen",
                    "tau_hot",
                    "relative_difference",
                ]
            )
            for step_idx, iteration_values in enumerate(self.hot_tau_history):
                step_time = float(
                    self.time_steps[min(step_idx + 1, len(self.time_steps) - 1)]
                )
                for iteration_idx, hot_values in enumerate(iteration_values):
                    hot_arr = np.asarray(hot_values, dtype=float)
                    if iteration_idx == len(iteration_values) - 1:
                        hot_final[step_idx, :] = hot_arr
                    for e_idx, (tau_f, tau_h) in enumerate(
                        zip(frozen[step_idx], hot_arr)
                    ):
                        denom = max(abs(float(tau_f)), np.finfo(float).eps)
                        rel = abs(float(tau_h) - float(tau_f)) / denom
                        writer.writerow(
                            [
                                step_idx + 1,
                                step_time,
                                iteration_idx,
                                e_idx,
                                float(tau_f),
                                float(tau_h),
                                rel,
                            ]
                        )

        n_steps = frozen.shape[0]
        times = np.asarray(self.time_steps[1 : n_steps + 1], dtype=float)
        if len(times) != n_steps:
            times = np.arange(1, n_steps + 1, dtype=float)
        else:
            times = times - float(self.time_steps[0])
        x_centres = 0.5 * (self.mesh[:-1] + self.mesh[1:])
        relative = np.abs(hot_final - frozen) / np.maximum(
            np.abs(frozen), np.finfo(float).eps
        )

        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        panels = [
            (axes[0, 0], frozen, r"Frozen $\tau(u_n)$", "viridis"),
            (
                axes[0, 1],
                hot_final,
                r"Hot $\tau(u_k)$ (final Newton iterate)",
                "viridis",
            ),
            (
                axes[1, 0],
                relative,
                r"Relative difference $|\tau_h-\tau_f|/|\tau_f|$",
                "magma",
            ),
        ]
        for ax, data, title, cmap in panels:
            image = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=[x_centres[0], x_centres[-1], times[0], times[-1]],
                cmap=cmap,
            )
            ax.set_title(title)
            ax.set_xlabel("Element centre x")
            ax.set_ylabel("Elapsed time")
            fig.colorbar(image, ax=ax, label="Value")

        ax = axes[1, 1]
        if np.any(np.isfinite(hot_final)):
            ax.plot(x_centres, frozen[-1], label=r"Frozen $\tau(u_n)$")
            ax.plot(x_centres, hot_final[-1], "--", label=r"Hot $\tau(u_k)$")
        ax.set_title("Final timestep spatial comparison")
        ax.set_xlabel("Element centre x")
        ax.set_ylabel("Stabilisation parameter")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(output_dir / "frozen_vs_hot_tau.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

    def assemble_global_system(
        self,
        solution_k: NDArray,
        solution_n: NDArray,
        solution_nm1: NDArray,
        taus: list,
    ) -> tuple[NDArray, NDArray]:

        elemental_residuals = []
        elemental_jacobians = []

        for e_idx, element in enumerate(self.elements):
            residual_e, jacobian_e = self.calculate_elemental_residual_jacobian(
                element=element,
                u_k_e=solution_k[element],
                u_n_e=solution_n[element],
                u_nm1_e=solution_nm1[element],
                f_e=self.forcing_current[element],
                tau_e=taus[e_idx],
            )

            elemental_residuals.append(residual_e)
            elemental_jacobians.append(jacobian_e)

        global_residual, global_jacobian = self.global_assembly(
            elemental_residuals,
            elemental_jacobians,
        )

        global_residual, global_jacobian = self.apply_homogeneous_bcs(
            global_residual, global_jacobian, solution_k
        )

        return global_residual, global_jacobian

    def calculate_elemental_residual_jacobian(
        self,
        element: tuple[int, int],
        u_k_e: NDArray,
        u_n_e: NDArray,
        u_nm1_e: NDArray,
        f_e: NDArray,
        tau_e: float | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Compute the elemental residual and Jacobian, including VMS terms."""
        residual_element = np.zeros(2)
        jacobian_element = np.zeros((2, 2))

        weight = self.element_size / 2
        gauss_points, gauss_weights = self.gauss_legendre(3)
        gradient_basis = self.gradient_basis_functions()

        for gauss_point, gauss_weight in zip(gauss_points, gauss_weights):
            basis = self.basis_functions(gauss_point)
            interpolated_fields = self._interpolate_fields(
                basis, gradient_basis, u_k_e, u_n_e, u_nm1_e, f_e
            )
            scale = gauss_weight * weight
            strong_residual = self.strong_elemental_residual(interpolated_fields)
            self.strong_residual_history.append(
                {
                    "step": self.current_solver_step,
                    "iteration": self.current_nr_iteration,
                    "element": element[0],
                    "gauss_point": gauss_point,
                    "x": element[0] * self.element_size
                    + self.element_size / 2 * (gauss_point + 1),
                    "residual": strong_residual,
                }
            )

            for e_i in range(len(element)):
                R_galerkin = scale * self.galerkin_elemental_residual(
                    e_i, basis, gradient_basis, interpolated_fields
                )
                residual_element[e_i] += R_galerkin

                if self._use_vms and tau_e is not None:
                    R_vms_advection, R_vms_nonlinear = scale * np.asarray(
                        self.vms_elemental_residual(
                            i=e_i,
                            gradient_basis=gradient_basis,
                            f=interpolated_fields,
                            strong_res=strong_residual,
                            tau_e=tau_e,
                            include_nonlinear=self.include_nonlinear_vms,
                        )
                    )
                    # ---- BUG FIX: nonlinear term was being dropped ----
                    residual_element[e_i] += R_vms_advection + R_vms_nonlinear

                for e_j in range(len(element)):
                    jacobian_element[e_i, e_j] += (
                        scale
                        * self.galerkin_elemental_jacobian(
                            e_i, e_j, basis, gradient_basis, interpolated_fields
                        )
                    )

                    if self._use_vms and tau_e is not None:
                        # ---- FIX 1: full Jacobian for cross + quadratic terms,
                        #             including HOT-τ derivatives, handled inside.
                        jacobian_element[e_i, e_j] += (
                            scale
                            * self.vms_elemental_jacobian(
                                i=e_i,
                                j=e_j,
                                basis=basis,
                                gradient_basis=gradient_basis,
                                f=interpolated_fields,
                                strong_res=strong_residual,
                                tau_e=tau_e,
                                u_e=u_k_e,
                            )
                        )

        return residual_element, jacobian_element

    def galerkin_elemental_residual(
        self,
        e_i: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
    ) -> float:
        a0, a1, a2 = self._time_derivative_coefficients()
        time_derivative = (
            basis[e_i] * (a0 * f["u_k"] + a1 * f["u_n"] + a2 * f["u_nm1"]) / self.dt
        )
        diffusion = self.viscosity * f["du_dx_k"] * gradient_basis[e_i]
        advection = basis[e_i] * f["u_k"] * f["du_dx_k"]
        forcing = basis[e_i] * f["forcing"]

        return time_derivative + diffusion + advection - forcing

    def strong_elemental_residual(self, f: dict) -> float:
        """Strong residual with BDF1 startup and BDF2 thereafter."""
        a0, a1, a2 = self._time_derivative_coefficients()
        time_term = (a0 * f["u_k"] + a1 * f["u_n"] + a2 * f["u_nm1"]) / self.dt
        return time_term + f["u_k"] * f["du_dx_k"] - f["forcing"]

    def galerkin_elemental_jacobian(
        self,
        e_i: int,
        e_j: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
    ) -> float:
        a0, _, _ = self._time_derivative_coefficients()
        mass = a0 * basis[e_i] * basis[e_j] / self.dt
        stiffness = self.viscosity * gradient_basis[e_i] * gradient_basis[e_j]
        advection = basis[e_i] * (
            basis[e_j] * f["du_dx_k"] + f["u_k"] * gradient_basis[e_j]
        )
        return mass + stiffness + advection

    def global_assembly(
        self,
        elemental_residuals: Sequence[NDArray],
        elemental_jacobians: Sequence[NDArray],
    ) -> tuple[NDArray, NDArray]:
        """Assemble element contributions into global R and J."""
        global_residual = np.zeros(self.n_nodes)
        global_jacobian = np.zeros((self.n_nodes, self.n_nodes))

        for e, element in enumerate(self.elements):
            i, j = element
            global_residual[i] += elemental_residuals[e][0]
            global_residual[j] += elemental_residuals[e][1]
            global_jacobian[i, i] += elemental_jacobians[e][0, 0]
            global_jacobian[i, j] += elemental_jacobians[e][0, 1]
            global_jacobian[j, i] += elemental_jacobians[e][1, 0]
            global_jacobian[j, j] += elemental_jacobians[e][1, 1]

        return global_residual, global_jacobian

    def apply_homogeneous_bcs(
        self,
        global_residual: NDArray,
        global_jacobian: NDArray,
        solution_k: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Enforce Dirichlet BCs via row-replacement."""
        bc_values = self.boundary_values  # (0, 0)
        for node, bc_value in zip(sorted(self.boundary_nodes), bc_values):
            global_residual[node] = solution_k[node] - bc_value

            global_jacobian[node, :] = 0
            global_jacobian[node, node] = 1
        return global_residual, global_jacobian

    # ------------------------------------------------------------------ #
    #  VMS Models
    # ------------------------------------------------------------------ #

    def vms_elemental_residual(
        self,
        i: int,
        gradient_basis: NDArray,
        f: dict[str, float],
        strong_res: float,
        tau_e: float,
        include_nonlinear: bool = True,
    ) -> tuple[float, float]:
        """Elemental VMS residual split into (cross, quadratic) contributions."""
        u_k = f["u_k"]
        advection = tau_e * u_k * strong_res
        nonlinear = 0.0
        if include_nonlinear:
            scale = getattr(self, "_nonlinear_vms_scale", 1.0)
            nonlinear -= 0.5 * scale * (tau_e * strong_res) ** 2
        return gradient_basis[i] * advection, gradient_basis[i] * nonlinear

    def vms_elemental_jacobian(
        self,
        i: int,
        j: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
        strong_res: float,
        tau_e: float,
        u_e: NDArray | None = None,
    ) -> float:
        """VMS Jacobian: cross term + quadratic term (+ HOT-τ derivatives)."""
        u_k = f["u_k"]
        du_k = f["du_dx_k"]
        a0, _, _ = self._time_derivative_coefficients()

        phi_j = basis[j]
        d_phi_j = gradient_basis[j]

        # ∂R/∂u_j at frozen tau
        dR_duj = (a0 / self.dt) * phi_j + du_k * phi_j + u_k * d_phi_j

        # Cross term: ∂/∂u_j [ τ u_k R ] (frozen τ)
        val = tau_e * (phi_j * strong_res + u_k * dR_duj)

        # Quadratic term: ∂/∂u_j [ -0.5 * scale * (τR)^2 ] (frozen τ)
        if self.include_nonlinear_vms:
            scale = getattr(self, "_nonlinear_vms_scale", 1.0)
            val -= scale * tau_e**2 * strong_res * dR_duj

        # HOT: contributions from ∂τ/∂u_j
        if self.tau_mode == TauMode.HOT and u_e is not None:
            dtau_duj = self._element_tau_derivative_wrt_dof(u_e, j)
            # ∂/∂u_j [ τ u_k R ] via τ
            val += dtau_duj * u_k * strong_res
            # ∂/∂u_j [ -0.5 * scale * (τR)^2 ] via τ
            if self.include_nonlinear_vms:
                scale = getattr(self, "_nonlinear_vms_scale", 1.0)
                val -= scale * tau_e * strong_res**2 * dtau_duj

        return gradient_basis[i] * val

    def _element_tau_derivative_wrt_dof(self, u_e: NDArray, j: int) -> float:
        """Derivative of the actual (decay-scaled) element tau wrt nodal DOF j."""
        if self.tau_model is None:
            return 0.0

        h = self.element_size
        decay = float(self.tau_exponential_decay)
        if decay == 0.0:
            return 0.0

        if self.tau_model in (TauModel.TWO_PARAMS_BASE, TauModel.TWO_PARAMS_TEMPORAL):
            c = (
                np.asarray(self.forced_corrections, dtype=float)
                if self.forced_corrections is not None
                else np.ones(2)
            )
            terms = [float((u_e[0] + u_e[1]) / h), float(4.0 * self.viscosity / h**2)]
            weights = [float(c[0]), float(c[1])]
            derivatives = [1.0 / h, 0.0]
            if self.tau_model == TauModel.TWO_PARAMS_TEMPORAL:
                terms.append(2.0 / self.dt)
                weights.append(1.0)
                derivatives.append(0.0)
        else:
            c = (
                np.asarray(self.forced_corrections, dtype=float)
                if self.forced_corrections is not None
                else np.ones(3)
            )
            alpha, beta, gamma = 0.099, 9.39, 2.16
            grad = self.gradient_basis_functions()
            u_x = float(grad @ u_e)
            avg = 0.5 * float(u_e[0] + u_e[1])
            terms = [alpha * avg / h, beta * self.viscosity / h**2, gamma * u_x]
            weights = [float(c[0]), float(c[1]), float(c[2])]
            derivatives = [alpha / (2.0 * h), 0.0, gamma * float(grad[j])]
            if self.tau_model == TauModel.THREE_PARAMS_TEMPORAL:
                terms.append(2.0 / self.dt)
                weights.append(1.0)
                derivatives.append(0.0)

        terms_arr = np.asarray(terms, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)
        derivatives_arr = np.asarray(derivatives, dtype=float)
        norm_squared = float(np.sum((weights_arr * terms_arr) ** 2))
        if norm_squared <= np.finfo(float).tiny:
            return 0.0

        tau_raw = norm_squared**-0.5
        d_tau_raw = -(tau_raw**3) * float(
            np.sum(weights_arr**2 * terms_arr * derivatives_arr)
        )
        return decay * d_tau_raw

    def compute_tau(
        self,
        u_e: NDArray,
        u_x_e: NDArray | None = None,
        external_coefficients: NDArray | None = None,
        record_components: bool = False,
    ) -> float:
        """Dispatch to the configured tau model."""
        has_temporal_parameter = self.tau_model in (
            TauModel.TWO_PARAMS_TEMPORAL,
            TauModel.THREE_PARAMS_TEMPORAL,
        )
        if self.tau_model in (TauModel.TWO_PARAMS_BASE, TauModel.TWO_PARAMS_TEMPORAL):
            return self.tau_model_two_params(
                u_e, has_temporal_parameter, external_coefficients, record_components
            )
        if self.tau_model in (
            TauModel.THREE_PARAMS_BASE,
            TauModel.THREE_PARAMS_TEMPORAL,
        ):
            return self.tau_model_three_params(
                u_e, u_x_e, has_temporal_parameter, external_coefficients
            )
        raise ValueError(f"Unknown tau_model {self.tau_model!r}")

    def tau_model_two_params(
        self,
        u_e: NDArray,
        has_temporal_parameter: bool,
        c: NDArray | None = None,
        record_components: bool = False,
    ) -> float:
        """2-parameter tau model with optional temporal stabilization."""
        if u_e is None or len(u_e) < 2:
            raise ValueError(
                "Input 'u_e' must contain at least 2 elements for averaging."
            )
        if c is not None:
            c_arr = np.asarray(c)
            if c_arr.shape != (2,):
                raise ValueError(
                    "Shape of correction coefficients is not congruent with chosen tau model (expected shape: (2,))."
                )
        else:
            c_arr = np.ones(2)

        elemental_average_u_e = 0.5 * (u_e[0] + u_e[1])
        base_terms = np.array(
            [
                2.0 * elemental_average_u_e / self.element_size,
                3 * 4.0 * self.viscosity / (self.element_size**2),
            ]
        )

        if record_components:
            self.tau_advective_history.append(base_terms[0] ** 2)
            self.tau_diffusive_history.append(base_terms[1] ** 2)

        if has_temporal_parameter:
            c_arr = np.append(c_arr, 1)
            base_terms = np.append(base_terms, 2 / self.dt)
            if record_components:
                self.tau_temporal_history.append(base_terms[2] ** 2)

        return float(np.sum((c_arr * base_terms) ** 2) ** -0.5)

    def tau_model_three_params(
        self,
        u_e: NDArray,
        u_x_e: NDArray,
        has_temporal_parameter: bool,
        c: NDArray | None = None,
        alpha: float = 0.099,
        beta: float = 9.39,
        gamma: float = 2.16,
    ) -> float:
        if u_e is None or len(u_e) < 2:
            raise ValueError(
                "Input 'u_e' must contain at least 2 elements for averaging."
            )

        if c is not None:
            c_arr = np.asarray(c)
            if c_arr.shape != (3,):
                raise ValueError(
                    "Shape of correction coefficients is not congruent with chosen tau model (expected shape: (3,))."
                )
        else:
            c_arr = np.ones(3)

        elemental_average_u_e = 0.5 * (u_e[0] + u_e[1])
        base_terms = np.array(
            [
                alpha * elemental_average_u_e / self.element_size,
                beta * self.viscosity / (self.element_size**2),
                gamma * u_x_e,
            ]
        )

        if has_temporal_parameter:
            c_arr = np.append(c_arr, 1)
            base_terms = np.append(base_terms, 2 / self.dt)

        return float(np.sum((c_arr * base_terms) ** 2) ** -0.5)

    # ------------------------------------------------------------------ #
    #  FEM Primitives
    # ------------------------------------------------------------------ #

    @staticmethod
    def gauss_legendre(number_of_points: int) -> tuple[Any, Any]:
        """Return Gauss–Legendre quadrature points and weights."""
        return np.polynomial.legendre.leggauss(deg=number_of_points)

    @staticmethod
    def basis_functions(ksi: float) -> NDArray:
        """Linear basis functions on the reference element [-1, 1]."""
        return np.array([0.5 * (1 - ksi), 0.5 * (1 + ksi)])

    def gradient_basis_functions(self) -> NDArray:
        """Gradient of linear basis functions mapped to physical space."""
        return np.array([-0.5, 0.5]) * (2 / self.element_size)

    # ------------------------------------------------------------------ #
    #  Other Helpers
    # ------------------------------------------------------------------ #

    def compute_reynolds_stress(self) -> NDArray:
        """Compute Reynolds stress <u'u'>."""
        snapshots = np.asarray(self.snapshots_solution)
        mean = np.mean(snapshots, axis=0)
        return np.mean((snapshots - mean) ** 2, axis=0)

    def _time_derivative_coefficients(self) -> tuple[float, float, float]:
        """BDF1 startup, followed by constant-step BDF2."""
        if self.current_solver_step == 0:
            return 1.0, -1.0, 0.0
        return 1.5, -2.0, 0.5

    @staticmethod
    def _interpolate_fields(
        basis: NDArray,
        gradient_basis: NDArray,
        u_k: NDArray,
        u_n: NDArray,
        u_nm1: NDArray,
        f: NDArray,
    ) -> dict:
        """Interpolate solution and gradient at a quadrature point."""
        return {
            "u_k": basis @ u_k,
            "u_n": basis @ u_n,
            "u_nm1": basis @ u_nm1,
            "forcing": basis @ f,
            "du_dx_k": gradient_basis @ u_k,
            "du_dx_n": gradient_basis @ u_n,
        }

    @staticmethod
    def is_residual_converged(
        residual: float | NDArray, tolerance: float = 1e-8
    ) -> bool:
        """True when the relative residual norm is below tolerance."""
        norm = np.linalg.norm(residual)
        return norm < tolerance * (1 + norm)

    @property
    def _use_vms(self) -> bool:
        """True only for analytic VMS mode (les)."""
        return self.simulation_mode == SimulationMode.VMS

    def resolve_current_forcing(self) -> None:
        self.forcing_current = self.reference_trajectory.forcing

    def calculate_mean_profile(self) -> np.ndarray:
        """Computes the time-averaged spatial mean profile."""
        if self.snapshots_solution is None or len(self.snapshots_solution) == 0:
            raise ValueError(
                "No solution history found. Run the solver before computing statistics."
            )
        return np.mean(np.asarray(self.snapshots_solution), axis=0)

    @staticmethod
    def _compute_time_stepping(
        dt: float, timespan: float, rel_tol: float = 1e-9
    ) -> tuple[int, NDArray]:
        """Build per-step dt array; shorten final step to land on t_start+timespan."""
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        if timespan <= 0:
            raise ValueError(f"domain_timespan must be positive, got {timespan!r}")

        n_full = int(np.floor(timespan / dt + rel_tol))
        remainder = timespan - n_full * dt

        if remainder > rel_tol * dt:
            step_dts = np.full(n_full + 1, dt)
            step_dts[-1] = remainder
        else:
            step_dts = np.full(n_full, dt)

        return len(step_dts), step_dts

    @contextmanager
    def timer(self, name: str) -> Generator[None, Any, None]:
        """Accumulate wall-clock time for a named phase."""
        start = perf_counter()
        yield
        self.timings_performance[name] = (
            self.timings_performance.get(name, 0.0) + perf_counter() - start
        )

    @staticmethod
    def throbber(time_step: int, _every: int = 2) -> str:
        """Cycle through eating-animation states based on time step."""
        states = ["nom..        ", "nom nom..    ", "nom nom nom.."]
        return states[(time_step // _every) % len(states)]

    # ------------------------------------------------------------------ #
    #  Energy and spectral analysis
    # ------------------------------------------------------------------ #

    def compute_energy_(self, solution: NDArray) -> float:
        """Compute total kinetic energy of the given solution snapshot."""
        return compute_energy(solution, self.domain_length)

    def compute_dissipation_(self, solution: NDArray) -> float:
        """Compute total viscous dissipation of the given solution snapshot."""
        return compute_dissipation(solution, self.domain_length, self.viscosity)

    def compute_energy_spectrum_(self, solution: NDArray) -> tuple[NDArray, NDArray]:
        """Compute energy spectrum of the given solution snapshot."""
        return compute_energy_spectrum(solution, self.domain_length)

    # ------------------------------------------------------------------ #
    #  Writing
    # ------------------------------------------------------------------ #

    def _extract_snapshot(self) -> None:
        """Store current solution and forcing as a snapshot."""
        self.snapshots_solution.append(self.solution.copy())
        self.snapshots_forcing.append(self.forcing_current.copy())

    def write_mean_solution_to_csv(self):
        assert self.mean_solution is not None
        mean_profile = self.mean_solution
        time = round(self.time, ndigits=3)
        file_dir = self.master_path / "mean_profiles"
        filepath = file_dir / f"mean_w_{time}.csv"
        filepath_npy = file_dir / f"mean_w_{time}.npy"
        file_dir.mkdir(parents=True, exist_ok=True)
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["node_index", "x_coordinate", "w"])
            for i in range(len(self.nodes)):
                writer.writerow([self.nodes[i], self.mesh[i], mean_profile[i]])

        np.save(filepath_npy, mean_profile)

        print(f"wrote mean profile at {self.master_path}")

    def write_solution_to_csv(self, save_path: Path | None = None) -> None:
        """Write extracted solution snapshots to CSV files."""
        solutions = self.snapshots_solution
        forcings = self.snapshots_forcing
        times = self.requested_snapshots[: len(solutions)]

        master_path = save_path if save_path is not None else self.master_path
        solution_dir = master_path / "u_field"
        solution_dir.mkdir(exist_ok=True, parents=True)

        for solution, time_value, forcing in zip(solutions, times, forcings):
            filepath = solution_dir / f"sol_t{time_value:.6f}.csv"
            with open(filepath, mode="w", newline="") as file_handle:
                writer = csv.writer(file_handle)
                writer.writerow(["node_index", "x_coordinate", "velocity", "forcing"])
                for i in range(len(self.nodes)):
                    writer.writerow(
                        [
                            self.nodes[i],
                            self.mesh[i],
                            solution[i],
                            forcing[i],
                        ]
                    )

        print(f"wrote {len(solutions)} snapshots at {self.master_path}")

    def write_reynolds_profile(self) -> None:
        reynolds_profile = self.reynolds_stress
        time = round(self.time, ndigits=3)

        file_dir = self.master_path / "reynolds_profiles"
        file_dir.mkdir(parents=True, exist_ok=True)

        filepath = file_dir / f"reynolds_stress_profile_{time}.csv"
        filepath_npy = file_dir / f"reynolds_stress_profile_{time}.npy"

        assert reynolds_profile is not None
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["node_index", "x_coordinate", "w"])
            for i in range(len(self.nodes)):
                writer.writerow([self.nodes[i], self.mesh[i], reynolds_profile[i]])

        np.save(filepath_npy, reynolds_profile)

    def write_config_to_json(self) -> None:
        """Serialize run configuration to config.json in the run directory."""
        raw_config: dict = {
            "run_id": self.run_id,
            "simulation_mode": self.simulation_mode,
            "tau_model": self.tau_model,
            "tau_mode": self.tau_mode.value,
            "domain_timespan": self.domain_timespan,
            "domain_length": self.domain_length,
            "dt": self.dt,
            "n_time_steps": self._n_time_steps,
            "viscosity": self.viscosity,
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "element_size": self.element_size,
            "include_nonlinear_vms": self.include_nonlinear_vms,
        }

        config_serializable = {
            k: (
                v.tolist()
                if isinstance(v, np.ndarray)
                else f"<callable: {getattr(v, '__name__', repr(v))}>"
                if callable(v)
                else v
            )
            for k, v in raw_config.items()
        }

        with open(self.master_path / "config.json", "w") as file_handle:
            json.dump(config_serializable, file_handle, indent=2)

    def write_strong_residual_history(self) -> None:
        keys = ("step", "iteration", "element", "gauss_point", "x", "residual")
        np.savez_compressed(
            self.master_path / "strong_residual_history.npz",
            **{k: np.array([r[k] for r in self.strong_residual_history]) for k in keys},
        )

    # ------------------------------------------------------------------  #
    #  Logging
    # ------------------------------------------------------------------ #

    def _setup_logger(self, suppress_file_logging: bool = False) -> logging.Logger:
        """Initialize a file logger for this run."""
        logger_ = logging.getLogger(str(self.run_id))
        logger_.setLevel(logging.INFO)
        if logger_.handlers or suppress_file_logging:
            return logger_

        formatter = logging.Formatter("[%(levelname)s] - %(message)s")
        fh = logging.FileHandler(
            self.master_path / f"{self.run_id}.log", encoding="utf-8"
        )
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger_.addHandler(fh)
        logger_.propagate = False
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        return logger_

    def post_processing(self) -> None:
        """Run post-plotting and post-logging."""
        self.post_plotting()
        self.plot_tau_parameters()
        self.post_logging()

    def post_logging(self) -> None:
        """Write a structured run summary to the log file."""
        self.logger.info("=" * 60)
        self.logger.info("RUN COMPLETE — id: %s", self.run_id)
        self.logger.info("Time Integration: Second Order Implicit Euler")
        self.logger.info("Simulation mode: %s", self.simulation_mode)
        self.logger.info("-" * 40)

        for key, value in self._format_config_for_display().items():
            self.logger.info("  %-30s %s", key, value)

        self.logger.info("-" * 40)

        if self.timings_performance:
            total = self.timings_performance.get("total_simulation") or sum(
                v
                for k, v in self.timings_performance.items()
                if k != "total_simulation"
            )
            for phase, time_elapsed in sorted(self.timings_performance.items()):
                if phase != "total_simulation":
                    pct = (100 * time_elapsed / total) if total > 0 else float("nan")
                    self.logger.info(
                        "  %-25s %.4fs (%5.1f%%)", phase, time_elapsed, pct
                    )
            self.logger.info("  %-25s %.4fs", "TOTAL", total)
        self.logger.info("=" * 60)

    def _format_config_for_display(self) -> dict[str, str]:
        """Format solver state for display in logs and console output."""
        snapshots_str = (
            f"{len(self.requested_snapshots)} snapshots, first 5: "
            f"{[round(float(t), 4) for t in self.requested_snapshots[:5]]}"
            if self.requested_snapshots is not None
            else "None"
        )

        return {
            "domain_timespan": f"{self.domain_timespan:.6g}",
            "domain_length": f"{self.domain_length:.6g}",
            "dt": f"{self.dt:.4e}",
            "n_time_steps": str(self._n_time_steps),
            "viscosity": f"{self.viscosity:.4e}",
            "n_nodes": str(self.n_nodes),
            "n_elements": str(self.n_elements),
            "element_size": f"{self.element_size:.4e}",
            "snapshot_factor": str(self.snapshot_factor),
            "requested_snapshots": snapshots_str,
        }

    def print_configuration(self) -> None:
        """Print run configuration in a clean tabular format."""
        width = 72
        column = 30

        def _row(label: str, value: str) -> None:
            print(f"  {label:<{column}} {value}")

        def _sep(char: str = "─") -> None:
            print(char * width)

        def _section(title: str) -> None:
            print()
            print(f"  {title}")
            _sep()

        _sep("═")
        print(
            f"  Solver Configuration  ·  mode: {self.simulation_mode}  ·  integration: 2nd-order implicit Euler"
        )
        _sep("═")

        _section("Mesh")
        _row("nodes", str(self.n_nodes))
        _row("elements", str(self.n_elements))
        _row("domain length", f"{self.domain_length:.4g}")
        _row("element size h", f"{self.element_size:.4e}")

        _section("Time")
        _row("timespan", f"{self.domain_timespan:.4g}")
        _row("dt", f"{self.dt:.4e}")
        _row("total steps", str(self._n_time_steps))
        if self.requested_snapshots is not None:
            n_ext = len(self.requested_snapshots)
            first_five = "  ".join(f"{t:.4f}" for t in self.requested_snapshots[:5])
            _row("snapshots", str(n_ext))
            _row("  first 5 times", first_five)

        _section("Physics")
        _row("viscosity ν", f"{self.viscosity:.4e}")

        _section("Paths")
        _row("output", str(self.master_path))

        _sep("═")

    # ------------------------------------------------------------------ #
    #  Post-plotting
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_positive_spectrum(
        wavenumbers: NDArray, spectrum: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Filter to non-negative wavenumbers."""
        mask = wavenumbers >= 0
        return wavenumbers[mask], spectrum[mask]

    def post_plotting(self, show_plot: bool = False) -> None:
        """Plot the final solution and diagnostics."""
        sgs_label = {
            "dns": "DNS",
            "les": "LES-VMS",
            "sgsp": "LES-SGSP",
            "avc": "LES-AVC",
        }.get(self.simulation_mode, self.simulation_mode)

        plt.rcParams.update(
            {
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 8.5,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.top": True,
                "ytick.right": True,
                "axes.linewidth": 0.8,
            }
        )

        style_dns = dict(color="black", linestyle="-", linewidth=1.6, zorder=3)
        style_model = dict(
            color="#0072B2",
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            markeredgewidth=0.9,
            zorder=4,
        )
        style_initial = dict(color="k", linestyle=":", linewidth=1.2, zorder=1)
        style_energy = dict(color="#B9314F", linestyle="-", linewidth=1.5, alpha=0.9)
        style_dissipation = dict(
            color="#47329A", linestyle="-", linewidth=1.5, alpha=0.9
        )

        fig = plt.figure(figsize=(10, 6.2), dpi=300, constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0), hspace=0.05)
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])

        for ax in (ax0, ax1, ax2):
            ax.minorticks_on()
            ax.tick_params(
                which="both", direction="in", top=True, right=True, length=4, width=0.8
            )
            ax.tick_params(which="minor", length=2, width=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
            ax.grid(False)

        resolved_style = style_dns if self.simulation_mode == "dns" else style_model
        ax0.plot(self.mesh, self.solution, label="Resolved solution", **resolved_style)
        ax0.plot(
            self.mesh,
            self.initial_condition,
            label="Initial condition",
            **style_initial,
        )
        ax0.axhline(0.0, color="black", linestyle=":", linewidth=0.5, alpha=0.4)
        ax0.set_xlabel(r"Coordinate $x$")
        ax0.set_ylabel(r"Velocity $u$")
        ax0.set_xlim(float(np.min(self.mesh)), float(np.max(self.mesh)))
        ax0.set_title(f"(a) Resolved solution | {sgs_label}", loc="left", pad=6)
        ax0.legend(
            loc="best",
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="0.8",
        )

        n_energy = min(len(self.time_steps), len(self.energy_history))
        n_dissipation = min(len(self.time_steps), len(self.dissipation_history))

        if n_energy:
            ax1.plot(
                self.time_steps[:n_energy],
                self.energy_history[:n_energy],
                label="Total energy",
                **style_energy,
            )

        if n_dissipation:
            ax1.plot(
                self.time_steps[:n_dissipation],
                self.dissipation_history[:n_dissipation],
                label="Dissipation",
                **style_dissipation,
            )

        ax1.set_xlabel(r"Time $t$")
        ax1.set_ylabel("Diagnostic value")
        ax1.set_title("(b) Energy and dissipation", loc="left", pad=6)

        if n_energy or n_dissipation:
            ax1.legend(
                loc="best",
                frameon=True,
                facecolor="white",
                framealpha=0.9,
                edgecolor="0.8",
            )

        wn, sp = self.get_positive_spectrum(
            *self.compute_energy_spectrum_(self.solution)
        )

        if len(wn) > 1 and len(sp) > 1:
            positive = (
                np.isfinite(wn[1:]) & np.isfinite(sp[1:]) & (wn[1:] > 0) & (sp[1:] > 0)
            )

            if np.any(positive):
                ax2.loglog(
                    wn[1:][positive],
                    sp[1:][positive],
                    color="#D55E00",
                    linestyle="-",
                    linewidth=1.5,
                    marker="o",
                    markersize=3.2,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    label=r"$E(k)$",
                )

        ax2.set_xlabel(r"Wavenumber $k$")
        ax2.set_ylabel(r"Energy spectrum $E(k)$")
        ax2.set_title("(c) Final-state energy spectrum", loc="left", pad=6)

        if ax2.lines:
            ax2.legend(
                loc="best",
                frameon=True,
                facecolor="white",
                framealpha=0.9,
                edgecolor="0.8",
            )

        filename = f"post_plotting_{self.simulation_mode}.png"
        output_path = self.master_path / filename
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  * Saved post-simulation plot        -> {filename}")

        if show_plot:
            plt.show()

        plt.close(fig)

    def plot_tau_parameters(self):
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            np.arange(len(self.tau_advective_history)),
            self.tau_advective_history,
            label=r"$\tau_{\mathrm{adv}}$",
        )

        ax.plot(
            np.arange(len(self.tau_diffusive_history)),
            self.tau_diffusive_history,
            label=r"$\tau_{\mathrm{diff}}$",
        )

        ax.set_xlabel("Time")
        ax.set_title("Evolution of Stabilization Parameters")

        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        filename = "tau_paramaters.png"
        output_path = self.master_path / filename
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig, ax


# ==================================================================== #
#  STANDALONE SELF-TEST                                                #
#  Run with:  python solver_base_2.py                                 #
# ==================================================================== #
if __name__ == "__main__":
    from pathlib import Path

    class _ShimProblem:
        def __init__(self, viscosity, domain_length, t_start, domain_timespan):
            self.viscosity = viscosity
            self.domain_length = domain_length
            self.t_start = t_start
            self.domain_timespan = domain_timespan

    class _ShimDisc:
        def __init__(self, n_nodes, dt):
            self.n_nodes_dns = n_nodes
            self.n_nodes_les = n_nodes
            self.dt_dns = dt
            self.dt_les = dt
            self.suppress_file_logging = True

    class _ShimRef:
        def __init__(self, x, u0, forcing_fn, time_steps):
            self.initial_condition = u0
            self.initial_forcing = forcing_fn(x, 0.0)
            self._x = x
            self._f_func = forcing_fn
            self._t_list = time_steps
            self._step = 0

        def set_step_index(self, step):
            self._step = step

        @property
        def forcing(self):
            t = self._t_list[min(self._step, len(self._t_list) - 1)]
            return self._f_func(self._x, t)

    def _run_case(
        name,
        *,
        u0_func,
        f_func,
        nu,
        L,
        nx,
        T,
        dt,
        simulation_mode,
        tau_model,
        tau_mode,
        include_nonlinear,
        coefficients=None,
    ):
        n_nodes = nx + 1
        x = np.linspace(0, L, n_nodes)
        u0 = u0_func(x)

        n_steps = int(np.ceil(T / dt))
        time_steps = np.concatenate(([0.0], np.cumsum(np.full(n_steps, dt))))

        problem = _ShimProblem(nu, L, 0.0, T)
        disc = _ShimDisc(n_nodes, dt)
        ref = _ShimRef(x, u0, f_func, time_steps)

        workdir = Path(__file__).parent.parent.resolve() / "solver_data" / "test"
        workdir.mkdir(exist_ok=True, parents=True)

        solver = SolverBase2(
            problem=problem,
            disc_config=disc,
            reference_trajectory=ref,
            simulation_mode=simulation_mode,
            master_path=workdir
            / f"u0{u0_func.__name__}_f0{f_func.__name__}"
            / f"{str(include_nonlinear)}"
            / f"{tau_model}",
            tau_model=tau_model,
            tau_mode=tau_mode,
            include_nonlinear_vms=include_nonlinear,
            external_correction_coefficients=coefficients,
        )

        try:
            solver.run_simulation()
            solver.post_processing()
            ok = True
            err = float(np.linalg.norm(solver.solution))
        except Exception as exc:
            ok = False
            err = float("nan")
            print(f"  [{name}]  FAILED: {type(exc).__name__}: {exc}")

        tag = "OK" if ok else "FAIL"
        print(f"  [{name}] {tag:4s}  |u_final|={err:.4e}")
        return ok

    def u_zero(x):
        return np.zeros_like(x)

    def u_sin(x):
        return -np.sin(np.pi * x)

    def f_zero(x, t):
        return np.zeros_like(x)

    def f_const(x, t):
        return 0.1 * np.ones_like(x)

    def f_sin(x, t):
        return 0.1 * np.sin(np.pi * x)

    def f_ramp(x, t):
        return 0.1 * min(t, 1.0) * np.sin(np.pi * x)

    L, T, dt = 1.0, 0.05, 1e-3
    nu = 1e-4

    print()
    print("=" * 78)
    print("  SolverBase2 self-test")
    print("=" * 78)
    print(f"  L={L}, T={T}, dt={dt}, nu={nu}")
    print()

    results = {}

    # ---- Phase 1: pure Galerkin, no VMS. Uses NO_MODEL. ----
    print("Phase 1: no VMS (Galerkin only) — plain FEM + BDF2")
    print("-" * 78)
    for nx in (16, 32):
        for tag, u_fn, f_fn in (
            ("u0=sin f=0", u_sin, f_zero),
            ("u0=0   f=const", u_zero, f_const),
            ("u0=sin f=sin", u_sin, f_sin),
        ):
            results[f"gal_{tag}_nx{nx}"] = _run_case(
                f"gal {tag} nx={nx}",
                u0_func=u_fn,
                f_func=f_fn,
                nu=nu,
                L=L,
                nx=nx,
                T=T,
                dt=dt,
                simulation_mode=SimulationMode.NO_MODEL,
                tau_model=None,
                tau_mode=TauMode.FROZEN,
                include_nonlinear=False,
            )

    # ---- Phase 2: VMS cross-term only ----
    print()
    print("Phase 2: VMS cross-term only (no quadratic)")
    print("-" * 78)
    for tm in (TauModel.TWO_PARAMS_BASE, TauModel.THREE_PARAMS_BASE):
        results[f"vms_cross_{tm.value}"] = _run_case(
            f"vms-cross u0=sin {tm.value}",
            u0_func=u_sin,
            f_func=f_zero,
            nu=nu,
            L=L,
            nx=32,
            T=T,
            dt=dt,
            simulation_mode=SimulationMode.VMS,
            tau_model=tm,
            tau_mode=TauMode.FROZEN,
            include_nonlinear=False,
        )

    # ---- Phase 3: VMS full (quadratic on) ----
    print()
    print("Phase 3: VMS + quadratic")
    print("-" * 78)
    for tm in (TauModel.TWO_PARAMS_BASE, TauModel.THREE_PARAMS_BASE):
        for tmode in (TauMode.FROZEN, TauMode.HOT):
            results[f"vms_full_{tm.value}_{tmode.value}"] = _run_case(
                f"vms-full u0=sin {tm.value} {tmode.value}",
                u0_func=u_sin,
                f_func=f_zero,
                nu=nu,
                L=L,
                nx=32,
                T=T,
                dt=dt,
                simulation_mode=SimulationMode.VMS,
                tau_model=tm,
                tau_mode=tmode,
                include_nonlinear=True,
            )

    # ---- Phase 4: forcing sensitivity ----
    print()
    print("Phase 4: forcing sensitivity")
    print("-" * 78)
    for tag, f_fn in (
        ("f=0", f_zero),
        ("f=const", f_const),
        ("f=sin", f_sin),
        ("f=ramp", f_ramp),
    ):
        results[f"vms_force_{tag}"] = _run_case(
            f"vms-full u0=sin {tag}",
            u0_func=u_sin,
            f_func=f_fn,
            nu=nu,
            L=L,
            nx=32,
            T=T,
            dt=dt,
            simulation_mode=SimulationMode.VMS,
            tau_model=TauModel.TWO_PARAMS_BASE,
            tau_mode=TauMode.FROZEN,
            include_nonlinear=True,
        )

    # ---- Summary ----
    print()
    print("=" * 78)
    n_ok = sum(1 for v in results.values() if v)
    print(f"  {n_ok}/{len(results)} cases passed")
    print("=" * 78)
