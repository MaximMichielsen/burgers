from __future__ import annotations

import csv
import json
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence, Generator

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

from setup.config_discretization import DiscretizationConfig
from setup.constants import (
    TOLERANCE_RESIDUAL,
    NR_RELAXATION,
    FIXED_HOMOGENEOUS_BCS,
    N_GAUSS,
)
from setup.problems import Problem
from utils.diagnostics import (
    compute_energy,
    compute_dissipation,
    compute_energy_spectrum,
)


class SimulationMode(str, Enum):
    DNS = "dns"
    NO_MODEL = "no_model"
    VMS = "vms"


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


class DNSLoader:
    def __init__(self, path_to_velocity: Path, path_to_forcing: Path, t_start: float):
        self.velocity_data = np.load(path_to_velocity / "w.npy")
        self.velocity_times = np.load(path_to_velocity / "times.npy")

        self.forcing_data = np.load(path_to_forcing / "f.npy")
        self.forcing_times = np.load(path_to_forcing / "times.npy")

        self._step_offset_forcing = np.argmin(np.abs(self.forcing_times - t_start))

    def get_initial_condition(self, time: float):
        velocity_ic_index = np.argmin(np.abs(self.velocity_times - time))
        return self.velocity_data[velocity_ic_index]

    def get_load_vector(self, index: int, time: float) -> NDArray:
        """Load load-vector at time-index."""
        forcing_index = np.argmin(np.abs(self.forcing_times - time))
        given_index = index + self._step_offset_forcing
        if given_index != forcing_index:
            raise RuntimeError(
                f"Forcing index is inconsistent! received {index} with offset {self._step_offset_forcing}, calculated {forcing_index} for t={time}."
            )

        return self.forcing_data[given_index]


class SolverBase:
    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        reference_trajectory: DNSLoader | DNSLoaderTest,
        simulation_mode: SimulationMode,
        master_path: Path,
        tau_model: TauModel | None = None,
        external_correction_coefficients: NDArray | None = None,
        include_nonlinear_vms: bool = True,
        testing_mode: bool = False,
        exact_solution: Callable[[NDArray, float], NDArray] | None = None,
        comparison_label: str | None = None,
        do_post_processing: bool = True,
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

        self.problem = problem
        self.disc_config = disc_config
        self.reference_trajectory = reference_trajectory
        self.master_path = master_path
        self.forced_corrections = external_correction_coefficients
        self.include_nonlinear_vms = include_nonlinear_vms
        self.testing_mode = bool(testing_mode)
        self.exact_solution = exact_solution
        self.comparison_label = comparison_label or (
            "Galerkin"
            if self.simulation_mode == SimulationMode.NO_MODEL
            else self.simulation_mode.upper()
        )
        if self.testing_mode and self.exact_solution is None:
            raise ValueError(
                "testing_mode=True requires an exact_solution(x, t) callable."
            )

        self.viscosity = problem.viscosity

        self.convergence_tolerance = TOLERANCE_RESIDUAL
        self.nr_relaxation: float = NR_RELAXATION

        # temporal parameters
        self.time: float = problem.t_start
        self.time_elapsed: float = 0.0
        self.prev_elapsed: float = 0.0
        self.solver_step_current: int = 0
        self.simulation_done: bool = False
        self.domain_timespan: float = problem.domain_timespan
        self.dt: float = (
            disc_config.dt_dns if simulation_mode == "dns" else disc_config.dt_les
        )
        self.n_time_steps = disc_config.n_timesteps

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
        self.initial_condition: NDArray = reference_trajectory.get_initial_condition(
            time=problem.t_start
        )

        # solution fields
        self.solution_now: NDArray = self.initial_condition.copy()
        self.solution_previous: NDArray = np.zeros_like(self.solution_now)
        self.mean_solution: NDArray | None = None
        self.reynolds_stress: NDArray | None = None

        # forcing field
        self.forcing_now: NDArray = reference_trajectory.get_load_vector(
            index=self.solver_step_current, time=self.time
        )
        self.forcing_next: NDArray = reference_trajectory.get_load_vector(
            index=self.solver_step_current + 1, time=self.time + self.dt
        )

        self.gauss_points, self.gauss_weights = self.gauss_legendre(N_GAUSS)

        # output
        self._snapshot_step_indices: frozenset[int] = frozenset(
            range(0, self.disc_config.n_timesteps, 1)
        )
        self.time_steps = np.arange(0, self.n_time_steps, self.dt)
        self.requested_snapshots: NDArray = self.time_steps[
            sorted(self._snapshot_step_indices)
        ]
        self.is_written_at_times: list[bool] | None = [False] * len(
            self.requested_snapshots
        )
        self.master_path: Path = master_path
        self.master_path.mkdir(parents=True, exist_ok=True)

        # snapshots
        self.snapshots_solution: list[NDArray] = []
        self.snapshots_forcing: list[NDArray] = []

        # Benchmarking
        self.run_id: str = datetime.now().strftime("%m%d_%H%M%S")
        self.timings_performance: dict = {}
        self.nr_residual_history: list = []
        self.nr_update_history: list = []
        self.frozen_tau_history: list = []
        self.energy_history: list = []
        self.dissipation_history: list = []
        self.logger = self._setup_logger(
            suppress_file_logging=disc_config.suppress_file_logging
        )

        # VMS parameter tracking
        self.current_nr_iteration = 0
        self.tau_history = []
        self.tau_advective_history = []
        self.tau_diffusive_history = []
        self.tau_temporal_history = []
        self.strong_residual_history = []

        # MMS verification history (populated only in testing mode).
        self.error_times: list[float] = []
        self.relative_l2_error_history: list[float] = []
        self.linf_error_history: list[float] = []
        if self.testing_mode:
            self._record_mms_error()

        self.do_post_processing = do_post_processing

    # ------------------------------------------------------------------ #
    #  Core
    # ------------------------------------------------------------------ #

    def run_simulation(self) -> None:
        with self.timer("total_simulation"):
            with tqdm(
                total=self.n_time_steps,
                desc=f"Eating Burgers | {self.throbber(0)}",
                file=sys.stdout,
            ) as pbar:
                for time_step in range(self.n_time_steps):
                    step_start = perf_counter()
                    self.advance_time_step()

                    pbar.set_description(f"Eating Burgers | {self.throbber(time_step)}")
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "t": f"{self.time:.3f}",
                            "dt": f"{self.dt:.3f}",
                            "step_time": f"{perf_counter() - step_start:.3f}s",
                        }
                    )

    def advance_time_step(self) -> None:
        """Advance one timestep, committing only converged solutions."""
        # Previous solution and time history remain untouched on failure.
        if self.solver_step_current == 0:
            self._extract_snapshot()  # add IC to snapshots

        self.set_forcing_current()
        # The nonlinear residual is assembled at t_{n+1}; refresh its forcing
        # every step rather than retaining the first timestep's forcing_next.
        self.forcing_next = self.get_forcing_next()

        solution_next = self.nr_iteration(
            self.solution_now,
            self.solution_previous,
        )

        old_solution = self.solution_now.copy()
        self.solution_previous = old_solution
        self.solution_now = solution_next

        self.energy_history.append(self.compute_energy_(self.solution_now))
        self.dissipation_history.append(self.compute_dissipation_(self.solution_now))

        self.time += self.dt
        self.time_elapsed += self.dt
        self.solver_step_current += 1

        if self.testing_mode:
            self._record_mms_error()

        self._extract_snapshot()

        if int(self.time_elapsed) > int(self.prev_elapsed):
            self.mean_solution = self.calculate_mean_profile()
            self.write_mean_solution_to_csv()

        self.prev_elapsed = self.time_elapsed

        if np.isclose(self.time_elapsed, self.domain_timespan, rtol=self.dt / 2):
            self.simulation_done = True
            self.mean_solution = self.calculate_mean_profile()
            self.reynolds_stress = self.compute_reynolds_stress()
            self.write_mean_solution_to_csv()
            self.write_reynolds_profile()

            if self.do_post_processing:
                self.post_processing()
                self.write_config_to_json()

    def nr_iteration(
        self,
        solution_now: NDArray,
        solution_previous: NDArray,
        max_iterations: int = 100,
    ) -> NDArray:
        """Newton solve with selectable frozen or lagged-hot element tau."""
        # NR Initialization -----------------------------------------------------------
        solution_n_min1 = solution_previous.copy()
        solution_n = solution_now.copy()
        solution_k = solution_now.copy()

        residual_history_loop = []
        update_history_loop = []

        # Frozen Tau -----------------------------------------------------------
        if self._use_vms:
            frozen_taus = self._evaluate_element_tau(solution_n, record_components=True)
        else:
            frozen_taus = [None] * self.n_elements
        self.frozen_tau_history.append(np.asarray(frozen_taus, dtype=float))

        # NR Iteration -----------------------------------------------------------

        converged = False
        self.current_nr_iteration = 0
        for iteration in range(max_iterations + 1):
            self.current_nr_iteration += 1
            global_residual, global_jacobian = self.assemble_global_system(
                solution_k, solution_n, solution_n_min1, frozen_taus
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
                    f"{self.solver_step_current + 1}, iteration {iteration + 1}."
                ) from exc

            update_history_loop.append(float(np.linalg.norm(delta_u)))

            alpha = 1.0
            min_alpha = 1e-6
            accepted = False
            trial_history = []

            while alpha >= min_alpha:
                candidate = solution_k + alpha * self.nr_relaxation * delta_u
                candidate_residual, _ = self.assemble_global_system(
                    candidate, solution_n, solution_n_min1, frozen_taus
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
                self.mean_solution = self.calculate_mean_profile()
                self.reynolds_stress = self.compute_reynolds_stress()
                self.write_mean_solution_to_csv()
                self.write_reynolds_profile()
                self.post_processing()

                print(f"\nNewton failure: timestep {self.solver_step_current + 1}")
                print(f"Iteration: {iteration + 1}")
                print(f"Current residual: {residual_norm:.8e}")
                print(f"Newton update norm: {np.linalg.norm(delta_u):.8e}")
                for alpha_trial, norm_trial in trial_history:
                    print(f"alpha={alpha_trial:.8e}, residual={norm_trial:.8e}")
                raise RuntimeError("Newton line search failed.")

        self.nr_residual_history.append(residual_history_loop)
        self.nr_update_history.append(update_history_loop)

        if not converged:
            raise RuntimeError(
                f"Newton failed to converge at timestep {self.solver_step_current + 1} "
                f"after {max_iterations} iterations. Final residual: "
                f"{residual_history_loop[-1]:.6e}"
            )

        return solution_k

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
                element_solution_k=solution_k[element],
                element_solution_n=solution_n[element],
                element_solution_nm1=solution_nm1[element],
                element_forcing=self.forcing_next[element],
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
        element_solution_k: NDArray,
        element_solution_n: NDArray,
        element_solution_nm1: NDArray,
        element_forcing: NDArray,
        tau_e: float | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Compute the elemental residual and Jacobian, including VMS terms."""
        residual_element = np.zeros(2)
        jacobian_element = np.zeros((2, 2))

        d_ksi_jacobian = self.element_size / 2
        gauss_points, gauss_weights = self.gauss_points, self.gauss_weights
        gradient_basis = self.gradient_basis_functions()

        for gauss_point, gauss_weight in zip(gauss_points, gauss_weights):
            basis = self.basis_functions(gauss_point)
            interpolated_fields = self._interpolate_fields(
                basis=basis,
                gradient_basis=gradient_basis,
                u_k=element_solution_k,
                u_n=element_solution_n,
                u_nm1=element_solution_nm1,
                forcing=element_forcing,
            )
            scale = gauss_weight * d_ksi_jacobian

            strong_residual = self.strong_elemental_residual(interpolated_fields)
            self.strong_residual_history.append(
                {
                    "step": self.solver_step_current,
                    "iteration": self.current_nr_iteration,
                    "element": element[0],
                    "gauss_point": gauss_point,
                    "x": element[0] * self.element_size
                    + self.element_size / 2 * (gauss_point + 1),
                    "residual": strong_residual,
                }
            )

            for e_i in range(len(element)):
                node_residual_galerkin = scale * self.galerkin_elemental_residual(
                    e_i=e_i,
                    basis=basis,
                    gradient_basis=gradient_basis,
                    f=interpolated_fields,
                )
                residual_element[e_i] += node_residual_galerkin

                if self._use_vms and tau_e is not None:
                    node_residual_vms_advection, node_residual_vms_nonlinear = (
                        scale
                        * np.asarray(
                            self.vms_elemental_residual(
                                i=e_i,
                                gradient_basis=gradient_basis,
                                f=interpolated_fields,
                                strong_res=strong_residual,
                                tau_e=tau_e,
                                include_nonlinear=self.include_nonlinear_vms,
                            )
                        )
                    )
                    residual_element[e_i] += (
                        node_residual_vms_advection + node_residual_vms_nonlinear
                    )

                for e_j in range(len(element)):
                    jacobian_element[e_i, e_j] += (
                        scale
                        * self.galerkin_elemental_jacobian(
                            e_i, e_j, basis, gradient_basis, interpolated_fields
                        )
                    )

                    if self._use_vms and tau_e is not None:
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
                                include_nonlinear=self.include_nonlinear_vms,
                            )
                        )

        return residual_element, jacobian_element

    @staticmethod
    def _interpolate_fields(
        basis: NDArray,
        gradient_basis: NDArray,
        u_k: NDArray,
        u_n: NDArray,
        u_nm1: NDArray,
        forcing: NDArray,
    ) -> dict:
        """Interpolate solution and gradient at a quadrature point."""
        return {
            "u_k": basis @ u_k,
            "u_n": basis @ u_n,
            "u_nm1": basis @ u_nm1,
            "forcing": basis @ forcing,
            "du_dx_k": gradient_basis @ u_k,
            "du_dx_n": gradient_basis @ u_n,
        }

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
    #  Galerkin element contribution
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Unresolved-scale element contribution
    # ------------------------------------------------------------------ #

    @staticmethod
    def vms_elemental_residual(
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
            nonlinear -= 0.5 * (tau_e * strong_res) ** 2
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
        include_nonlinear: bool = True,
    ) -> float:
        """Consistent derivative of the VMS residual with respect to u_k[j].

        Tau is treated as frozen, consistently with the element-wise tau
        evaluation performed before the Newton iterations of each timestep.
        """
        a0, _, _ = self._time_derivative_coefficients()

        # dR/du_j for R = (a0*u_k + a1*u_n + a2*u_nm1)/dt
        #                + u_k*(du_k/dx) - f.
        d_strong_residual = (
            (a0 / self.dt) * basis[j]
            + basis[j] * f["du_dx_k"]
            + f["u_k"] * gradient_basis[j]
        )

        # d(tau*u_k*R)/du_j, with frozen tau.
        derivative = tau_e * (basis[j] * strong_res + f["u_k"] * d_strong_residual)

        if include_nonlinear:
            derivative -= tau_e**2 * strong_res * d_strong_residual

        return gradient_basis[i] * derivative

    # ------------------------------------------------------------------ #
    #  URSM
    # ------------------------------------------------------------------ #

    def strong_elemental_residual(self, fields: dict) -> float:
        """Strong residual with BDF1 startup and BDF2 thereafter."""
        a0, a1, a2 = self._time_derivative_coefficients()
        time_term = (
            a0 * fields["u_k"] + a1 * fields["u_n"] + a2 * fields["u_nm1"]
        ) / self.dt
        return time_term + fields["u_k"] * fields["du_dx_k"] - fields["forcing"]

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
            tau_e = self.compute_tau(
                u_e,
                du_e,
                self.forced_corrections,
                record_components=record_components,
            )
            taus.append(float(tau_e))
        return taus

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
        if (
            self.tau_model
            in (
                TauModel.THREE_PARAMS_BASE,
                TauModel.THREE_PARAMS_TEMPORAL,
            )
            and u_x_e is not None
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
    #  Time Discretization
    # ------------------------------------------------------------------ #

    def _time_derivative_coefficients(self) -> tuple[float, float, float]:
        """BDF1 startup, followed by constant-step BDF2."""
        if self.solver_step_current == 0:
            return 1.0, -1.0, 0.0

        return 1.5, -2.0, 0.5

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def initialize_elements(self) -> NDArray:
        """Build element connectivity array [[0,1], [1,2], …]."""
        return np.column_stack((self.nodes[:-1], self.nodes[1:]))

    # ------------------------------------------------------------------ #
    #  Index Tracking
    # ------------------------------------------------------------------ #

    def set_forcing_current(self) -> None:
        self.forcing_now = self.reference_trajectory.get_load_vector(
            index=self.solver_step_current, time=self.time
        )

    def get_forcing_next(self) -> NDArray:
        return self.reference_trajectory.get_load_vector(
            index=self.solver_step_current + 1, time=self.time + self.dt
        )

    # ------------------------------------------------------------------ #
    #  MMS verification
    # ------------------------------------------------------------------ #

    def _record_mms_error(self) -> None:
        """Record relative spatial L2 and absolute L-infinity MMS errors."""
        if not self.testing_mode or self.exact_solution is None:
            return

        exact = np.asarray(self.exact_solution(self.mesh, self.time), dtype=float)
        numerical = np.asarray(self.solution_now, dtype=float)

        if exact.shape != numerical.shape:
            raise ValueError(
                f"Exact solution shape {exact.shape} does not match numerical "
                f"solution shape {numerical.shape}."
            )

        error = numerical - exact
        error_l2 = float(np.sqrt(np.trapezoid(error**2, self.mesh)))
        exact_l2 = float(np.sqrt(np.trapezoid(exact**2, self.mesh)))
        relative_l2 = (
            error_l2 / exact_l2 if exact_l2 > np.finfo(float).eps else error_l2
        )

        self.error_times.append(float(self.time))
        self.relative_l2_error_history.append(relative_l2)
        self.linf_error_history.append(float(np.max(np.abs(error))))

    def plot_relative_error_comparison(
        self,
        comparison_data: dict[str, tuple[Sequence[float], Sequence[float]]]
        | None = None,
    ):
        """Plot MMS relative-L2 error histories for one or more solver variants.

        `comparison_data` is extensible for later Galerkin/VMS/tau comparisons:
        {"label": (times, relative_errors)}. With no argument, this run alone
        is plotted.
        """
        if not self.testing_mode:
            raise RuntimeError(
                "Relative-error comparison is available only in testing_mode."
            )

        if comparison_data is None:
            comparison_data = {
                self.comparison_label: (
                    self.error_times,
                    self.relative_l2_error_history,
                )
            }

        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
        for label, (times, errors) in comparison_data.items():
            ax.semilogy(times, errors, linewidth=1.6, label=label)

        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"Relative $L^2$ error")
        ax.set_title("MMS verification: relative error comparison")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()

        filename = "mms_relative_error_comparison.png"
        fig.savefig(self.master_path / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  * Saved MMS relative-error comparison -> {filename}")

    # ------------------------------------------------------------------ #
    #  Statistics
    # ------------------------------------------------------------------ #

    def calculate_mean_profile(self) -> np.ndarray:
        """Computes the time-averaged spatial mean profile."""
        if self.snapshots_solution is None or len(self.snapshots_solution) == 0:
            raise ValueError(
                "No solution history found. Run the solver before computing statistics."
            )
        return np.mean(np.asarray(self.snapshots_solution), axis=0)

    def compute_reynolds_stress(self) -> NDArray:
        """Compute Reynolds stress <u'u'>. In 1D equivalent to the variance of the solution field."""
        snapshots = np.asarray(self.snapshots_solution)
        mean = np.mean(snapshots, axis=0)
        return np.mean((snapshots - mean) ** 2, axis=0)

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
    #  Convergence checking
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_residual_converged(
        residual: float | NDArray, tolerance: float = 1e-8
    ) -> bool:
        """True when the relative residual norm is below tolerance."""
        norm = np.linalg.norm(residual)
        return norm < tolerance * (1 + norm)

    # ------------------------------------------------------------------ #
    #  Internal flags
    # ------------------------------------------------------------------ #

    @property
    def _use_vms(self) -> bool:
        """True only for analytic VMS mode (les)."""
        return self.simulation_mode == SimulationMode.VMS

    # ------------------------------------------------------------------ #
    #  inter-run diagnostics
    # ------------------------------------------------------------------ #

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
    #  Post-processing
    # ------------------------------------------------------------------ #

    def post_processing(self) -> None:
        """Run post-plotting and post-logging."""
        self.post_plotting()
        self.plot_tau_parameters()
        if self.testing_mode:
            self.plot_relative_error_comparison()
        self.post_logging()

    # ------------------------------------------------------------------ #
    #  Writing
    # ------------------------------------------------------------------ #

    def _extract_snapshot(self) -> None:
        """Store current solution and forcing as a snapshot."""
        self.snapshots_solution.append(self.solution_now.copy())
        self.snapshots_forcing.append(self.forcing_now.copy())

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
            "domain_timespan": self.domain_timespan,
            "domain_length": self.domain_length,
            "dt": self.dt,
            "n_time_steps": self.n_time_steps,
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
            "n_time_steps": str(self.n_time_steps),
            "viscosity": f"{self.viscosity:.4e}",
            "n_nodes": str(self.n_nodes),
            "n_elements": str(self.n_elements),
            "element_size": f"{self.element_size:.4e}",
            "requested_snapshots": snapshots_str,
        }

    # ------------------------------------------------------------------ #
    #  Plotting
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_positive_spectrum(
        wavenumbers: NDArray, spectrum: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Filter to non-negative wavenumbers."""
        mask = wavenumbers >= 0
        return wavenumbers[mask], spectrum[mask]

    def post_plotting(self) -> None:
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
        ax0.plot(
            self.mesh, self.solution_now, label="Resolved solution", **resolved_style
        )
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
            *self.compute_energy_spectrum_(self.solution_now)
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
        print(f"  * Saved post-simulation plot        -> {self.master_path}/{filename}")
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


@dataclass
class DNSLoaderTest:
    velocity_data: NDArray
    velocity_times: NDArray
    forcing_data: NDArray
    forcing_times: NDArray
    t_start: float

    def __post_init__(self):
        self._step_offset_forcing = np.argmin(np.abs(self.forcing_times - self.t_start))

    def get_initial_condition(self, time: float):
        velocity_ic_index = np.argmin(np.abs(self.velocity_times - time))
        return self.velocity_data[velocity_ic_index]

    def get_load_vector(self, index: int, time: float) -> NDArray:
        """Load load-vector at time-index."""
        forcing_index = np.argmin(np.abs(self.forcing_times - time))
        given_index = index + self._step_offset_forcing
        if given_index != forcing_index:
            raise RuntimeError(
                f"Forcing index is inconsistent! received {index} with offset {self._step_offset_forcing}, calculated {forcing_index} for t={time}."
            )

        return self.forcing_data[given_index]


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.resolve()
    master_dir = project_root / "runs" / "solver_testing"

    domain_length = 2.0
    n_nodes_les = 33
    dt_les = 0.1
    t_start = 0.0
    t_end = 4

    run_dir = master_dir / f"n{n_nodes_les}_dt{dt_les}_t0{t_start}_tf{t_end}"

    # Discretization
    mesh_les = np.linspace(0.0, domain_length, n_nodes_les)
    times_velocity = np.arange(t_start, t_end + 0.5 * dt_les, dt_les)

    # Manufactured problem
    test_problem = Problem(
        domain_length=domain_length, t_start=t_start, t_end=t_end, name="test_problem"
    )

    nu = test_problem.viscosity
    k = np.pi / domain_length

    # Broadcast spatial and temporal coordinates
    X = mesh_les[None, :]
    T = times_velocity[:, None]

    # Spatial modes
    S1 = np.sin(5.0 * k * X)
    S2 = np.sin(9.0 * k * X)
    S3 = np.sin(13.0 * k * X)

    C1 = np.cos(5.0 * k * X)
    C2 = np.cos(9.0 * k * X)
    C3 = np.cos(13.0 * k * X)

    # Temporal amplitudes
    a1 = np.cos(3.0 * T)
    a2 = 0.35 * np.sin(7.0 * T)
    a3 = 0.15 * np.cos(11.0 * T)

    # Temporal derivatives
    a1_t = -3.0 * np.sin(3.0 * T)
    a2_t = 2.45 * np.cos(7.0 * T)
    a3_t = -1.65 * np.sin(11.0 * T)

    # Exact velocity
    velocity_data = a1 * S1 + a2 * S2 + a3 * S3

    # Exact derivatives
    u_t = a1_t * S1 + a2_t * S2 + a3_t * S3

    u_x = k * (5.0 * a1 * C1 + 9.0 * a2 * C2 + 13.0 * a3 * C3)

    u_xx = -(k**2) * (25.0 * a1 * S1 + 81.0 * a2 * S2 + 169.0 * a3 * S3)

    # Manufactured forcing
    forcing_data = u_t + velocity_data * u_x - nu * u_xx

    forcing_times = times_velocity.copy()

    def manufactured_solution(x: NDArray, t: float) -> NDArray:
        return (
            np.sin(5.0 * k * x) * np.cos(3.0 * t)
            + 0.35 * np.sin(9.0 * k * x) * np.sin(7.0 * t)
            + 0.15 * np.sin(13.0 * k * x) * np.cos(11.0 * t)
        )

    # Reference trajectory
    test_reference = DNSLoaderTest(
        velocity_data=velocity_data,
        velocity_times=times_velocity,
        forcing_data=forcing_data,
        forcing_times=forcing_times,
        t_start=t_start,
    )

    test_disc_config = DiscretizationConfig(
        test_problem,
        n_nodes_les,
        dt_les,
    )

    # Run all three formulations against the same manufactured solution.
    # The VMS residual is active, while its Jacobian contribution is
    # intentionally omitted for now (Galerkin Jacobian only).
    run_specs = (
        ("Galerkin", SimulationMode.NO_MODEL, None),
        ("VMS - 2 parameter base", SimulationMode.VMS, TauModel.TWO_PARAMS_BASE),
        ("VMS - 3 parameter base", SimulationMode.VMS, TauModel.THREE_PARAMS_BASE),
    )

    comparison_data: dict[str, tuple[Sequence[float], Sequence[float]]] = {}
    solvers: dict[str, SolverBase] = {}

    for label, simulation_mode, tau_model in run_specs:
        run_dir = (
            master_dir
            / f"n{n_nodes_les}_dt{dt_les}_t0{t_start}_tf{t_end}"
            / label.lower().replace(" ", "_").replace("-", "")
        )

        solver = SolverBase(
            test_problem,
            test_disc_config,
            simulation_mode=simulation_mode,
            reference_trajectory=test_reference,
            master_path=run_dir,
            tau_model=tau_model,
            testing_mode=True,
            exact_solution=manufactured_solution,
            comparison_label=label,
            include_nonlinear_vms=True,
        )

        print(f"\\nRunning {label} ...")
        solver.run_simulation()

        solvers[label] = solver
        comparison_data[label] = (
            solver.error_times,
            solver.relative_l2_error_history,
        )

    # One shared plot containing Galerkin and both VMS tau-model histories.
    comparison_solver = solvers["Galerkin"]
    comparison_solver.master_path = (
        master_dir / f"n{n_nodes_les}_dt{dt_les}_t0{t_start}_tf{t_end}"
    )
    comparison_solver.master_path.mkdir(parents=True, exist_ok=True)
    comparison_solver.plot_relative_error_comparison(comparison_data)

    print("Velocity shape:", velocity_data.shape)
    print("Forcing shape:", forcing_data.shape)

    print("Left boundary max:", np.max(np.abs(velocity_data[:, 0])))

    print("Right boundary max:", np.max(np.abs(velocity_data[:, -1])))

    print("Forcing min/max:", forcing_data.min(), forcing_data.max())
