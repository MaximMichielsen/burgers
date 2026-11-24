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
_GAUSS_POINTS, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(3)
_GAUSS_BASIS = np.column_stack(
    (0.5 * (1.0 - _GAUSS_POINTS), 0.5 * (1.0 + _GAUSS_POINTS))
)
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
        self._gradient_basis = np.array(
            [-1.0 / self.element_size, 1.0 / self.element_size]
        )

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
        """Newton solve with elementwise tau frozen at the start of the timestep."""
        solution_nm1 = solution_prev
        solution_n = solution
        solution_k = solution.copy()

        # Tau is evaluated exactly once per timestep, from u_n.
        if self._use_vms:
            frozen_taus = self._evaluate_element_tau(solution_n, record_components=True)
            self.frozen_tau_history.append(np.asarray(frozen_taus, dtype=float))
        else:
            frozen_taus = [None] * self.n_elements

        residual_history_loop: list[float] = []
        update_history_loop: list[float] = []

        for iteration in range(max_iterations + 1):
            self.current_nr_iteration = iteration
            residual, jacobian_bands = self._assemble_global_system(
                solution_k,
                solution_n,
                solution_nm1,
                frozen_taus,
                with_jacobian=True,
                record_diagnostics=True,
            )
            residual_norm = float(np.linalg.norm(residual))
            residual_history_loop.append(residual_norm)

            if self.is_residual_converged(residual_norm, self.convergence_tolerance):
                break
            if iteration == max_iterations:
                break

            try:
                delta_u = self._solve_tridiagonal(jacobian_bands, -residual)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    f"Singular Newton Jacobian at timestep {self.current_solver_step + 1}, "
                    f"iteration {iteration + 1}."
                ) from exc

            update_history_loop.append(float(np.linalg.norm(delta_u)))

            # Residual-only line search: do not assemble or allocate a Jacobian.
            alpha = 1.0
            min_alpha = 1e-6
            accepted = False
            trial_history: list[tuple[float, float]] = []
            while alpha >= min_alpha:
                candidate = solution_k + alpha * self.nr_relaxation * delta_u
                candidate_residual, _ = self._assemble_global_system(
                    candidate,
                    solution_n,
                    solution_nm1,
                    frozen_taus,
                    with_jacobian=False,
                    record_diagnostics=False,
                )
                candidate_norm = float(np.linalg.norm(candidate_residual))
                trial_history.append((alpha, candidate_norm))
                if np.isfinite(candidate_norm) and candidate_norm < residual_norm:
                    solution_k = candidate
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                print(f"\nNewton failure: timestep {self.current_solver_step + 1}")
                print(f"Iteration: {iteration + 1}")
                print(f"Current residual: {residual_norm:.8e}")
                print(f"Newton update norm: {np.linalg.norm(delta_u):.8e}")
                for alpha_trial, norm_trial in trial_history:
                    print(f"alpha={alpha_trial:.8e}, residual={norm_trial:.8e}")
                raise RuntimeError("Newton line search failed.")
        else:  # pragma: no cover - loop always exits through a condition above
            pass

        self.residual_history.append(residual_history_loop)
        self.update_history.append(update_history_loop)
        if not self.is_residual_converged(
            residual_history_loop[-1], self.convergence_tolerance
        ):
            raise RuntimeError(
                f"Newton failed to converge at timestep {self.current_solver_step + 1} "
                f"after {max_iterations} iterations. Final residual: {residual_history_loop[-1]:.6e}"
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
        """Write the frozen elementwise tau history to CSV and a heatmap."""
        if not self._use_vms or not self.frozen_tau_history:
            return

        output_dir = self.master_path / "tau_history"
        output_dir.mkdir(parents=True, exist_ok=True)
        tau = np.asarray(self.frozen_tau_history, dtype=float)
        times = np.asarray(self.time_steps[1 : len(tau) + 1], dtype=float)
        if len(times) != len(tau):
            times = np.arange(1, len(tau) + 1, dtype=float)
        x_centres = 0.5 * (self.mesh[:-1] + self.mesh[1:])

        with (output_dir / "frozen_tau.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestep", "time", "element", "x_centre", "tau_frozen"])
            for step, row in enumerate(tau):
                time_value = float(times[step])
                for element, value in enumerate(row):
                    writer.writerow(
                        [step + 1, time_value, element, x_centres[element], value]
                    )

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        image = axes[0].imshow(
            tau,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[x_centres[0], x_centres[-1], times[0], times[-1]],
            cmap="viridis",
        )
        axes[0].set(
            title=r"Frozen stabilisation parameter $\tau(u_n)$",
            xlabel="Element centre x",
            ylabel="Time",
        )
        fig.colorbar(image, ax=axes[0], label=r"$\tau$")
        axes[1].plot(x_centres, tau[-1])
        axes[1].set(
            title="Frozen tau at final timestep",
            xlabel="Element centre x",
            ylabel=r"$\tau$",
        )
        axes[1].grid(True, alpha=0.3)
        fig.savefig(output_dir / "frozen_tau.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

    def assemble_global_system(
        self,
        solution_k: NDArray,
        solution_n: NDArray,
        solution_nm1: NDArray,
        taus: Sequence[float | None],
    ) -> tuple[NDArray, NDArray]:
        """Assemble residual and tridiagonal Jacobian bands."""
        residual, bands = self._assemble_global_system(
            solution_k,
            solution_n,
            solution_nm1,
            taus,
            with_jacobian=True,
            record_diagnostics=False,
        )
        # Preserve the original public API (dense Jacobian). The Newton solver
        # calls the private banded path directly and never pays this cost.
        jacobian = np.diag(bands[1])
        jacobian += np.diag(bands[0, 1:], 1)
        jacobian += np.diag(bands[2, :-1], -1)
        return residual, jacobian

    def _assemble_global_system(
        self,
        solution_k: NDArray,
        solution_n: NDArray,
        solution_nm1: NDArray,
        taus: Sequence[float | None],
        *,
        with_jacobian: bool,
        record_diagnostics: bool,
    ) -> tuple[NDArray, NDArray | None]:
        """Assemble directly into global arrays, exploiting 1D P1 sparsity.

        Jacobian storage is three rows: upper, diagonal, lower. The layout is
        compatible with the Thomas solver below and requires O(N) memory.
        """
        n = self.n_nodes
        residual = np.zeros(n, dtype=float)
        if with_jacobian:
            bands = np.zeros((3, n), dtype=float)
        else:
            bands = None

        h = self.element_size
        grad0, grad1 = -1.0 / h, 1.0 / h
        a0, a1, a2 = self._time_derivative_coefficients()
        inv_dt = 1.0 / self.dt
        nu = self.viscosity
        nonlinear_vms = self.include_nonlinear_vms
        use_vms = self._use_vms

        for e, (left, right) in enumerate(self.elements):
            uk0, uk1 = solution_k[left], solution_k[right]
            un0, un1 = solution_n[left], solution_n[right]
            um0, um1 = solution_nm1[left], solution_nm1[right]
            f0, f1 = self.forcing_current[left], self.forcing_current[right]
            tau = taus[e] if use_vms else None

            re0 = re1 = 0.0
            j00 = j01 = j10 = j11 = 0.0
            du = grad0 * uk0 + grad1 * uk1
            time0 = (a0 * uk0 + a1 * un0 + a2 * um0) * inv_dt
            time1 = (a0 * uk1 + a1 * un1 + a2 * um1) * inv_dt

            for q in range(3):
                b0, b1 = _GAUSS_BASIS[q]
                scale = _GAUSS_WEIGHTS[q] * (0.5 * h)
                u = b0 * uk0 + b1 * uk1
                time_value = b0 * time0 + b1 * time1
                forcing = b0 * f0 + b1 * f1
                strong = time_value + u * du - forcing

                if record_diagnostics:
                    self.strong_residual_history.append(
                        {
                            "step": self.current_solver_step,
                            "iteration": self.current_nr_iteration,
                            "element": int(left),
                            "gauss_point": float(_GAUSS_POINTS[q]),
                            "x": float(left * h + 0.5 * h * (_GAUSS_POINTS[q] + 1.0)),
                            "residual": float(strong),
                        }
                    )

                # Galerkin residual.
                re0 += scale * (
                    b0 * time_value + nu * du * grad0 + b0 * u * du - b0 * forcing
                )
                re1 += scale * (
                    b1 * time_value + nu * du * grad1 + b1 * u * du - b1 * forcing
                )

                if tau is not None:
                    fine_value = tau * u * strong
                    if nonlinear_vms:
                        fine_value -= 0.5 * (tau * strong) ** 2
                    re0 += scale * grad0 * fine_value
                    re1 += scale * grad1 * fine_value

                if not with_jacobian:
                    continue

                # d(strong residual)/d(u_j), j=left/right.
                dstrong0 = a0 * inv_dt * b0 + du * b0 + u * grad0
                dstrong1 = a0 * inv_dt * b1 + du * b1 + u * grad1

                # Galerkin Jacobian entries.
                g00 = (
                    a0 * inv_dt * b0 * b0
                    + nu * grad0 * grad0
                    + b0 * (b0 * du + u * grad0)
                )
                g01 = (
                    a0 * inv_dt * b0 * b1
                    + nu * grad0 * grad1
                    + b0 * (b1 * du + u * grad1)
                )
                g10 = (
                    a0 * inv_dt * b1 * b0
                    + nu * grad1 * grad0
                    + b1 * (b0 * du + u * grad0)
                )
                g11 = (
                    a0 * inv_dt * b1 * b1
                    + nu * grad1 * grad1
                    + b1 * (b1 * du + u * grad1)
                )

                if tau is not None:
                    if nonlinear_vms:
                        v0 = tau * (b0 * strong + (u - tau * strong) * dstrong0)
                        v1 = tau * (b1 * strong + (u - tau * strong) * dstrong1)
                    else:
                        v0 = tau * (b0 * strong + u * dstrong0)
                        v1 = tau * (b1 * strong + u * dstrong1)
                    g00 += grad0 * v0
                    g01 += grad0 * v1
                    g10 += grad1 * v0
                    g11 += grad1 * v1

                j00 += scale * g00
                j01 += scale * g01
                j10 += scale * g10
                j11 += scale * g11

            residual[left] += re0
            residual[right] += re1
            if with_jacobian:
                bands[1, left] += j00
                bands[0, left + 1] += j01
                bands[2, left] += j10
                bands[1, right] += j11

        # Row-replacement Dirichlet conditions, matching the original solver.
        left_bc, right_bc = self.boundary_values
        residual[0] = solution_k[0] - left_bc
        residual[-1] = solution_k[-1] - right_bc
        if with_jacobian:
            bands[1, 0] = 1.0
            bands[0, 1] = 0.0
            bands[1, -1] = 1.0
            bands[2, -2] = 0.0
        return residual, bands

    @staticmethod
    def _solve_tridiagonal(bands: NDArray, rhs: NDArray) -> NDArray:
        """Solve a tridiagonal system stored as [upper, diagonal, lower]."""
        n = rhs.size
        upper = bands[0]
        diagonal = bands[1]
        lower = bands[2]
        c = np.empty(max(n - 1, 0), dtype=float)
        d = np.empty(n, dtype=float)
        tiny = np.finfo(float).tiny

        pivot = diagonal[0]
        if not np.isfinite(pivot) or abs(pivot) <= tiny:
            raise np.linalg.LinAlgError("Zero or non-finite tridiagonal pivot.")
        if n > 1:
            c[0] = upper[1] / pivot
        d[0] = rhs[0] / pivot

        for i in range(1, n):
            pivot = diagonal[i] - lower[i - 1] * c[i - 1]
            if not np.isfinite(pivot) or abs(pivot) <= tiny:
                raise np.linalg.LinAlgError(
                    f"Zero or non-finite tridiagonal pivot at row {i}."
                )
            if i < n - 1:
                c[i] = upper[i + 1] / pivot
            d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / pivot

        solution = np.empty(n, dtype=float)
        solution[-1] = d[-1]
        for i in range(n - 2, -1, -1):
            solution[i] = d[i] - c[i] * solution[i + 1]
        return solution

    def calculate_elemental_residual_jacobian(
        self,
        element: tuple[int, int],
        u_k_e: NDArray,
        u_n_e: NDArray,
        u_nm1_e: NDArray,
        f_e: NDArray,
        tau_e: float | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Reference-compatible local residual/Jacobian using frozen tau."""
        residual = np.zeros(2, dtype=float)
        jacobian = np.zeros((2, 2), dtype=float)
        h = self.element_size
        grad = (-1.0 / h, 1.0 / h)
        a0, a1, a2 = self._time_derivative_coefficients()
        inv_dt = 1.0 / self.dt
        du = grad[0] * u_k_e[0] + grad[1] * u_k_e[1]
        time0 = (a0 * u_k_e[0] + a1 * u_n_e[0] + a2 * u_nm1_e[0]) * inv_dt
        time1 = (a0 * u_k_e[1] + a1 * u_n_e[1] + a2 * u_nm1_e[1]) * inv_dt

        for q in range(3):
            b0, b1 = _GAUSS_BASIS[q]
            scale = _GAUSS_WEIGHTS[q] * 0.5 * h
            u = b0 * u_k_e[0] + b1 * u_k_e[1]
            time_value = b0 * time0 + b1 * time1
            forcing = b0 * f_e[0] + b1 * f_e[1]
            strong = time_value + u * du - forcing
            for i, bi in enumerate((b0, b1)):
                gi = grad[i]
                residual[i] += scale * (
                    bi * time_value
                    + self.viscosity * du * gi
                    + bi * u * du
                    - bi * forcing
                )
                if tau_e is not None and self._use_vms:
                    value = tau_e * u * strong
                    if self.include_nonlinear_vms:
                        value -= 0.5 * (tau_e * strong) ** 2
                    residual[i] += scale * gi * value

                for j, bj in enumerate((b0, b1)):
                    gj = grad[j]
                    dstrong = a0 * inv_dt * bj + du * bj + u * gj
                    jac = (
                        a0 * inv_dt * bi * bj
                        + self.viscosity * gi * gj
                        + bi * (bj * du + u * gj)
                    )
                    if tau_e is not None and self._use_vms:
                        if self.include_nonlinear_vms:
                            dv = tau_e * (bj * strong + (u - tau_e * strong) * dstrong)
                        else:
                            dv = tau_e * (bj * strong + u * dstrong)
                        jac += gi * dv
                    jacobian[i, j] += scale * jac

            if getattr(self, "_record_local_diagnostics", False):
                self.strong_residual_history.append(
                    {
                        "step": self.current_solver_step,
                        "iteration": self.current_nr_iteration,
                        "element": int(element[0]),
                        "gauss_point": float(_GAUSS_POINTS[q]),
                        "x": float(element[0] * h + 0.5 * h * (_GAUSS_POINTS[q] + 1.0)),
                        "residual": float(strong),
                    }
                )
        return residual, jacobian

    def global_assembly(
        self,
        elemental_residuals: Sequence[NDArray],
        elemental_jacobians: Sequence[NDArray],
    ) -> tuple[NDArray, NDArray]:
        """Assemble elemental contributions into dense arrays (compatibility API)."""
        residual = np.zeros(self.n_nodes, dtype=float)
        jacobian = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for e, (left, right) in enumerate(self.elements):
            re, je = elemental_residuals[e], elemental_jacobians[e]
            residual[left] += re[0]
            residual[right] += re[1]
            jacobian[left, left] += je[0, 0]
            jacobian[left, right] += je[0, 1]
            jacobian[right, left] += je[1, 0]
            jacobian[right, right] += je[1, 1]
        return residual, jacobian

    def apply_homogeneous_bcs(
        self,
        global_residual: NDArray,
        global_jacobian: NDArray,
        solution_k: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Enforce Dirichlet conditions by row replacement (dense API)."""
        for node, value in zip(sorted(self.boundary_nodes), self.boundary_values):
            global_residual[node] = solution_k[node] - value
            global_jacobian[node, :] = 0.0
            global_jacobian[node, node] = 1.0
        return global_residual, global_jacobian

    # ------------------------------------------------------------------ #
    #  VMS Models
    # ------------------------------------------------------------------ #

    @staticmethod
    def vms_elemental_residual(
        i,
        gradient_basis: NDArray,
        f: dict[str, float],
        strong_res: float,
        tau_e: float,
        include_nonlinear: bool = True,
    ) -> float:
        """Elemental VMS residual with an optional quadratic fine-scale term."""
        u_k = f["u_k"]
        value = tau_e * u_k * strong_res
        if include_nonlinear:
            value -= 0.5 * (tau_e * strong_res) ** 2
        return gradient_basis[i] * value

    def compute_tau(
        self,
        u_e: NDArray,
        u_x_e: NDArray | None = None,
        external_coefficients: NDArray | None = None,
        record_components: bool = False,
    ) -> float:
        """Dispatch to the configured tau model. u_e: nodal values for the element (length 2)."""
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
        """2 Parameter tau model with possible inclusion of temporal stabilization, currently non-adaptable."""
        # 1. Validate u_e shape/length to prevent IndexError
        if u_e is None or len(u_e) < 2:
            raise ValueError(
                "Input 'u_e' must contain at least 2 elements for averaging."
            )
        # 2. Validate and standardize correction coefficients shape
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
        # 1. Validate u_e shape/length
        if u_e is None or len(u_e) < 2:
            raise ValueError(
                "Input 'u_e' must contain at least 2 elements for averaging."
            )

        # 2. Validate and standardize correction coefficients shape (expected: length 3)
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
        """Compute Reynolds stress <u'u'>, equivalent to variance of the velocity in this 1D context."""
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
        """Computes the time-averaged spatial mean profile over the stored solution history."""
        if self.snapshots_solution is None or len(self.snapshots_solution) == 0:
            raise ValueError(
                "No solution history found. Run the solver before computing statistics."
            )

        return np.mean(np.asarray(self.snapshots_solution), axis=0)

    @staticmethod
    def _compute_time_stepping(
        dt: float, timespan: float, rel_tol: float = 1e-9
    ) -> tuple[int, NDArray]:
        """Build the per-step dt array: fixed `dt` for every step, except the
        final step, which is shortened so the simulation lands exactly on
        t_start + timespan instead of overshooting it.

        Returns (n_time_steps, step_dts).
        """
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
            "tau_policy": "frozen",
            "domain_timespan": self.domain_timespan,
            "domain_length": self.domain_length,
            "dt": self.dt,
            "n_time_steps": self._n_time_steps,
            "viscosity": self.viscosity,
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "element_size": self.element_size,
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

        # --- Mesh ---
        _section("Mesh")
        _row("nodes", str(self.n_nodes))
        _row("elements", str(self.n_elements))
        _row("domain length", f"{self.domain_length:.4g}")
        _row("element size h", f"{self.element_size:.4e}")

        # --- Time ---
        _section("Time")
        _row("timespan", f"{self.domain_timespan:.4g}")
        _row("dt", f"{self.dt:.4e}")
        _row("total steps", str(self._n_time_steps))
        if self.requested_snapshots is not None:
            n_ext = len(self.requested_snapshots)
            first_five = "  ".join(f"{t:.4f}" for t in self.requested_snapshots[:5])
            _row("snapshots", str(n_ext))
            _row("  first 5 times", first_five)

        # --- Physics ---
        _section("Physics")
        _row("viscosity ν", f"{self.viscosity:.4e}")

        # --- Paths ---
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
        """Plot the final solution and diagnostics using the thesis plot style."""
        sgs_label = {
            "dns": "DNS",
            "les": "LES-VMS",
            "sgsp": "LES-SGSP",
            "avc": "LES-AVC",
        }.get(self.simulation_mode, self.simulation_mode)

        # Publication-style defaults, matching the training/profile plots.
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

        # Same restrained palette and open-marker treatment as the
        # DNS / projected-DNS / baseline / improved profile comparisons.
        style_dns = dict(
            color="black",
            linestyle="-",
            linewidth=1.6,
            zorder=3,
        )
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
        style_initial = dict(
            color="k",
            linestyle=":",
            linewidth=1.2,
            zorder=1,
        )
        style_energy = dict(
            color="#B9314F",
            linestyle="-",
            linewidth=1.5,
            alpha=0.9,
        )
        style_dissipation = dict(
            color="#47329A",
            linestyle="-",
            linewidth=1.5,
            alpha=0.9,
        )

        fig = plt.figure(figsize=(10, 6.2), dpi=300, constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0), hspace=0.05)
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])

        # Consistent axis treatment across all panels.
        for ax in (ax0, ax1, ax2):
            ax.minorticks_on()
            ax.tick_params(
                which="both",
                direction="in",
                top=True,
                right=True,
                length=4,
                width=0.8,
            )
            ax.tick_params(which="minor", length=2, width=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
            ax.grid(
                False,
            )

        # (a) Resolved solution and initial condition.
        resolved_style = style_dns if self.simulation_mode == "dns" else style_model
        ax0.plot(
            self.mesh,
            self.solution,
            label="Resolved solution",
            **resolved_style,
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
        ax0.set_title(
            f"(a) Resolved solution | {sgs_label}",
            loc="left",
            pad=6,
        )
        ax0.legend(
            loc="best",
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="0.8",
        )

        # (b) Energy and dissipation histories. Clip all histories to their
        # common available length so mismatched diagnostic arrays stay safe.
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

        # (c) Positive-wavenumber energy spectrum.
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

        # if self.tau_temporal_history:
        #     ax.plot(
        #         np.arange(len(self.tau_temporal_history)),
        #         self.tau_temporal_history,
        #         label=r"$\tau_{\mathrm{time}}$",
        #     )

        ax.set_xlabel("Time")
        ax.set_title("Evolution of Stabilization Parameters")

        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        filename = "tau_paramaters.png"
        output_path = self.master_path / filename
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig, ax
