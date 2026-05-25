"""1D FEM solver for the Burgers' equation.

Simulation modes
----------------
"dns"      - Pure Galerkin, no SGS model.
"no_model" - Same as "dns";
             To be used when solver settings are coarse (LES-like).
"les"    - Galerkin + analytic VMS/SGS stabilisation (tau-based).
"ann"    - Galerkin + ANN-predicted SGS corrections;
             analytic VMS terms are fully disabled.
             Requires "ann_model" or "ann_model_path"
             + "ann_model_class" at construction time.

Parameters
----------
"initial_condition"        - A callable to built the IC within the solver, or;
                             A numpy array given if pure stochastic data.

"boundary_condition_type": - "dirichlet": fixed values at boundaries. Value specified else 0.
                           - "periodic" : right boundary collapses onto the left

"external_forcing"         - A callable to built f withing the solver, or;
                             A numpy array given if pure stochastic data.
"""

import csv
import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from itertools import chain
from pathlib import Path
from time import perf_counter
from typing import Callable, Generator, Any, Iterable

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

from constants import TOLERANCE_RESIDUAL, TOLERANCE_UPDATE, MAXIMUM_ITERATIONS

logger = logging.getLogger(__name__)


class BurgersPure:
    """Burgers FEM solver: M·U_t + A(U)·U + ν·K₀·U + C_fs(U) = f.

    Parameters
    ----------
    configuration:
        Dict produced by :meth:`create_config`.
    """

    def __init__(
        self,
        configuration: dict,
    ) -> None:
        # ── solver configuration ────────────────────────────────────────
        self.configuration: dict = configuration
        self.domain_timespan: float = self.configuration["domain_timespan"]
        self.simulation_time_elapsed: float = 0.0
        self.domain_length: float = self.configuration["domain_length"]
        self.dt: float = self.configuration["time_step"]
        self.relaxation_factor: float | None = self.configuration["relax"]
        self.viscosity: float = self.configuration["viscosity"]
        self.simulation_mode: str | None = self.configuration["simulation_mode"]
        self.max_iterations: int = self.configuration["max_iterations"]

        _valid_types = {"dns", "no_model", "les", "ann"}
        if self.simulation_mode not in _valid_types:
            raise ValueError(
                f"Unknown simulation_type {self.simulation_mode!r}. "
                f"Expected one of {_valid_types}."
            )

        # ── benchmarking ────────────────────────────────────────────────
        self.run_id: str = datetime.now().strftime("%m%d_%H%M%S")
        self.timings_performance: dict = {}
        self.time_steps: list = []
        self.residual_history: list | None = []
        self.update_history: list | None = []
        self.energy_history: list = []
        self.dissipation_history: list = []

        # ── meshing ─────────────────────────────────────────────────────
        self.n_nodes: int = self.configuration["node_amount"]
        self.n_elements: int = self.n_nodes - 1
        self.nodes: NDArray = np.arange(0, self.n_nodes)
        self.boundary_nodes: set[int] = {int(self.nodes[0]), int(self.nodes[-1])}
        self.node_coords: NDArray = np.linspace(
            start=0, stop=self.domain_length, num=self.n_nodes
        )
        self.elements: NDArray = self.initialize_elements()
        self.element_size: float = self.domain_length / (self.n_nodes - 1)

        # ── boundary conditions ──────────────────────────────────────────
        _valid_bc_types = {"dirichlet", "fixed", "periodic"}
        self.boundary_condition_type: str = self.configuration[
            "boundary_condition_type"
        ]
        if self.boundary_condition_type not in _valid_bc_types:
            raise ValueError(
                f"Unknown boundary_condition_type {self.boundary_condition_type!r}. "
                f"Expected one of {_valid_bc_types}."
            )
        self.boundary_condition_value = self.configuration.get(
            "boundary_condition_value", 0.0
        )

        # ── IC ──────────────────────────────────────────────────────────
        self.initial_condition: NDArray = self.set_initial_condition(
            initial_condition=self.configuration["initial_condition"]
        )
        self.solution: NDArray = self.initial_condition.copy()
        self.forcing: NDArray | Callable[[NDArray, float], NDArray] | None = (
            self.configuration["external_forcing"]
            if self.configuration["external_forcing"] is not None
            else None
        )
        self.forcing_is_steady: bool = self.configuration["forcing_steady"]
        self.forcing_current: NDArray | None = None

        # ── output ──────────────────────────────────────────────────────
        self.write_solutions: bool = True
        self.extract_at_times: list | None = configuration["extract_at_times"]
        self.is_extracted_at_times: list[bool] | None = (
            [False for _ in self.extract_at_times]
            if self.extract_at_times is not None
            else None
        )
        self.extracted_solutions: list[NDArray] | None = None
        self.extracted_forcings: list[NDArray] | None = None

        self.master_path: Path | str | None = self.configuration["master_path"]
        self.master_path.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logger()

    # ------------------------------------------------------------------ #
    #  Configuration
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_config(
        initial_condition: NDArray | Callable,
        simulation_mode: str,
        node_amount: int,
        viscosity: float,
        time_step: float,
        domain_timespan: float,
        domain_length: float,
        boundary_condition_type: str,
        boundary_condition_value: float | NDArray | Callable | None = None,
        external_forcing: NDArray | Callable | None = None,
        forcing_steady: bool = True,
        run_objective: str = "N/A",
        convergence_tol_residual: float = TOLERANCE_RESIDUAL,
        convergence_tol_update: float = TOLERANCE_UPDATE,
        max_iterations: int = MAXIMUM_ITERATIONS,
        relaxation: float | None = None,
        extract_at_times: list | NDArray | None = None,
        master_path: str | Path | None = None,
    ) -> dict:
        """Build and return a configuration dictionary."""
        return {
            "simulation_mode": str(simulation_mode),
            "objective": str(run_objective),
            "boundary_condition_type": boundary_condition_type,
            "boundary_condition_value": boundary_condition_value,
            "extract_at_times": extract_at_times,
            "node_amount": node_amount,
            "domain_timespan": domain_timespan,
            "time_step": time_step,
            "domain_length": domain_length,
            "convergence_tol_residual": convergence_tol_residual,
            "convergence_tol_update": convergence_tol_update,
            "initial_condition": initial_condition,
            "external_forcing": external_forcing,
            "forcing_steady": forcing_steady,
            "max_iterations": max_iterations,
            "relax": relaxation,
            "viscosity": viscosity,
            "master_path": master_path,
        }

    # ------------------------------------------------------------------ #
    #  Main Solver Pipeline
    # ------------------------------------------------------------------ #

    @staticmethod
    def throbber(time_step: int, _every: int = 2) -> str:
        """Generates a throbber that cycles its state based on the time step."""
        states = ["nom..        ", "nom nom..    ", "nom nom nom.."]
        frame_index = (time_step // _every) % len(states)
        return states[frame_index]

    def run_simulation(self) -> None:
        """Run the full simulation and write output."""
        total_steps = int(self.domain_timespan / self.dt)
        self.extracted_solutions = []
        self.extracted_forcings = []
        idx_extract = 0

        with self.timer("total_simulation"):
            with tqdm(
                total=total_steps,
                desc=f"Eating Burgers | {self.throbber(time_step=0)}",
                file=sys.stdout,
            ) as pbar:
                for time_step in range(total_steps):
                    step_start = perf_counter()
                    self.time_steps.append(time_step)

                    # Advance Time Step
                    # -----------------
                    self.advance_time_step()
                    # -----------------

                    idx_extract = self._maybe_extract_solution(idx_extract)

                    step_time = perf_counter() - step_start
                    pbar.set_description(f"Eating Burgers | {self.throbber(time_step)}")
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "t": f"{self.simulation_time_elapsed:.3f}",
                            "dt": f"{self.dt:.3f}",
                            "step_time": f"{step_time:.3f}s",
                        }
                    )

            # Flush any extraction checkpoints not yet triggered.
            if self.extract_at_times is not None:
                while idx_extract < len(self.extract_at_times):
                    self.extracted_solutions.append(self.solution.copy())
                    self.extracted_forcings.append(
                        self.forcing_current.copy()
                        if self.forcing_current is not None
                        else np.zeros_like(self.solution)
                    )
                    logger.info(
                        "Extracted solution at t=%.4f (end-of-simulation flush)",
                        self.extract_at_times[idx_extract],
                    )
                    idx_extract += 1

        if self.write_solutions:
            self.write_config_to_json()
            self.write_solution_to_csv()

    def advance_time_step(self) -> None:
        """Advance the solution by one time-step: U^{n+1} ← U^n."""
        if callable(self.forcing):  # built from function
            self.forcing_current = (
                self.forcing(self.node_coords, self.simulation_time_elapsed)
                if not self.forcing_is_steady
                else self.forcing(self.node_coords)
            )
        elif self.forcing is None:  # no forcing
            self.forcing_current = np.zeros_like(self.solution)
        else:
            self.forcing_current = self.forcing  # data array

        self.solution = self.nr_iteration(solution=self.solution)
        self.energy_history.append(self.compute_energy(self.solution))
        self.dissipation_history.append(self.compute_dissipation(self.solution))
        self.simulation_time_elapsed += self.dt

    def nr_iteration(self, solution: NDArray) -> NDArray:
        """Newton–Raphson iteration: returns U^{n+1}."""
        solution_n = solution.copy()  # U^n  — frozen reference
        solution_k = solution.copy()  # U^k, updated each NR step
        residual_history_loop: list = []
        update_history_loop: list = []

        for _ in range(self.max_iterations):
            # ── Element loop ──
            with self.timer("elemental_iterations"):
                elemental_residuals, elemental_jacobians = zip(
                    *(
                        self.calculate_elemental_residual_jacobian(
                            element=element,
                            u_k=solution_k[element],
                            u_n=solution_n[element],
                            f_e=(
                                self.forcing_current[element]
                                if self.forcing_current is not None
                                else None
                            ),
                        )
                        for element in self.elements
                    )
                )

            # ── Global assembly ──
            with self.timer("global_assembly"):
                global_residual, global_jacobian = self.global_assembly(
                    elemental_residuals, elemental_jacobians
                )

            # ── Boundary conditions ──
            global_residual, global_jacobian = self._apply_boundary_conditions(
                global_residual, global_jacobian, solution_k
            )

            residual_history_loop.append(np.linalg.norm(global_residual))

            # ── Linear solve ──
            with self.timer("linear_solve"):
                delta_u = np.linalg.solve(global_jacobian, -global_residual)
                if self.boundary_condition_type == "periodic":
                    delta_u_reduced = delta_u.copy()
                    delta_u = np.zeros_like(solution_k)
                    delta_u[:-1] = delta_u_reduced
                    delta_u[-1] = delta_u[0]

            update_history_loop.append(np.linalg.norm(delta_u))

            # ── Solution update ──
            with self.timer("solution_update"):
                solution_k += (
                    delta_u * (1 - self.relaxation_factor)
                    if self.relaxation_factor is not None
                    else delta_u
                )

            # ── Convergence checks ──
            with self.timer("convergence_checking"):
                if self.is_update_converged(correction=delta_u):
                    break
            with self.timer("convergence_checking"):
                if self.is_residual_converged(residual=global_residual):
                    break

        self.residual_history.append(residual_history_loop)
        self.update_history.append(update_history_loop)
        return solution_k

    def calculate_elemental_residual_jacobian(
        self,
        element: tuple[int, int],
        u_k: NDArray,
        u_n: NDArray,
        f_e: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Compute R_e ∈ ℝ² and J_e ∈ ℝ²ˣ² for one element."""
        residual_element = np.zeros(2)
        jacobian_element = np.zeros((2, 2))

        weight = abs(self.element_size / 2)
        points, weights = self.gauss_legendre(3)
        gradient_basis = self.reference_gradient_basis_functions()

        for g_p, g_w in zip(points, weights):
            basis = self.reference_basis_functions(g_p)
            interp_fields = self._interpolate_fields(basis, gradient_basis, u_k, u_n)
            midpoint_fields = self._midpoint_fields(interp_fields)
            strong_res = self._strong_residual(interp_fields)
            scale = g_w * weight
            f_interp = float(basis @ f_e) if f_e is not None else 0.0

            for i in range(len(element)):
                residual_element[i] += scale * self._residual_integrand(
                    i, basis, gradient_basis, interp_fields, midpoint_fields, f_interp
                )
                if self._use_vms:
                    residual_element[i] += scale * self._vms_residual_integrand(
                        i, gradient_basis, interp_fields
                    )

                for j in range(len(element)):
                    jacobian_element[i, j] += scale * self._jacobian_integrand(
                        i, j, basis, gradient_basis, interp_fields
                    )
                    if self._use_vms:
                        jacobian_element[i, j] += scale * self._vms_jacobian_integrand(
                            i, j, basis, gradient_basis, interp_fields, strong_res
                        )

        return residual_element, jacobian_element

    # ------------------------------------------------------------------ #
    #  Boundary conditions
    # ------------------------------------------------------------------ #

    def _apply_boundary_conditions(
        self,
        global_residual: NDArray,
        global_jacobian: NDArray,
        solution_k: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Dispatch to the correct BC enforcement based on ``boundary_condition_type``."""
        if self.boundary_condition_type in ("dirichlet", "fixed"):
            return self._apply_dirichlet_bcs(
                global_residual, global_jacobian, solution_k
            )
        else:  # "periodic" — validated at construction
            return self._apply_periodic_bcs(global_residual, global_jacobian)

    def _apply_dirichlet_bcs(
        self,
        global_residual: NDArray,
        global_jacobian: NDArray,
        solution_k: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Enforce Dirichlet BCs via row-replacement.

        The target value at each boundary node is obtained by evaluating
        ``self.boundary_condition_value`` — a callable produced from
        whatever the caller passed (scalar, array, or function) — at the
        physical coordinate of that node.  This mirrors exactly how
        ``initial_condition`` and ``forcing`` are handled.
        """
        for node in self.boundary_nodes:
            target = self.boundary_condition_value
            global_residual[node] = solution_k[node] - target
            global_jacobian[node, :] = 0
            global_jacobian[node, node] = 1
        return global_residual, global_jacobian

    @staticmethod
    def _apply_periodic_bcs(
        global_residual: NDArray,
        global_jacobian: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Enforce periodic BCs by folding the last DOF into the first."""
        global_residual[0] += global_residual[-1]
        global_jacobian[0, :] += global_jacobian[-1, :]
        global_jacobian[:, 0] += global_jacobian[:, -1]
        return global_residual[:-1], global_jacobian[:-1, :-1]

    # ------------------------------------------------------------------ #
    #  Global assembly
    # ------------------------------------------------------------------ #

    def global_assembly(
        self,
        elemental_residuals: Iterable[NDArray],
        elemental_jacobians: list[NDArray],
    ) -> tuple[NDArray, NDArray]:
        """Assemble element contributions into global R and J."""
        global_residual = np.zeros(self.n_nodes)
        global_jacobian = np.zeros((self.n_nodes, self.n_nodes))

        elemental_residuals = list(elemental_residuals)

        for e, element in enumerate(self.elements):
            i, j = element
            global_residual[i] += elemental_residuals[e][0]
            global_residual[j] += elemental_residuals[e][1]
            global_jacobian[i, i] += elemental_jacobians[e][0, 0]
            global_jacobian[i, j] += elemental_jacobians[e][0, 1]
            global_jacobian[j, i] += elemental_jacobians[e][1, 0]
            global_jacobian[j, j] += elemental_jacobians[e][1, 1]

        return global_residual, global_jacobian

    # ------------------------------------------------------------------ #
    #  Analytical SGS model
    # ------------------------------------------------------------------ #

    def compute_tau(self, variable_u: float | NDArray) -> float:
        """VMS stabilisation parameter τ."""
        term_time = (2 / self.dt) ** 2
        term_adv = (2 * abs(variable_u) / self.element_size) ** 2
        term_diff = (4 * self.viscosity / self.element_size**2) ** 2
        return 0.5 / np.sqrt(term_time + term_adv + term_diff)

    # ------------------------------------------------------------------ #
    #  Convergence NR
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_residual_converged(
        residual: float | NDArray, tolerance: float = 1e-6
    ) -> bool:
        """Return ``True`` when the residual norm is below tolerance."""
        norm = np.linalg.norm(residual)
        return norm < tolerance * (1 + np.linalg.norm(residual))

    @staticmethod
    def is_update_converged(correction: NDArray, tolerance: float = 1e-6) -> bool:
        """Return ``True`` when the update norm is below tolerance."""
        return np.linalg.norm(correction) < tolerance

    @staticmethod
    def _interpolate_fields(
        basis: NDArray,
        gradient_basis: NDArray,
        u_k: NDArray,
        u_n: NDArray,
    ) -> dict:
        return {
            "u_k": basis @ u_k,
            "u_n": basis @ u_n,
            "du_k": gradient_basis @ u_k,
            "du_n": gradient_basis @ u_n,
        }

    @staticmethod
    def _midpoint_fields(f: dict[str, float]) -> dict[str, float]:
        """Crank–Nicolson midpoint averages."""
        return {
            "u_mid": 0.5 * (f["u_k"] + f["u_n"]),
            "du_mid": 0.5 * (f["du_k"] + f["du_n"]),
        }

    def _strong_residual(self, f: dict[str, float]) -> float:
        u_mid = 0.5 * (f["u_k"] + f["u_n"])
        du_mid = 0.5 * (f["du_k"] + f["du_n"])
        return (f["u_k"] - f["u_n"]) / self.dt + u_mid * du_mid

    def _residual_integrand(
        self,
        i: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
        mid: dict[str, float],
        f_interp: float = 0.0,
    ) -> float:
        time_derivative = basis[i] * (f["u_k"] - f["u_n"]) / self.dt
        diffusion = self.viscosity * mid["du_mid"] * gradient_basis[i]
        advection = basis[i] * mid["u_mid"] * mid["du_mid"]
        forcing = basis[i] * f_interp
        return time_derivative + diffusion + advection - forcing

    def _vms_residual_integrand(
        self,
        i: int,
        gradient_basis: NDArray,
        f: dict[str, float],
    ) -> float:
        u_mid = 0.5 * (f["u_k"] + f["u_n"])
        du_mid = 0.5 * (f["du_k"] + f["du_n"])
        tau_mid = self.compute_tau(u_mid)
        return (u_mid * gradient_basis[i]) * tau_mid * u_mid * du_mid

    def _jacobian_integrand(
        self,
        i: int,
        j: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
    ) -> float:
        mass = basis[i] * basis[j] / self.dt
        stiffness = self.viscosity * gradient_basis[i] * gradient_basis[j]
        advection = basis[i] * (basis[j] * f["du_k"] + f["u_k"] * gradient_basis[j])
        return mass + 0.5 * (stiffness + advection)

    def _vms_jacobian_integrand(
        self,
        i: int,
        j: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
        strong_res: float,
    ) -> float:
        u_mid = 0.5 * (f["u_k"] + f["u_n"])
        du_mid = 0.5 * (f["du_k"] + f["du_n"])
        tau_mid = self.compute_tau(u_mid)
        spatial_part = basis[j] * du_mid + u_mid * gradient_basis[j]
        tau_part = basis[j] * gradient_basis[i] * tau_mid * strong_res
        return spatial_part + tau_part

    # ------------------------------------------------------------------ #
    #  Initialization
    # ------------------------------------------------------------------ #

    def initialize_elements(self) -> NDArray:
        """Build element connectivity array ``[[0,1], [1,2], …]``."""
        idx = self.nodes
        return np.column_stack((idx[:-1], idx[1:]))

    def set_initial_condition(self, initial_condition: NDArray | Callable) -> NDArray:
        """Set the initial condition from an array or a callable."""
        if isinstance(initial_condition, Callable):
            return initial_condition(self.node_coords)

        return initial_condition.copy()

    def initial_solution_is_valid(self) -> None:
        """Raise if the initial solution has wrong size or is missing."""
        print("=" * 60)
        print("Checking initial solution...")
        print("-" * 60)
        if self.solution is None:
            raise ValueError("No initial solution given.")
        if len(self.solution) != self.n_nodes:
            raise ValueError(f"Initial solution of invalid size: {len(self.solution)}")
        print("Initial solution valid.")
        print("=" * 60)

    # ------------------------------------------------------------------ #
    #  FEM Primitives
    # ------------------------------------------------------------------ #

    @staticmethod
    def gauss_legendre(number_of_points: int) -> tuple[Any, Any]:
        """Return Gauss–Legendre quadrature points and weights."""
        return np.polynomial.legendre.leggauss(deg=number_of_points)

    @staticmethod
    def reference_basis_functions(ksi: float) -> NDArray:
        """Linear basis functions on the reference element."""
        return np.array([0.5 * (1 - ksi), 0.5 * (1 + ksi)])

    def reference_gradient_basis_functions(self) -> NDArray:
        """Gradient of linear basis functions mapped to physical space."""
        return np.array([-0.5, 0.5]) * (2 / self.element_size)

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #
    @property
    def _use_vms(self) -> bool:
        """``True`` only for the analytic VMS mode (``les``)."""
        return self.simulation_mode == "les"

    @property
    def _use_ann(self) -> bool:
        """``True`` only for the ANN-coupled LES mode (``ann``)."""
        return self.simulation_mode == "ann"

    @property
    def total_convergence_history(self) -> tuple[NDArray, NDArray]:
        """Total residual and update norm history across all time steps."""
        residual_history = np.array(list(chain.from_iterable(self.residual_history)))
        update_history = np.array(list(chain.from_iterable(self.update_history)))
        return residual_history, update_history

    # ------------------------------------------------------------------ #
    #  Timing
    # ------------------------------------------------------------------ #

    @contextmanager
    def timer(self, name: str) -> Generator[None, Any, None]:
        """Accumulate wall-clock time for a named phase."""
        start = perf_counter()
        yield
        self.timings_performance[name] = (
            self.timings_performance.get(name, 0.0) + perf_counter() - start
        )

    # ------------------------------------------------------------------ #
    #  Logging / Post-Processing
    # ------------------------------------------------------------------ #
    def write_solution_to_csv(self) -> None:
        """Write solution snapshot(s) to CSV files."""
        nodes = self.nodes
        coordinates = self.node_coords
        run_dir = self.master_path

        if self.extract_at_times is None:
            solutions = [self.solution]
            forcings = [self.forcing_current]
            times = [self.simulation_time_elapsed]
        else:
            solutions = self.extracted_solutions
            forcings = self.extracted_forcings
            times = self.extract_at_times[: len(solutions)]

        write_count = 0
        for solution, time, forcing in zip(solutions, times, forcings):
            filepath = run_dir / f"sol_t{time:.6f}.csv"
            with open(filepath, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["node_index", "x_coordinate", "velocity", "forcing"])
                for i in range(len(nodes)):
                    writer.writerow([nodes[i], coordinates[i], solution[i], forcing[i]])
            write_count += 1

        print(f"wrote {write_count} snapshots at {run_dir}")

    def post_processing(self) -> None:
        """Run post-plotting and post-logging."""
        self.post_plotting()
        self.post_logging()

    def _setup_logger(self) -> logging.Logger:
        logger_ = logging.getLogger(str(self.run_id))
        logger_.setLevel(logging.INFO)
        if logger_.handlers:
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
        """Write a structured summary to the run log."""
        self.logger.info("=" * 60)
        self.logger.info("START OUTPUT")
        self.logger.info(
            "Run started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        self.logger.info("RUN COMPLETE — id: %s", self.run_id)
        self.logger.info("=" * 60)

        config_loggable = {}
        for k, v in self.configuration.items():
            if k in ("solution_initial", "master_path", "initial_condition"):
                continue
            elif k == "extract_at_times":
                if v is not None:
                    config_loggable[k] = (
                        f"{len(v)} extractions, "
                        f"first 5: {[float(t) for t in v[:5]]}"
                    )
                else:
                    config_loggable[k] = "None"
            elif k in ("convergence_tol_residual", "convergence_tol_update"):
                config_loggable[k] = f"{v:.2e}"
            elif k == "external_forcing":
                if callable(v):
                    module = getattr(v, "__module__", "")
                    config_loggable[k] = f"{v.__name__} (from {module})"
                elif v is None:
                    config_loggable[k] = "None"
                else:
                    config_loggable[k] = f"array, shape {np.array(v).shape}"
            elif isinstance(v, float):
                config_loggable[k] = f"{v:.6g}"
            else:
                config_loggable[k] = v

        self.logger.info("Configuration settings:")
        self.logger.info("  Time Integration: Second Order Implicit Euler")
        self.logger.info("  Simulation mode   : %s", self.simulation_mode)
        self.logger.info("  " + "-" * 40)
        for k, v in config_loggable.items():
            self.logger.info("  %-30s %s", k, v)
        self.logger.info("  " + "-" * 40)

        if self.timings_performance:
            self.logger.info("Timings:")
            total = self.timings_performance.get("total_simulation")
            if total is None:
                self.logger.warning("total_simulation timing not recorded (run may have been interrupted).")
                total = sum(
                    v for k, v in self.timings_performance.items()
                    if k != "total_simulation"
                )
            for phase, t in sorted(self.timings_performance.items()):
                if phase != "total_simulation":
                    pct = (100 * t / total) if total > 0 else float("nan")
                    self.logger.info("  %-25s %.4fs (%5.1f%%)", phase, t, pct)
            self.logger.info("  %-25s %.4fs", "TOTAL", total)
        else:
            self.logger.warning("No timing data recorded.")

        if self.residual_history:
            res, upd = self.total_convergence_history
            self.logger.info("Convergence summary:")
            self.logger.info(
                "  Residual  — initial: %.4e  final: %.4e  max: %.4e",
                res[0],
                res[-1],
                np.max(res),
            )
            if len(upd) > 0:
                self.logger.info(
                    "  Update    — initial: %.4e  final: %.4e  max: %.4e",
                    upd[0],
                    upd[-1],
                    np.max(upd),
                )
            tol = self.configuration.get("convergence_tol_residual", 1e-6)
            if res[-1] < tol:
                self.logger.info("  Status    — CONVERGED")
            else:
                self.logger.warning(
                    "  Status    — NOT CONVERGED (final %.4e > tol %.4e)", res[-1], tol
                )
            self.logger.info("  Total NR iterations: %d", len(res))
        else:
            self.logger.warning("No convergence history recorded.")

        self.logger.info("END OUTPUT")
        self.logger.info("=" * 60)

    def write_config_to_json(self) -> None:
        """Write run configuration to ``config.json`` in the run directory."""
        config_serializable = {}
        for k, v in self.configuration.items():
            if k in ("solution_initial", "master_path"):
                continue
            elif isinstance(v, np.ndarray):
                config_serializable[k] = v.tolist()
            elif callable(v):
                config_serializable[k] = (
                    f"<callable: {getattr(v, '__name__', repr(v))}>"
                )
            else:
                config_serializable[k] = v
        config_serializable["run_id"] = self.run_id
        with open(self.master_path / "config.json", "w") as f:
            json.dump(config_serializable, f, indent=2)

    def print_configuration(self) -> None:
        """Print run configuration to stdout."""
        print("=" * 60)
        print("Configuration settings")
        print("Time Integration: Second Order Implicit Euler")
        print(f"Simulation mode: {self.simulation_mode}")
        print("=" * 60)
        for k, v in self.configuration.items():
            if k == "simulation_mode":  # already printed above
                continue
            if k == "extract_at_times":
                if v is not None:
                    print(f"extract_at_times: {len(v)} extractions, "
                          f"first 5 steps: {[float(t) for t in v[:5]]}")
                else:
                    print("extract_at_times: None")
            elif k == "convergence_tol_residual" or k == "convergence_tol_update":
                print(f"{k}: {v:.2e}")
            elif k == "external_forcing":
                if callable(v):
                    module = getattr(v, '__module__', '')
                    print(f"external_forcing: {v.__name__} (from {module})")
                elif v is None:
                    print("external_forcing: None")
                else:
                    print(f"external_forcing: array, shape {np.array(v).shape}")
            elif k not in ("initial_condition",):
                print(f"{k}: {round(v, 4) if isinstance(v, float) else v}")
        print("=" * 60)

    @staticmethod
    def moving_stats(arr: list, window: int = 5) -> tuple[NDArray, NDArray]:
        """Compute rolling mean and std for convergence plots."""
        arr = np.array(arr)
        if arr.size == 0:
            return np.array([]), np.array([])
        w = min(window, len(arr))
        means = np.convolve(arr, np.ones(w) / w, mode="same")
        stds = np.array(
            [
                np.std(arr[max(0, i - w // 2) : min(len(arr), i + w // 2 + 1)])
                for i in range(len(arr))
            ]
        )
        return means, stds

    def post_plotting(self, show_plot: bool = False) -> None:
        """Plot solution and convergence diagnostics, save to disk, and optionally display."""
        first_res = [r[0] for r in self.residual_history if r]
        last_res = [r[-1] for r in self.residual_history if r]
        first_upd = [u[0] for u in self.update_history if u]
        last_upd = [u[-1] for u in self.update_history if u]
        fr_mean, fr_std = self.moving_stats(first_res)
        lr_mean, lr_std = self.moving_stats(last_res)
        fu_mean, fu_std = self.moving_stats(first_upd)
        lu_mean, lu_std = self.moving_stats(last_upd)

        sgs_label = {"dns": "DNS", "les": "LES-VMS", "ann": "LES-ANN"}.get(
            self.simulation_mode, self.simulation_mode
        )

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(3, 2)

        # Solution
        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(
            self.node_coords,
            self.solution,
            color="royalblue",
            linestyle="-",
            marker="o",
            label="Resolved solution",
        )
        ax0.plot(
            self.node_coords,
            self.initial_condition,
            color="grey",
            linestyle="--",
            label="Initial solution",
        )
        ax0.set_xlabel(r"$x \in [0, 2\pi]$")
        ax0.set_ylabel("Velocity")
        ax0.grid(True)
        ax0.legend()
        ax0.set_title(f"Solution  [SGS: {sgs_label}]")

        # Global convergence
        ax1 = fig.add_subplot(gs[1, 0])
        t = np.arange(len(fr_mean))
        ax1.plot(t, fr_mean, color="royalblue", label="Residual (first)")
        ax1.fill_between(
            t, fr_mean - fr_std, fr_mean + fr_std, color="royalblue", alpha=0.15
        )
        ax1.plot(t, lr_mean, color="navy", linestyle="--", label="Residual (last)")
        ax1.fill_between(
            t, lr_mean - lr_std, lr_mean + lr_std, color="navy", alpha=0.15
        )
        ax1.plot(t, fu_mean, color="tab:orange", label="Update (first)")
        ax1.fill_between(
            t, fu_mean - fu_std, fu_mean + fu_std, color="tab:orange", alpha=0.15
        )
        ax1.plot(t, lu_mean, color="darkorange", linestyle="--", label="Update (last)")
        ax1.fill_between(
            t, lu_mean - lu_std, lu_mean + lu_std, color="darkorange", alpha=0.15
        )
        tol_r = self.configuration["convergence_tol_residual"]
        tol_u = self.configuration["convergence_tol_update"]
        if tol_r != tol_u:
            ax1.axhline(y=tol_r, color="lightskyblue", linestyle="--")
            ax1.axhline(y=tol_u, color="lightsalmon", linestyle="--")
        else:
            ax1.axhline(y=tol_r, color="lightgray", linestyle="--")
        ax1.set_yscale("log")
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Norm")
        ax1.set_title("Global convergence (smoothed)")
        ax1.grid(True)
        ax1.legend()

        # Last NR iteration
        ax2 = fig.add_subplot(gs[1, 1])
        if self.residual_history:
            ax2.plot(
                self.residual_history[-1], "o-", label="Residual", color="royalblue"
            )
            if self.update_history[-1]:
                ax2.plot(
                    self.update_history[-1], "x--", label="Update", color="tab:orange"
                )
        ax2.set_yscale("log")
        ax2.set_xlabel("Newton iteration")
        ax2.set_ylabel("Norm")
        ax2.set_title("Last Newton iteration convergence")
        ax2.grid(True)
        ax2.legend()

        # Energy & dissipation
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(
            self.time_steps, self.energy_history, color="red", label="Total energy"
        )
        ax3.plot(
            self.time_steps,
            self.dissipation_history,
            color="purple",
            label="Dissipation",
        )
        ax3.set_xlabel("Time step")
        ax3.set_title("Energy and dissipation evolution")
        ax3.grid(True)
        ax3.legend()

        # Spectral analysis
        ax4 = fig.add_subplot(gs[2, 1])
        wn, sp = self.get_positive_spectrum(
            *self.compute_energy_spectrum(self.solution)
        )
        ax4.loglog(wn[1:], sp[1:], marker="o")
        ax4.set_xlabel("Wavenumber k")
        ax4.set_ylabel("E(k)")
        ax4.set_title("Spectral Analysis (at end)")
        ax4.grid(True)

        plt.tight_layout()

        # --- Saving Routine ---
        if hasattr(self, "master_path") and self.master_path:
            # Ensure the directory exists
            save_dir = Path(self.master_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Define file name (dynamically names it based on the simulation mode)
            file_path = save_dir / f"post_plotting_{self.simulation_mode}.png"

            # Save the figure (dpi=300 keeps it crisp for reports/papers)
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            print(f"Post-simulation plot saved to: {file_path}")

        # --- Showing / Closing Routine ---
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def compute_energy(self, solution: NDArray) -> float:
        """Integrate ½u² over the domain."""
        energy = 0.0
        jacobian = self.element_size / 2
        points, weights = self.gauss_legendre(2)
        for element in self.elements:
            u_e = solution[element]
            for g_p, g_w in zip(points, weights):
                basis = self.reference_basis_functions(g_p)
                energy += 0.5 * g_w * abs(jacobian) * (basis @ u_e) ** 2
        return energy

    def compute_dissipation(self, solution: NDArray) -> float:
        """Integrate ν(∂u/∂x)² over the domain."""
        dissipation = 0.0
        jacobian = self.element_size / 2
        points, weights = self.gauss_legendre(2)
        dn_dx = self.reference_gradient_basis_functions()
        for element in self.elements:
            u_e = solution[element]
            for g_p, g_w in zip(points, weights):
                du_dx = dn_dx @ u_e
                dissipation += self.viscosity * g_w * abs(jacobian) * du_dx**2
        return dissipation

    @staticmethod
    def get_positive_spectrum(
        wavenumbers: NDArray, spectrum: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Keep only non-negative wavenumbers."""
        mask = wavenumbers >= 0
        return wavenumbers[mask], spectrum[mask]

    def compute_energy_spectrum(self, solution: NDArray) -> tuple[NDArray, NDArray]:
        """Return wavenumbers and spectral energy of the given solution."""
        u_hat = np.fft.fft(solution)
        wavenumbers = (
            np.fft.fftfreq(len(solution), d=self.domain_length / len(solution))
            * 2
            * np.pi
        )
        spectrum = 0.5 * np.abs(u_hat) ** 2 / len(solution)
        return wavenumbers, spectrum

    def _maybe_extract_solution(self, idx_extract: int) -> int:
        if self.extract_at_times is None:
            return idx_extract
        if (
            idx_extract < len(self.extract_at_times)
            and self.simulation_time_elapsed >= self.extract_at_times[idx_extract]
        ):
            self.extracted_solutions.append(self.solution.copy())
            self.extracted_forcings.append(
                self.forcing_current.copy()
                if self.forcing_current is not None
                else np.zeros_like(self.solution)
            )
            idx_extract += 1
        return idx_extract
