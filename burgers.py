"""Class for 1D FEM solver based on the Burgers' equation."""

import csv
import json
import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from itertools import chain, cycle
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Generator, Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from constants import (
    MAXIMUM_ITERATIONS,
    TOLERANCE_RESIDUAL,
    TOLERANCE_UPDATE,
)

logger = logging.getLogger(__name__)


class Burgers:
    """Burgers class solver to simulate the discrete system: M * U_t + A(U) * U + nu * K_0 * U + C_fs(U) = f."""

    def __init__(
        self,
        configuration: dict,
    ) -> None:
        # ========================================================================== #
        # -------------------------- solver configuration -------------------------- #
        self.configuration: dict = configuration
        self.domain_timespan: float = self.configuration["domain_timespan"]
        self.simulation_time_elapsed: float = 0
        self.domain_length: float = self.configuration["domain_length"]
        self.dt: float = self.configuration["time_step"]
        self.relaxation_factor: float | None = self.configuration["relax"]
        self.viscosity: float = self.configuration["viscosity"]
        self.simulation_type: str | None = self.configuration["simulation_type"]
        self.max_iterations: int = self.configuration["max_iterations"]
        # --------------------------     benchmarking     -------------------------- #
        self.run_id: str = datetime.now().strftime("%m%d_%H%M%S")
        self.timings_performance: dict = {}
        self.time_steps: list = []
        self.residual_history: list | None = []
        self.update_history: list | None = []
        self.energy_history: list = []
        self.dissipation_history: list = []
        # --------------------------       meshing        -------------------------- #
        self.n_nodes: int = self.configuration["node_amount"]
        self.n_elements: int = self.n_nodes - 1
        self.nodes: NDArray = np.arange(0, self.n_nodes)
        self.boundary_nodes: set[int] = {int(self.nodes[0]), int(self.nodes[-1])}
        self.node_cords: NDArray = np.linspace(
            start=0, stop=self.domain_length, num=self.n_nodes
        )

        self.elements: NDArray = self.initialize_elements()
        self.element_size: float = self.domain_length / (
            self.n_nodes - 1
        )  #  Assuming linear mesh
        self.mesh: tuple[NDArray, NDArray] = (self.node_cords, self.elements)
        # --------------------------        output        -------------------------- #
        self.write_solutions: bool = True
        self.solution: NDArray = self.configuration["initial_condition"].copy()
        self.initial_condition: NDArray | None = self.configuration[
            "initial_condition"
        ].copy()  #  for plotting
        self.forcing: NDArray | Callable[[NDArray, float], NDArray] | None = (
            self.configuration["forcing"]
            if self.configuration["forcing"] is not None
            else None
        )
        self.forcing_current: NDArray | None = None
        self.boundary_conditions: str = self.configuration["boundary_conditions"]
        self.time_extractions: list | None = configuration["time_extractions"]
        self.is_extracted_at_times: list | bool | None = (
            [False for _ in self.time_extractions]
            if self.time_extractions is not None
            else None
        )
        self.extracted_solutions: list[NDArray] | None = None
        self.extracted_forcings: list[NDArray] | None = None
        self.run_dir: Path | str = (
            Path(__file__).resolve().parent
            / "runs"
            / f"run_{str(self.configuration['simulation_type'])}_n{self.n_nodes}_{self.run_id}"
            / "solver_data"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        # ========================================================================== #

    @staticmethod
    def create_config(
        initial_condition: NDArray,
        simulation_type: str,
        node_amount: int,
        viscosity: float,
        time_step: float,
        domain_timespan: float,
        domain_length: float,
        forcing: NDArray | str | None = None,
        run_objective: str = "standard",
        boundary_conditions: str = "fixed",
        convergence_tol_residual: float = TOLERANCE_RESIDUAL,
        convergence_tol_update: float = TOLERANCE_UPDATE,
        max_iterations: int = MAXIMUM_ITERATIONS,
        relaxation: float | None = None,
        time_extractions: list | NDArray | None = None,
    ) -> dict:
        """Create configuration dictionary."""
        return {
            "simulation_type": str(simulation_type),
            "objective": str(run_objective),
            "boundary_conditions": boundary_conditions,
            "time_extractions": time_extractions,
            "node_amount": node_amount,
            "domain_timespan": domain_timespan,
            "time_step": time_step,
            "domain_length": domain_length,
            "convergence_tol_residual": convergence_tol_residual,
            "convergence_tol_update": convergence_tol_update,
            "initial_condition": initial_condition,
            "forcing": forcing,
            "max_iterations": max_iterations,
            "relax": relaxation,
            "viscosity": viscosity,
        }

    @property
    def total_convergence_history(self) -> tuple[NDArray, NDArray]:
        """Provides the total history of convergence parameters."""
        residual_history = np.array(list(chain.from_iterable(self.residual_history)))
        update_history = np.array(list(chain.from_iterable(self.update_history)))
        return residual_history, update_history

    @contextmanager
    def timer(self, name: str) -> Generator[None, Any, None]:
        """Functionality to time the program."""
        start = perf_counter()
        yield
        elapsed = perf_counter() - start
        self.timings_performance[name] = (
            self.timings_performance.get(name, 0.0) + elapsed
        )

    @staticmethod
    def gauss_legendre(number_of_points: int) -> tuple[Any, Any]:
        """Provides the Gaussian points and weights."""
        return np.polynomial.legendre.leggauss(deg=number_of_points)

    @staticmethod
    def reference_basis_functions(ksi: float) -> NDArray:
        """Basis functions in reference domain."""
        return np.array([0.5 * (1 - ksi), 0.5 * (1 + ksi)])

    def reference_gradient_basis_functions(self) -> NDArray:
        """Gradient of basis functions in reference domain."""
        return np.array([-0.5, 0.5]) * (2 / self.element_size)

    def initialize_elements(self) -> NDArray:
        """Initialize array of elements using nodes [i, i+1]."""
        idx = self.nodes
        return np.column_stack((idx[:-1], idx[1:]))

    def initial_solution_is_valid(self) -> None:
        """Checks if the given initial solution is of valid form."""
        print("=" * 60)
        print("Checking initial solution...")
        print("-" * 60)
        if self.solution is None:
            raise ValueError("No initial solution given.")
        elif len(self.solution) != self.n_nodes:
            raise ValueError(f"Initial solution of invalid size: {len(self.solution)}")
        else:
            print("Initial solution valid.")
        print("=" * 60)

    def run_simulation(self) -> None:
        """Runs the simulation."""
        total_steps = int(self.domain_timespan / self.dt)

        throbber = cycle(["nom..        ", "nom nom..    ", "nom nom nom.."])
        throbber_every = 2

        self.extracted_solutions = []
        self.extracted_forcings = []
        idx_extract = 0

        throbber_state = next(throbber)

        with self.timer("total_simulation"):
            with tqdm(
                total=total_steps,
                desc=f"Eating Burgers | {throbber_state}",
                file=sys.stdout,
            ) as pbar:
                for time_step in range(total_steps):
                    step_start = perf_counter()
                    self.time_steps.append(time_step)
                    self.advance_time_step()

                    idx_extract = self._maybe_extract_solution(idx_extract)

                    step_time = perf_counter() - step_start

                    if time_step % throbber_every == 0:
                        throbber_state = next(throbber)

                    pbar.set_description(f"Eating Burgers | {throbber_state}")
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "t": f"{self.simulation_time_elapsed:.3f}",
                            "dt": f"{self.dt:.3f}",
                            "step_time": f"{step_time:.3f}s",
                        }
                    )
            # flush any checkpoints not yet triggered (e.g. final time landing just short)
            if self.time_extractions is not None:
                while idx_extract < len(self.time_extractions):
                    self.extracted_solutions.append(self.solution.copy())

                    append_forcing = self.forcing_current.copy()
                    print(f"forcing {append_forcing}")
                    self.extracted_forcings.append(append_forcing)

                    logger.info(
                        "Extracted solution at t=%.4f (end-of-simulation flush)",
                        self.time_extractions[idx_extract],
                    )
                    idx_extract += 1

        if self.write_solutions:
            self.write_config_to_json()
            self.write_solution_to_csv()

    def advance_time_step(self) -> None:
        """Advances the solution by one time-step, U^n+1 <- U^n."""
        if callable(self.forcing):
            self.forcing_current = self.forcing(
                self.node_cords, self.simulation_time_elapsed
            )

        elif self.forcing == "uniform":
            self.forcing_current = np.ones_like(self.solution)

        elif self.forcing is None:
            self.forcing_current = np.zeros_like(self.solution)  # static array or None

        else:
            self.forcing_current = np.zeros_like(self.solution)  # static array or None

        self.solution = self.nr_iteration(solution=self.solution)
        self.energy_history.append(self.compute_energy(self.solution))
        self.dissipation_history.append(self.compute_dissipation(self.solution))
        self.simulation_time_elapsed += self.dt

    def nr_iteration(self, solution: NDArray) -> NDArray:
        """Newton Raphson iteration loop to approximate U^n+1."""
        solution_n = solution.copy()  # U^n
        solution_k = solution.copy()  # U^k, k=0
        residual_history_loop = []
        update_history_loop = []

        for _ in range(self.max_iterations):
            with self.timer("elemental_iterations"):
                elemental_residuals, elemental_jacobians = zip(
                    *(
                        self.calculate_elemental_residual_jacobian(
                            element=element,
                            u_k=solution_k[element],
                            u_n=solution_n[element],
                            f_e=self.forcing_current[element]
                            if self.forcing_current is not None
                            else None,
                        )
                        for element in self.elements
                    )
                )

            with self.timer("global_assembly"):
                global_residual, global_jacobian = self.global_assembly(
                    elemental_residuals, elemental_jacobians
                )

            with self.timer("boundary_conditions"):
                global_residual, global_jacobian = self._apply_boundary_conditions(
                    global_residual, global_jacobian, solution_k
                )

            residual_history_loop.append(np.linalg.norm(global_residual))

            with self.timer("linear_solve"):
                delta_u = np.linalg.solve(global_jacobian, -global_residual)

                if self.configuration["boundary_conditions"] == "periodic":
                    delta_u_reduced = delta_u.copy()
                    delta_u = np.zeros_like(solution_k)
                    delta_u[:-1] = delta_u_reduced
                    delta_u[-1] = delta_u[0]

            update_history_loop.append(np.linalg.norm(delta_u))

            with self.timer("solution_update"):
                solution_k += (
                    delta_u * (1 - self.relaxation_factor)
                    if self.relaxation_factor is not None
                    else delta_u
                )

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
        """Calculate the residual R_e ∈ ℝ² and Jacobian J_e ∈ ℝ²ˣ² for one element."""
        residual_element = np.zeros(2)
        jacobian_element = np.zeros((2, 2))

        weight = abs(self.element_size / 2)  # |dx/dξ|
        points, weights = self.gauss_legendre(3)
        gradient_basis = self.reference_gradient_basis_functions()  # [dN_1, dN_2]
        is_les = self.configuration["simulation_type"] != "dns"

        for g_p, g_w in zip(points, weights):
            basis = self.reference_basis_functions(g_p)  # [N_1(ξ), N_2(ξ)]
            interp_fields = self._interpolate_fields(basis, gradient_basis, u_k, u_n)
            midpoint_fields = self._midpoint_fields(interp_fields)
            strong_res = self._strong_residual(interp_fields)
            scale = g_w * weight
            f_interp = float(basis @ f_e) if f_e is not None else 0.0

            for i in range(len(element)):
                residual_element[i] += scale * self._residual_integrand(
                    i, basis, gradient_basis, interp_fields, midpoint_fields, f_interp
                )

                if is_les:
                    residual_element[i] += scale * self._vms_residual_integrand(
                        i, gradient_basis, interp_fields
                    )

                for j in range(len(element)):
                    jacobian_element[i, j] += scale * self._jacobian_integrand(
                        i, j, basis, gradient_basis, interp_fields
                    )

                    if is_les:
                        jacobian_element[i, j] += scale * self._vms_jacobian_integrand(
                            i, j, basis, gradient_basis, interp_fields, strong_res
                        )

        return residual_element, jacobian_element

    def global_assembly(
        self, elemental_residuals: Iterable[NDArray], elemental_jacobians: list[NDArray]
    ) -> tuple[NDArray, NDArray]:
        """Assemble elemental residuals and elemental Jacobians to construct global matrices R, J."""
        global_residual = np.zeros(self.n_nodes)
        global_jacobian = np.zeros([self.n_nodes, self.n_nodes])

        for e, element in enumerate(self.elements):
            i, j = element
            residual_e = elemental_residuals[e]
            jacobian_e = elemental_jacobians[e]
            global_residual[i] += residual_e[0]
            global_residual[j] += residual_e[1]
            global_jacobian[i][i] += jacobian_e[0][0]
            global_jacobian[i][j] += jacobian_e[0][1]
            global_jacobian[j][i] += jacobian_e[1][0]
            global_jacobian[j][j] += jacobian_e[1][1]

        return global_residual, global_jacobian

    def compute_tau(self, variable_u: float | NDArray) -> float:
        """Computes the sgs contribution function tau."""
        term_time = (2 / self.dt) ** 2
        term_adv = (2 * abs(variable_u) / self.element_size) ** 2
        term_diff = (4 * self.viscosity / self.element_size**2) ** 2
        return 0.5 / np.sqrt(term_time + term_adv + term_diff)

    def _apply_boundary_conditions(
        self,
        global_residual: NDArray,
        global_jacobian: NDArray,
        solution_k: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Dispatch to the correct BC method based on configuration."""
        bc_type = self.boundary_conditions
        if bc_type == "fixed":
            return self._apply_fixed_bcs(
                global_residual, global_jacobian, solution_k, target_value=0
            )
        elif bc_type == "fixed_one":
            return self._apply_fixed_bcs(
                global_residual, global_jacobian, solution_k, target_value=1
            )
        elif bc_type == "periodic":
            return self._apply_periodic_bcs(global_residual, global_jacobian)
        else:
            raise ValueError(
                f"Unknown boundary condition type: '{bc_type!r}'. Expected 'fixed' or 'periodic'."
            )

    def _apply_fixed_bcs(
        self,
        global_residual: NDArray,
        global_jacobian: NDArray,
        solution_k: NDArray,
        target_value: float = 0.0,
    ) -> tuple[NDArray, NDArray]:
        """Enforce Dirichlet (fixed) boundary conditions by row-replacement."""
        for node in self.boundary_nodes:
            global_residual[node] = solution_k[node] - target_value
            global_jacobian[node, :] = 0
            global_jacobian[node, node] = 1
        return global_residual, global_jacobian

    @staticmethod
    def _apply_periodic_bcs(
        global_residual: NDArray,
        global_jacobian: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Enforce periodic BCs by folding the last node into the first, then removing it."""
        global_residual[0] += global_residual[-1]
        global_jacobian[0, :] += global_jacobian[-1, :]
        global_jacobian[:, 0] += global_jacobian[:, -1]

        return global_residual[:-1], global_jacobian[:-1, :-1]

    @staticmethod
    def is_residual_converged(
        residual: float | NDArray, tolerance: float = 1e-6
    ) -> bool:
        """Checks if the residual R is small enough to ensure convergence."""
        norm = np.linalg.norm(residual)
        return norm < tolerance * (1 + np.linalg.norm(residual))

    @staticmethod
    def is_update_converged(correction: NDArray, tolerance: float = 1e-6) -> bool:
        """Checks if Delta_U is small enough to ensure convergence."""
        return np.linalg.norm(correction) < tolerance

    def compute_energy(self, solution: NDArray) -> float:
        """Compute total energy of the given solution."""
        energy = 0
        jacobian = self.element_size / 2
        points, weights = self.gauss_legendre(number_of_points=2)

        for element in self.elements:
            u_e = np.array([solution[node] for node in element])

            for g_p, g_w in zip(points, weights):
                basis = self.reference_basis_functions(g_p)
                u_interp = basis @ u_e

                energy += 0.5 * g_w * abs(jacobian) * u_interp**2
        return energy

    def compute_energy_spectrum(self, solution: NDArray) -> tuple[NDArray, NDArray]:
        """Spectral analysis of energy values."""
        length = self.domain_length
        n_nodes = len(solution)
        u_hat = np.fft.fft(solution)
        wavenumbers = np.fft.fftfreq(n_nodes, d=length / n_nodes) * 2 * np.pi
        spectrum = 0.5 * np.abs(u_hat**2) / n_nodes
        return wavenumbers, spectrum

    @staticmethod
    def get_positive_spectrum(
        wavenumbers: NDArray, spectrum: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Keep only positive parts."""
        mask = wavenumbers >= 0
        return wavenumbers[mask], spectrum[mask]

    def compute_dissipation(self, solution: NDArray) -> float:
        """Compute dissipation level of the given solution."""
        dissipation = 0.0
        jacobian = self.element_size / 2
        points, weights = self.gauss_legendre(2)
        dn_dx = self.reference_gradient_basis_functions()

        for element in self.elements:
            u_e = np.array([solution[node] for node in element])

            for g_p, g_w in zip(points, weights):
                du_dx = dn_dx @ u_e
                dissipation += self.viscosity * g_w * abs(jacobian) * (du_dx**2)

        return dissipation

    @staticmethod
    def _interpolate_fields(
        basis: NDArray,
        gradient_basis: NDArray,
        u_k: NDArray,
        u_n: NDArray,
    ) -> dict:
        """Interpolate field values and gradients at a Gauss point."""
        return {
            "u_k": basis @ u_k,
            "u_n": basis @ u_n,
            "du_k": gradient_basis @ u_k,
            "du_n": gradient_basis @ u_n,
        }

    @staticmethod
    def _midpoint_fields(f: dict[str, float]) -> dict[str, float]:
        """Compute midpoint (Crank-Nicolson) averages of interpolated fields."""
        return {
            "u_mid": 0.5 * (f["u_k"] + f["u_n"]),
            "du_mid": 0.5 * (f["du_k"] + f["du_n"]),
        }

    def _strong_residual(self, f: dict[str, float]) -> float:
        """Evaluate the strong-form residual at a Gauss point."""
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
        """Gauss-point integrand for the residual vector (DNS / Galerkin part)."""
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
        """SGS / VMS stabilisation contribution to the residual."""
        u_mid = 0.5 * (f["u_k"] + f["u_n"])
        du_mid = 0.5 * (f["du_k"] + f["du_n"])

        tau_mid = self.compute_tau(u_mid)

        vms = (u_mid * gradient_basis[i]) * tau_mid * u_mid * du_mid

        return vms

    def _jacobian_integrand(
        self,
        i: int,
        j: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
    ) -> float:
        """Gauss-point integrand for the Jacobian matrix (DNS / Galerkin part)."""
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
        """SGS / VMS stabilisation contribution to the Jacobian."""
        u_mid = 0.5 * (f["u_k"] + f["u_n"])
        du_mid = 0.5 * (f["du_k"] + f["du_n"])

        tau_mid = self.compute_tau(u_mid)

        spatial_part = basis[j] * du_mid + u_mid * gradient_basis[j]
        tau_part = basis[j] * gradient_basis[i] * tau_mid * strong_res

        return spatial_part + tau_part

    def print_configuration(self) -> None:
        """Print the configuration to see run details."""
        print("=" * 60)
        print("Configuration settings")
        print("Time Integration: Second Order Implicit Euler")
        print("=" * 60)
        for k, v in self.configuration.items():
            if k == "time_extractions":
                print(f"{k}: {len(v)}")
            elif k != "solution_initial":
                print(f"{k}: {round(v, ndigits=4) if isinstance(v, float) else v}")
        print("=" * 60)

    def _maybe_extract_solution(self, idx_extract: int) -> int:
        """Snapshot the solution at each extraction checkpoint."""
        if self.time_extractions is None:
            return idx_extract

        if (
            idx_extract < len(self.time_extractions)
            and self.simulation_time_elapsed >= self.time_extractions[idx_extract]
        ):
            self.extracted_solutions.append(self.solution.copy())
            self.extracted_forcings.append(
                self.forcing_current.copy()
                if self.forcing_current is not None
                else np.zeros_like(self.solution)
            )
            idx_extract += 1

        return idx_extract

    def write_solution_to_csv(self) -> None:
        """Write solution(s) to CSV file(s) inside a run-specific directory."""
        nodes = self.nodes
        coordinates = self.node_cords
        run_dir = self.run_dir

        if self.time_extractions is None:
            solutions = [self.solution]
            forcings = [self.forcing_current]
            times = [self.simulation_time_elapsed]

        else:
            solutions = self.extracted_solutions
            forcings = self.extracted_forcings
            times = self.time_extractions[: len(solutions)]

        write_count = 0

        for solution, time, forcing in zip(solutions, times, forcings):
            filepath = run_dir / f"sol_t{time:.3f}.csv"

            with open(filepath, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["node_index", "x_coordinate", "velocity", "forcing"])

                for i in range(len(nodes)):
                    writer.writerow([nodes[i], coordinates[i], solution[i], forcing[i]])

            write_count += 1

        print(f"wrote {write_count} snapshots at {run_dir}")

    def write_config_to_json(self) -> None:
        """Write run configuration to a JSON file in the run directory."""
        config_serializable = {}
        for k, v in self.configuration.items():
            if k == "solution_initial":
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

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config_serializable, f, indent=2)

    def post_processing(self) -> None:
        """Post-processing of the solution from the simulation."""
        self.post_plotting()
        self.post_logging()

    def post_logging(self) -> None:
        """Post logging routine."""
        self.logger.info("=" * 60)
        self.logger.info("START OUTPUT")
        self.logger.info("-" * 60)
        self.logger.info(
            "Run started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        self.logger.info("-" * 60)
        self.logger.info("RUN COMPLETE — id: %s", self.run_id)
        self.logger.info("=" * 60)
        # --- Settings ---
        config_loggable = {
            k: round(v, ndigits=4) if isinstance(v, float) else v
            for k, v in self.configuration.items()
            if k != "solution_initial"
        }
        self.logger.info("-" * 60)
        self.logger.info("Configuration settings:")
        self.logger.info("  " + "Time Integration: Second Order Implicit Euler")
        self.logger.info("  " + "-" * 40)
        self.logger.info("  %-30s %s", "Parameter", "Value")
        self.logger.info("  " + "-" * 40)
        for k, v in config_loggable.items():
            if k == "time_extractions":
                self.logger.info("  %-30s %s", k, len(v))
            else:
                self.logger.info("  %-30s %s", k, v)
        self.logger.info("  " + "-" * 40)
        self.logger.info("-" * 60)
        # --- Timings ---
        if self.timings_performance:
            self.logger.info("Timings:")
            total = self.timings_performance["total_simulation"]
            for phase, t in sorted(self.timings_performance.items()):
                if phase != "total_simulation":
                    self.logger.info(
                        "  %-25s %.4fs (%5.1f%%)", phase, t, 100 * t / total
                    )
            self.logger.info("  %-25s %.4fs", "TOTAL", total)
        else:
            self.logger.warning("No timing data recorded.")
        self.logger.info("-" * 60)
        # --- Convergence ---
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
        self.logger.info("-" * 60)
        self.logger.info("END OUTPUT")
        self.logger.info("=" * 60)

    def _setup_logger(self) -> logging.Logger:
        """Set up a run-specific logger writing to logs/<run_id>.log."""
        logger_ = logging.getLogger(str(self.run_id))  # unique logger per run
        logger_.setLevel(logging.INFO)

        if logger_.handlers:  # avoid duplicate handlers on re-instantiation
            return logger_

        formatter = logging.Formatter("[%(levelname)s] - %(message)s")

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(
            self.run_dir / f"{self.run_id}.log", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        logger_.addHandler(stream_handler)
        logger_.addHandler(file_handler)
        logger_.propagate = False  # don't bubble up to the root logger

        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        return logger_

    @staticmethod
    def moving_stats(arr: list, window: int = 5) -> tuple[NDArray, NDArray]:
        """Create mean and standard deviations for global convergence plotting."""
        arr = np.array(arr)
        if arr.size == 0:
            return np.array([]), np.array([])
        effective_window = min(window, len(arr))
        means = np.convolve(
            arr, np.ones(effective_window) / effective_window, mode="same"
        )
        stds = np.array(
            [
                np.std(
                    arr[
                        max(0, i - effective_window // 2) : min(
                            len(arr), i + effective_window // 2 + 1
                        )
                    ]
                )
                for i in range(len(arr))
            ]
        )
        return means, stds

    def post_plotting(self) -> None:
        """Plot solution + convergence diagnostics."""
        first_res = [r[0] for r in self.residual_history if len(r) > 0]
        last_res = [r[-1] for r in self.residual_history if len(r) > 0]
        first_upd = [u[0] for u in self.update_history if len(u) > 0]
        last_upd = [u[-1] for u in self.update_history if len(u) > 0]
        fr_mean, fr_std = self.moving_stats(first_res)
        lr_mean, lr_std = self.moving_stats(last_res)
        fu_mean, fu_std = self.moving_stats(first_upd)
        lu_mean, lu_std = self.moving_stats(last_upd)

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(3, 2)

        # 1. Solution (top full width)
        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(
            self.node_cords,
            self.solution,
            color="royalblue",
            linestyle="-",
            marker="o",
            label="Resolved solution",
        )
        ax0.plot(
            self.node_cords,
            self.initial_condition,
            color="grey",
            linestyle="--",
            label="Initial solution",
        )
        ax0.set_xlabel(r"$x \in [0, 2\pi]$")
        ax0.set_ylabel("Velocity")
        ax0.grid(True)
        ax0.legend()
        ax0.set_title("Solution")

        # 2. Global convergence (per timestep)
        ax1 = fig.add_subplot(gs[1, 0])
        t = np.arange(len(fr_mean))
        # --- Residual (first) ---
        ax1.plot(t, fr_mean, color="royalblue", label="Residual (first)")
        ax1.fill_between(
            t, fr_mean - fr_std, fr_mean + fr_std, color="royalblue", alpha=0.15
        )
        # --- Residual (last) ---
        ax1.plot(t, lr_mean, color="navy", linestyle="--", label="Residual (last)")
        ax1.fill_between(
            t, lr_mean - lr_std, lr_mean + lr_std, color="navy", alpha=0.15
        )
        # --- Update (first) ---
        ax1.plot(t, fu_mean, color="tab:orange", label="Update (first)")
        ax1.fill_between(
            t, fu_mean - fu_std, fu_mean + fu_std, color="tab:orange", alpha=0.15
        )
        # --- Update (last) ---
        ax1.plot(t, lu_mean, color="darkorange", linestyle="--", label="Update (last)")
        ax1.fill_between(
            t, lu_mean - lu_std, lu_mean + lu_std, color="darkorange", alpha=0.15
        )

        tolerance_linestyle = "--"

        if (
            self.configuration["convergence_tol_residual"]
            != self.configuration["convergence_tol_update"]
        ):
            ax1.axhline(
                y=self.configuration["convergence_tol_residual"],
                color="lightskyblue",
                linestyle=tolerance_linestyle,
            )
            ax1.axhline(
                y=self.configuration["convergence_tol_update"],
                color="lightsalmon",
                linestyle=tolerance_linestyle,
            )
        else:
            ax1.axhline(
                y=self.configuration["convergence_tol_residual"],
                color="lightgray",
                linestyle=tolerance_linestyle,
            )

        ax1.set_yscale("log")
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Norm")
        ax1.set_title("Global convergence (smoothed)")
        ax1.grid(True)
        ax1.legend()

        # 3. Last Newton iteration convergence
        ax2 = fig.add_subplot(gs[1, 1])
        if len(self.residual_history) > 0:
            res_last_nr = self.residual_history[-1]
            upd_last_nr = self.update_history[-1]
            ax2.plot(res_last_nr, "o-", label="Residual", color="royalblue")
            if len(upd_last_nr) > 0:
                ax2.plot(upd_last_nr, "x--", label="Update", color="tab:orange")
        ax2.set_yscale("log")
        ax2.set_xlabel("Newton iteration")
        ax2.set_ylabel("Norm")
        ax2.set_title("Last Newton iteration convergence")
        ax2.grid(True)
        ax2.legend()

        # 4. Energy evolution + Dissipation evolution
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
        # ax3.set_yscale("linear")
        ax3.set_xlabel("Time step")
        # ax3.set_ylabel("")
        ax3.set_title("Energy and dissipation evolution")
        ax3.grid(True)
        ax3.legend()

        # 5. spectral
        ax4 = fig.add_subplot(gs[2, 1])
        wavenumbers, spectrum = self.compute_energy_spectrum(self.solution)
        wavenumbers, spectrum = self.get_positive_spectrum(wavenumbers, spectrum)

        ax4.loglog(wavenumbers[1:], spectrum[1:], marker="o")
        ax4.set_xlabel("Wavenumber k")
        ax4.set_ylabel("E(k)")
        ax4.set_title("Spectral Analysis (at end)")
        ax4.grid(True)

        plt.tight_layout()
        plt.show()
