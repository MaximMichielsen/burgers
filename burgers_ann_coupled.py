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
    INPUT_STENCIL,
    OUTPUT_STENCIL,
    NORM_STATS,
    INPUT_UNITS,
    OUTPUT_UNITS,
)

logger = logging.getLogger(__name__)


def _to_callable(field) -> Callable | NDArray | None:
    if callable(field):
        return field
    arr = np.asarray(field)
    return lambda x: (
        np.interp(x, np.linspace(0, 1, len(arr)), arr) if field is not None else None
    )


class Burgers:
    """Burgers class solver to simulate the discrete system: M * U_t + A(U) * U + nu * K_0 * U + C_fs(U) = f.

    When simulation_type == "les_ann", the VMS/SGS stabilisation terms are
    replaced by predictions from a pre-trained MLP (SGSPredictor).  The model
    can be supplied at construction time via ``ann_model`` **or** loaded from
    disk via ``ann_model_path``; at least one must be provided for this mode.
    """

    def __init__(
        self,
        configuration: dict,
        ann_model=None,  # optional: a loaded SGSPredictor instance
        ann_model_path: str
        | Path
        | None = None,  # optional: path/dir to saved .pth file
        ann_model_class=None,  # the SGSPredictor class (not instance)
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
        self.solution: NDArray = self.set_initial_condition(
            initial_condition=self.configuration["initial_condition"]
        )
        self.initial_condition: NDArray = self.solution.copy()
        self.forcing: NDArray | Callable[[NDArray, float], NDArray] | None = (
            self.configuration["forcing"]
            if self.configuration["forcing"] is not None
            else None
        )
        self.forcing_is_steady: bool = self.configuration["forcing_behavior"]
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

        # --------------------------      ANN / SGS       -------------------------- #
        # Keeps the last two LES snapshots so the stencil builder can form
        # ū^{n-1} and ū^{n-2} exactly as during training.
        self._ann_solution_history: list[NDArray] = []

        self.ann_model = self._load_ann_model(
            ann_model, ann_model_path, ann_model_class
        )
        self.ann_norm_stats: dict | None = self._load_norm_stats(ann_model_path)
        # ========================================================================== #

    # ------------------------------------------------------------------ #
    #  ANN helpers                                                         #
    # ------------------------------------------------------------------ #

    def _load_ann_model(self, model, model_path: str | Path | None, model_class):
        """Return a ready-to-use SGSPredictor, or None if not les_ann mode.

        Parameters
        ----------
        model:
            A pre-instantiated, pre-loaded model object (takes priority).
        model_path:
            Directory or .pth file path.  When a directory is given the loader
            looks for ``sgs_mlp_model.pth`` inside it.
        model_class:
            The *class* (not an instance) used to instantiate the network before
            loading weights — e.g. ``SGSPredictor`` imported in the caller.
            Required when ``model_path`` is supplied and ``model`` is None.
        """
        if self.simulation_type != "les_ann":
            return None

        if model is not None:
            model.eval()
            return model

        if model_path is not None:
            import torch

            model_path = Path(model_path)

            # Accept either a directory (look for the canonical filename)
            # or a direct path to the .pth file.
            if model_path.is_dir():
                candidates = list(model_path.glob("*.pth")) + list(
                    model_path.glob("*.pt")
                )
                if not candidates:
                    raise FileNotFoundError(
                        f"No .pth/.pt file found in directory '{model_path}'."
                    )
                weights_path = candidates[0]
                if len(candidates) > 1:
                    logger.warning(
                        "Multiple weight files found in '%s'; using '%s'.",
                        model_path,
                        weights_path.name,
                    )
            else:
                weights_path = model_path

            if model_class is None:
                raise ValueError(
                    "model_class must be provided when loading from ann_model_path "
                    "(e.g. pass model_class=SGSPredictor from your ann.py)."
                )

            loaded = model_class(input_dim=INPUT_UNITS, output_dim=OUTPUT_UNITS)
            loaded.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
            loaded.eval()
            logger.info("ANN SGS model loaded from '%s'.", weights_path)
            return loaded

        raise ValueError(
            "simulation_type='les_ann' requires either ann_model or ann_model_path."
        )

    def _load_norm_stats(self, model_path: str | Path | None) -> dict | None:
        """Load normalisation statistics saved alongside the model weights.

        Searches the following locations in order:
          1. The directory itself (model_path if dir, else model_path.parent)
          2. Same stem as NORM_STATS but with .npz extension (numpy savez behaviour)
        Raises a RuntimeError if stats are not found so the caller is never
        silently running un-normalised through the ANN.
        """
        if self.simulation_type != "les_ann":
            return None

        if model_path is None:
            raise ValueError(
                "ann_model_path must be provided for les_ann so normalisation "
                "stats can be located alongside the weights."
            )

        model_path = Path(model_path)
        search_dir = model_path if model_path.is_dir() else model_path.parent

        stem = Path(NORM_STATS).stem  # strip .npz if already present
        candidates = [
            search_dir / NORM_STATS,  # exact constant name
            search_dir / (stem + ".npz"),  # numpy savez appends .npz
        ]

        for stats_path in candidates:
            if stats_path.exists():
                data = np.load(stats_path, allow_pickle=True)
                stats = {k: data[k] for k in data.files}
                logger.info("Normalisation stats loaded from '%s'.", stats_path)
                return stats

        raise FileNotFoundError(
            f"Normalisation stats not found. Searched:\n"
            + "\n".join(f"  {p}" for p in candidates)
            + "\n\nCopy the norm-stats file next to the model weights "
            "(see main.py save block), or pass stats explicitly."
        )

    # ------------------------------------------------------------------
    #  Stencil builder — mirrors extract_stencil() from projection.py
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_stencil(field: NDArray) -> NDArray:
        """Build a 4-point periodic stencil [i-2, i-1, i, i+1] for every node."""
        return np.stack(
            [
                np.roll(field, 2),  # i-2
                np.roll(field, 1),  # i-1
                field,  # i
                np.roll(field, -1),  # i+1
            ],
            axis=1,
        )  # (N, 4)

    def _build_ann_input(
        self,
        u_now: NDArray,
        u_prev1: NDArray,
        u_prev2: NDArray,
        du_bar_dt: NDArray,
        forcing: NDArray,
    ) -> NDArray:
        """Assemble the (N, 20) feature matrix expected by the ANN.

        Column layout (matches build_features in projection.py):
            cols  0– 3  : stencil of ū^n
            cols  4– 7  : stencil of ū^{n-1}
            cols  8–11  : stencil of ū^{n-2}
            cols 12–15  : stencil of ∂ū/∂t
            cols 16–19  : stencil of forcing
        """
        return np.hstack(
            [
                self._extract_stencil(u_now),  # (N, 4)
                self._extract_stencil(u_prev1),  # (N, 4)
                self._extract_stencil(u_prev2),  # (N, 4)
                self._extract_stencil(du_bar_dt),  # (N, 4)
                self._extract_stencil(forcing),  # (N, 4)
            ]
        )  # → (N, 20)

    def _normalise_input(self, X: NDArray) -> NDArray:
        if self.ann_norm_stats is None:
            return X
        return (X - self.ann_norm_stats["X_mean"]) / self.ann_norm_stats["X_std"]

    def _denormalise_output(self, y: NDArray) -> NDArray:
        if self.ann_norm_stats is None:
            return y
        return y * self.ann_norm_stats["y_std"] + self.ann_norm_stats["y_mean"]

    def _predict_sgs_nodal(
        self,
        u_now: NDArray,
        forcing: NDArray,
        du_bar_dt: NDArray,
    ) -> NDArray:
        """Run the ANN and return the per-node SGS correction vector (N,).

        The ANN predicts 4 output channels per node matching the weak-form
        fine-scale forcing term from eq. (2.4):

            term_0 : (w̄, ∂/∂x(ū² + 0.5τ))  — resolved+SGS convective flux deriv
            term_1 : (w̄_l, u'_t)             — unsteady SGS, left  basis weight
            term_2 : (w̄_r, u'_t)             — unsteady SGS, right basis weight
            term_3 : (w̄_x, u'_x)·ν           — SGS diffusion / stress gradient

        term_0 contains the *resolved* convective flux which the Galerkin
        element loop already computes; only the pure SGS part (terms 1–3)
        is added to the global residual to avoid double-counting.
        """
        import torch  # local import — no dependency for non-ANN runs

        history = self._ann_solution_history
        u_prev1 = history[-1] if len(history) >= 1 else np.zeros_like(u_now)
        u_prev2 = history[-2] if len(history) >= 2 else np.zeros_like(u_now)

        X_raw = self._build_ann_input(u_now, u_prev1, u_prev2, du_bar_dt, forcing)
        X_norm = self._normalise_input(X_raw)

        with torch.no_grad():
            tensor_in = torch.tensor(X_norm, dtype=torch.float32)
            tensor_out = self.ann_model(tensor_in)  # (N, 4)
            y_norm = tensor_out.numpy()

        y = self._denormalise_output(y_norm)         # (N, 4)  physical units

        # Only add the pure SGS channels (1, 2, 3); channel 0 is the resolved
        # convective flux derivative already present in the Galerkin residual.
        # term_1 + term_2 assembles (w̄, u'_t); term_3 is the stress gradient.
        sgs_nodal = y[:, 1] + y[:, 2] + y[:, 3]  # (N,)

        logger.debug(
            "[ANN SGS] channel norms  — 0:%.3e  1:%.3e  2:%.3e  3:%.3e",
            np.linalg.norm(y[:, 0]),
            np.linalg.norm(y[:, 1]),
            np.linalg.norm(y[:, 2]),
            np.linalg.norm(y[:, 3]),
        )
        logger.debug(
            "[ANN SGS] correction norm: %.3e  (residual will shift by this amount)",
            np.linalg.norm(sgs_nodal),
        )

        return sgs_nodal

    def _update_solution_history(self, u: NDArray) -> None:
        """Keep a rolling buffer of the last 2 resolved snapshots."""
        self._ann_solution_history.append(u.copy())
        if len(self._ann_solution_history) > 2:
            self._ann_solution_history.pop(0)

    # ================================================================== #
    #  Public / existing API — unchanged below unless noted               #
    # ================================================================== #

    def set_initial_condition(self, initial_condition: NDArray | Callable) -> NDArray:
        """Set the initial condition based on the type of initial condition given."""
        if isinstance(initial_condition, Callable):
            return initial_condition(self.node_cords)
        return initial_condition.copy()

    @staticmethod
    def create_config(
        initial_condition: NDArray | Callable,
        simulation_type: str,
        node_amount: int,
        viscosity: float,
        time_step: float,
        domain_timespan: float,
        domain_length: float,
        forcing: NDArray | Callable | None = None,
        forcing_is_steady: bool = True,
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
            "forcing_behavior": forcing_is_steady,
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
            # flush any checkpoints not yet triggered
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
            self.forcing_current = (
                self.forcing(self.node_cords, self.simulation_time_elapsed)
                if not self.forcing_is_steady
                else self.forcing(self.node_cords)
            )
        elif self.forcing is None:
            self.forcing_current = np.zeros_like(self.solution)
        else:
            self.forcing_current = self.forcing

        # Snapshot before updating so the NR loop sees U^n in the history.
        if self.simulation_type == "les_ann":
            self._update_solution_history(self.solution)

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

        # ---- ANN: compute per-node SGS correction once per time-step ----
        # We evaluate the model at the start of the NR loop (using U^n) so
        # that the correction is fixed during the inner Newton iterations.
        # This is the standard "frozen SGS" approach; it keeps the Jacobian
        # consistent with the standard Galerkin part and avoids the
        # significant cost of differentiating through the network every
        # Newton step.
        ann_sgs_nodal: NDArray | None = None
        if self.simulation_type == "les_ann":
            with self.timer("ann_sgs_prediction"):
                du_bar_dt = (
                    (solution_n - self._ann_solution_history[-1]) / self.dt
                    if len(self._ann_solution_history) >= 1
                    else np.zeros_like(solution_n)
                )
                ann_sgs_nodal = self._predict_sgs_nodal(
                    u_now=solution_n,
                    forcing=self.forcing_current
                    if self.forcing_current is not None
                    else np.zeros_like(solution_n),
                    du_bar_dt=du_bar_dt,
                )

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

            # ---- ANN: add frozen nodal SGS corrections to the residual ----
            if ann_sgs_nodal is not None:
                with self.timer("ann_sgs_assembly"):
                    global_residual = self._assemble_ann_sgs(
                        global_residual, ann_sgs_nodal
                    )

            with self.timer("boundary_conditions"):
                global_residual, global_jacobian = self._apply_boundary_conditions(
                    global_residual, global_jacobian, solution_k
                )

            residual_history_loop.append(np.linalg.norm(global_residual))

            print('jacob:', np.isnan(global_jacobian).any())
            print('res:', np.isnan(global_residual).any())
            print("cond:", np.linalg.cond(global_jacobian))
            print("max ann:", max(ann_sgs_nodal))
            print("min ann:", min(ann_sgs_nodal))

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

    # ------------------------------------------------------------------
    #  ANN SGS assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_ann_sgs(
        global_residual: NDArray,
        sgs_nodal: NDArray,
    ) -> NDArray:
        """Add the ANN SGS correction (channels 1+2+3) to the global residual.

        Channel 0 (resolved convective flux derivative) is intentionally
        excluded to avoid double-counting with the Galerkin element loop.
        The remaining channels are already nodal weak-form quantities that
        map 1-to-1 onto the global DOF vector — no element loop required.
        """
        return global_residual + sgs_nodal

    # ------------------------------------------------------------------
    #  Element-level residual / Jacobian  (unchanged)
    # ------------------------------------------------------------------

    def calculate_elemental_residual_jacobian(
        self,
        element: tuple[int, int],
        u_k: NDArray,
        u_n: NDArray,
        f_e: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Calculate the residual R_e ∈ ℝ² and Jacobian J_e ∈ ℝ²ˣ² for one element.

        For les_ann mode the analytic VMS terms are skipped; the SGS correction
        is applied globally in nr_iteration via _assemble_ann_sgs.
        """
        residual_element = np.zeros(2)
        jacobian_element = np.zeros((2, 2))

        weight = abs(self.element_size / 2)
        points, weights = self.gauss_legendre(3)
        gradient_basis = self.reference_gradient_basis_functions()

        # "les_vms"  → analytic VMS stabilisation
        # "les_ann"  → ANN-based SGS (no analytic VMS terms here)
        # "dns"      → pure Galerkin, no SGS
        use_vms = self.simulation_type not in ("dns", "les_ann")

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

                if use_vms:
                    residual_element[i] += scale * self._vms_residual_integrand(
                        i, gradient_basis, interp_fields
                    )

                for j in range(len(element)):
                    jacobian_element[i, j] += scale * self._jacobian_integrand(
                        i, j, basis, gradient_basis, interp_fields
                    )

                    if use_vms:
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
        global_residual[0] += global_residual[-1]
        global_jacobian[0, :] += global_jacobian[-1, :]
        global_jacobian[:, 0] += global_jacobian[:, -1]
        return global_residual[:-1], global_jacobian[:-1, :-1]

    @staticmethod
    def is_residual_converged(
        residual: float | NDArray, tolerance: float = 1e-6
    ) -> bool:
        norm = np.linalg.norm(residual)
        return norm < tolerance * (1 + np.linalg.norm(residual))

    @staticmethod
    def is_update_converged(correction: NDArray, tolerance: float = 1e-6) -> bool:
        return np.linalg.norm(correction) < tolerance

    def compute_energy(self, solution: NDArray) -> float:
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
        mask = wavenumbers >= 0
        return wavenumbers[mask], spectrum[mask]

    def compute_dissipation(self, solution: NDArray) -> float:
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
        return {
            "u_k": basis @ u_k,
            "u_n": basis @ u_n,
            "du_k": gradient_basis @ u_k,
            "du_n": gradient_basis @ u_n,
        }

    @staticmethod
    def _midpoint_fields(f: dict[str, float]) -> dict[str, float]:
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

    def print_configuration(self) -> None:
        print("=" * 60)
        print("Configuration settings")
        print("Time Integration: Second Order Implicit Euler")
        print("=" * 60)
        for k, v in self.configuration.items():
            if k == "time_extractions":
                print(f"{k}: {len(v)}")
            elif k != "solution_initial":
                print(f"{k}: {round(v, ndigits=4) if isinstance(v, float) else v}")
        if self.simulation_type == "les_ann":
            print(f"ANN SGS model loaded: {self.ann_model is not None}")
            print(f"Normalisation stats loaded: {self.ann_norm_stats is not None}")
        print("=" * 60)

    def _maybe_extract_solution(self, idx_extract: int) -> int:
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
        config_serializable["ann_model_loaded"] = self.ann_model is not None

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config_serializable, f, indent=2)

    def post_processing(self) -> None:
        self.post_plotting()
        self.post_logging()

    def post_logging(self) -> None:
        self.logger.info("=" * 60)
        self.logger.info("START OUTPUT")
        self.logger.info("-" * 60)
        self.logger.info(
            "Run started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        self.logger.info("-" * 60)
        self.logger.info("RUN COMPLETE — id: %s", self.run_id)
        self.logger.info("=" * 60)
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
            self.logger.info("  %-30s %s", k, v)
        self.logger.info("  " + "-" * 40)
        if self.simulation_type == "les_ann":
            self.logger.info(
                "  %-30s %s", "ANN SGS model loaded", self.ann_model is not None
            )
            self.logger.info(
                "  %-30s %s", "Norm stats loaded", self.ann_norm_stats is not None
            )
        self.logger.info("-" * 60)
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
        logger_ = logging.getLogger(str(self.run_id))
        logger_.setLevel(logging.INFO)
        if logger_.handlers:
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
        logger_.propagate = False
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        return logger_

    @staticmethod
    def moving_stats(arr: list, window: int = 5) -> tuple[NDArray, NDArray]:
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
        ax0.set_title(
            f"Solution  [SGS: {'ANN' if self.simulation_type == 'les_ann' else self.simulation_type}]"
        )

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
        if (
            self.configuration["convergence_tol_residual"]
            != self.configuration["convergence_tol_update"]
        ):
            ax1.axhline(
                y=self.configuration["convergence_tol_residual"],
                color="lightskyblue",
                linestyle="--",
            )
            ax1.axhline(
                y=self.configuration["convergence_tol_update"],
                color="lightsalmon",
                linestyle="--",
            )
        else:
            ax1.axhline(
                y=self.configuration["convergence_tol_residual"],
                color="lightgray",
                linestyle="--",
            )
        ax1.set_yscale("log")
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Norm")
        ax1.set_title("Global convergence (smoothed)")
        ax1.grid(True)
        ax1.legend()

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
