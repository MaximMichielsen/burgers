"""Coupled Burgers FEM solver: Galerkin + ANN-predicted SGS correction.

Inherits from BurgersPure and overrides:
    advance_time_step  — maintains the 3-level lagged solution history and
                         records per-step diagnostics into the blow-up buffer.
    nr_iteration       — injects the frozen ANN correction into the global
                         residual after assembly (Jacobian left unchanged).
    run_simulation     — wraps the parent loop in try/finally so buffered
                         history is flushed on both clean and blown-up exits.

The Jacobian is unchanged (pure Galerkin). The ANN inputs are strictly from
lagged time levels (LFS), so the correction does not depend on u^k and adding
it to the Jacobian would be incorrect.

Blow-up buffer
--------------
_BlowupBuffer accumulates per-step snapshots of the RL state vector sn from
the research proposal (Eq. 2.8): raw E(k,t), dissipation rate, and the
previous artificial viscosity α_{n-1}. On termination the buffer is flushed
to a .npz archive and .txt summary regardless of exit reason.

References: Robijns (2019), Pusuluri (2021), Rajampeta (2022),
            Research Proposal Sec. 2.3.
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

from constants import (
    OUTPUT_UNITS,
    BLOWUP_BUFFER_SIZE,
    BLOWUP_THRESHOLD,
)
from ml.data_assembly.training_data_assembly import (
    gradient_basis_functions,
    build_input_stencil,
)
from ml.ml_agents.predictor import SGSPredictor, load_predictor

from problems_and_configurations.disc_config import DiscretisationConfig
from problems_and_configurations.problems import Problem
from ml.ml_agents.solver_configs import SGSPConfig
from solvers.burgers_base import BurgersBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blow-up history buffer
# ---------------------------------------------------------------------------


@dataclass
class _BlowupBuffer:
    """Rolling per-step snapshot buffer for post-blow-up diagnostics and RL data.

    Older entries are dropped once max_steps is reached to keep memory bounded.
    """

    max_steps: int = 5_000

    time_values: list[float] = field(default_factory=list)
    amplitude_snapshots: list[NDArray] = field(default_factory=list)
    energy_spectra: list[NDArray] = field(default_factory=list)
    dissipation_values: list[float] = field(default_factory=list)
    artificial_viscosity_values: list[float] = field(default_factory=list)
    energy_values: list[float] = field(default_factory=list)
    residual_norms_first: list[float] = field(default_factory=list)
    residual_norms_last: list[float] = field(default_factory=list)
    update_norms_last: list[float] = field(default_factory=list)
    corrector_pass_counts: list[int] = field(default_factory=list)

    def record(
        self,
        *,
        time_val: float,
        amplitudes: NDArray,
        energy_spectrum: NDArray,
        dissipation: float,
        artificial_viscosity: float,
        energy: float,
        residual_norm_first: float,
        residual_norm_last: float,
        update_norm_last: float,
        corrector_passes: int,
    ) -> None:
        """Append one timestep; drop oldest entry when buffer is full."""
        if len(self.time_values) >= self.max_steps:
            for lst in (
                self.time_values,
                self.amplitude_snapshots,
                self.energy_spectra,
                self.dissipation_values,
                self.artificial_viscosity_values,
                self.energy_values,
                self.residual_norms_first,
                self.residual_norms_last,
                self.update_norms_last,
                self.corrector_pass_counts,
            ):
                lst.pop(0)

        self.time_values.append(time_val)
        self.amplitude_snapshots.append(amplitudes.copy())
        self.energy_spectra.append(energy_spectrum.copy())
        self.dissipation_values.append(dissipation)
        self.artificial_viscosity_values.append(artificial_viscosity)
        self.energy_values.append(energy)
        self.residual_norms_first.append(residual_norm_first)
        self.residual_norms_last.append(residual_norm_last)
        self.update_norms_last.append(update_norm_last)
        self.corrector_pass_counts.append(corrector_passes)

    def __len__(self) -> int:
        return len(self.time_values)


def diagnose_sgsp_predictions(
    solver: BurgersSGSP,
    n_steps: int = 10,
) -> None:
    """Run n_steps and print ANN correction statistics per step."""

    for step_idx in range(n_steps):
        solver.advance_time_step()
        sgsp_output = solver.compute_sgsp_contribution()

        if sgsp_output is None:
            print(f"Step {step_idx:03d}: correction=None (warmup)")
            continue

        col_names = ["cross", "reynolds", "temporal_L", "temporal_R", "viscous"]
        col_norms = np.abs(sgsp_output).mean(axis=0)  # mean over elements
        global_norm = np.linalg.norm(sgsp_output)

        print(
            f"Step {step_idx:03d}: "
            f"||correction||={float(global_norm):.3e}"
            + " | ".join(f"{name}={val:.3e}" for name, val in zip(col_names, col_norms))
        )

    # Also print normalization stats for context
    print(f"\ny_mean: {solver.y_mean}")
    print(f"y_std:  {solver.y_std}")


# ---------------------------------------------------------------------------
# Coupled solver
# ---------------------------------------------------------------------------
WARMUP_STEPS = 3


class BurgersSGSP(BurgersBase):
    """Burgers FEM solver with ANN-predicted SGS closure.

    Extends BurgersPure by injecting the trained SGS predictor into the
    global residual at each Newton–Raphson call.

    Additional configuration keys
    ------------------------------
    sgsp_model_path             : Path to sgs_predictor.pt.
    normalisation_stats_path   : Path to normalisation_stats.npz.
    sgsp_warmup_steps           : Pure-Galerkin steps before ANN activation (default 2).
    blowup_threshold           : Amplitude above which blow-up is declared (default 1e4).
    blowup_buffer_size         : Rolling window size for history buffer (default 5000).
    blown_up_path              : Output directory on blow-up; falls back to master_path.
    """

    def __init__(
        self,
        problem: Problem,
        disc_cfg: DiscretisationConfig,
        simulation_mode: str,
        master_path: Path,
        sgsp_cfg: SGSPConfig,
        snapshot_factor: int | None = 1,
    ) -> None:
        super().__init__(
            problem, disc_cfg, simulation_mode, master_path, snapshot_factor
        )

        self.clip_pusuluri: bool = sgsp_cfg.clip_pusuluri
        self.clip_rajampeta: bool = sgsp_cfg.clip_rajampeta

        if self.clip_rajampeta and not self.clip_pusuluri:
            raise ValueError("clip_rajampeta requires clip_pusuluri to be enabled.")

        self._sgsp_model_path = sgsp_cfg.sgsp_model_path

        self._u_bar_history: list[NDArray] = []
        self._du_bar_dt_history: list[NDArray] = []
        self._forcing_history: list[NDArray] = []

        self._sgsp_warmup_steps: int = WARMUP_STEPS
        self._step_count: int = 0
        self._grad_basis: NDArray = gradient_basis_functions(self.element_size)

        self._blowup_threshold: float = BLOWUP_THRESHOLD
        self._blowup_buffer: _BlowupBuffer = _BlowupBuffer(
            max_steps=int(BLOWUP_BUFFER_SIZE)
        )

        self._artificial_viscosity_prev: float = 0.0
        self._last_sgsp_correction: NDArray | None = None

        self._blown_up_path: Path = sgsp_cfg.blown_up_path

        self._sgsp_model_path = sgsp_cfg.sgsp_model_path
        self._normalization_path = sgsp_cfg.normalization_path
        self.set_off_predictions: bool = sgsp_cfg.set_off_predictor

        if self.set_off_predictions:
            self._predictor: SGSPredictor | None = None
            self._x_mean: NDArray | None = None
            self._x_std: NDArray | None = None
            self.y_mean: NDArray | None = None
            self.y_std: NDArray | None = None
        else:
            self._predictor = load_predictor(sgsp_cfg.sgsp_model_path)
            self._predictor.eval()

            norm_data = np.load(sgsp_cfg.normalization_path)
            self._x_mean = norm_data["X_mean"].astype(np.float32)
            self._x_std = norm_data["X_std"].astype(np.float32)
            self.y_mean = norm_data["y_mean"].astype(np.float32)
            self.y_std = norm_data["y_std"].astype(np.float32)

        self._u_bar_history: list[NDArray] = []
        self._du_bar_dt_history: list[NDArray] = []
        self._forcing_history: list[NDArray] = []

        self._sgsp_warmup_steps: int = WARMUP_STEPS
        self._step_count: int = 0
        self._grad_basis: NDArray = gradient_basis_functions(self.element_size)

        if self.clip_pusuluri and not self.set_off_predictions:
            self._y_lower_bound: NDArray = (
                self.y_mean - sgsp_cfg.sigma_multiplier * self.y_std
            )
            self._y_upper_bound: NDArray = (
                self.y_mean + sgsp_cfg.sigma_multiplier * self.y_std
            )

    # ------------------------------------------------------------------ #
    #  run_simulation — blow-up guard around parent loop
    # ------------------------------------------------------------------ #

    def run_simulation(self) -> None:
        """Run with guaranteed buffer flush on both clean and blown-up exits."""
        blowup_detected: bool = False

        try:
            self.snapshots_solution = []
            self.snapshots_forcing = []

            # IC snapshot (forcing evaluated at t=0)
            self.resolve_current_forcing()
            self._extract_snapshot()

            with self.timer("total_simulation"):
                with tqdm(
                    total=self._n_time_steps,
                    desc=f"Eating Burgers | {self.throbber(0)}",
                    file=sys.stdout,
                ) as pbar:
                    for time_step_idx in range(self._n_time_steps):
                        step_start = perf_counter()

                        step_ok = self.advance_time_step()
                        if not step_ok:
                            blowup_detected = True
                            logger.warning(
                                "Blow-up termination at t=%.6f",
                                self.simulation_time_elapsed,
                            )
                            break

                        if (time_step_idx + 1) in self._snapshot_step_indices:
                            self._extract_snapshot()

                        pbar.set_description(
                            f"Eating Burgers | {self.throbber(time_step_idx)}"
                        )
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "t": f"{self.simulation_time_elapsed:.3f}",
                                "dt": f"{self.dt:.3f}",
                                "step_time": f"{perf_counter() - step_start:.3f}s",
                            }
                        )

        except RuntimeError as exc:
            blowup_detected = True
            logger.warning("Blow-up termination: %s", exc)

        except Exception as exc:
            blowup_detected = True
            logger.error(
                "Unexpected solver exception: %s. Saving buffer before re-raise.", exc
            )
            raise

        finally:
            if blowup_detected:
                self._blown_up_path.mkdir(parents=True, exist_ok=True)
                self.master_path = self._blown_up_path
            self.write_config_to_json()
            if not blowup_detected:
                self.write_solution_to_csv()
            else:
                self._write_blowup_solutions_to_csv()
            if len(self._blowup_buffer) > 0:
                self._save_blowup_buffer(label="blowup" if blowup_detected else "clean")

    # ------------------------------------------------------------------ #
    #  advance_time_step — lagged history + blow-up detection
    # ------------------------------------------------------------------ #

    def advance_time_step(self) -> bool:
        """Advance one step; return False if blow-up detected, True otherwise.

        Replaces RuntimeError raises so direct callers (e.g. diagnostics) don't crash.
        """
        if self._detect_blowup(self.solution):
            logger.warning(
                "Blow-up detected at t=%.6f (pre-step): max|u|=%.4e > threshold %.4e",
                self.simulation_time_elapsed,
                np.max(np.abs(self.solution)),
                self._blowup_threshold,
            )
            return False

        super().advance_time_step()

        if self._detect_blowup(self.solution):
            logger.warning(
                "Blow-up detected at t=%.6f (post-step): max|u|=%.4e > threshold %.4e",
                self.simulation_time_elapsed,
                np.max(np.abs(self.solution)),
                self._blowup_threshold,
            )
            self._update_lagged_history()
            self._step_count += 1
            self._record_buffer_step()
            return False

        self._update_lagged_history()
        self._step_count += 1
        self._record_buffer_step()
        return True

    def _update_lagged_history(self) -> None:
        """Maintain the 3-level (t^n, t^{n-1}, t^{n-2}) solution history."""
        if len(self._u_bar_history) == 0:
            for _ in range(2):
                self._u_bar_history.append(self.initial_condition.copy())
                self._du_bar_dt_history.append(np.zeros(self.n_nodes))
                self._forcing_history.append(
                    self.forcing_current.copy()
                    if self.forcing_current is not None
                    else np.zeros(self.n_nodes)
                )

        u_bar_new = self.solution.copy()
        du_bar_dt_new = (u_bar_new - self._u_bar_history[-1]) / self.dt

        self._u_bar_history.append(u_bar_new)
        self._du_bar_dt_history.append(du_bar_dt_new)
        self._forcing_history.append(
            self.forcing_current.copy()
            if self.forcing_current is not None
            else np.zeros(self.n_nodes)
        )

        if len(self._u_bar_history) > 3:
            self._u_bar_history.pop(0)
            self._du_bar_dt_history.pop(0)
            self._forcing_history.pop(0)

    def _record_buffer_step(self) -> None:
        """Record the current step into the blow-up buffer."""
        wavenumbers, energy_spectrum = self.compute_energy_spectrum(self.solution)
        _, positive_spectrum = self.get_positive_spectrum(wavenumbers, energy_spectrum)

        res_step = self.residual_history[-1] if self.residual_history else [0.0]
        upd_step = self.update_history[-1] if self.update_history else [0.0]

        self._blowup_buffer.record(
            time_val=self.simulation_time_elapsed,
            amplitudes=self.solution,
            energy_spectrum=positive_spectrum,
            dissipation=self.dissipation_history[-1]
            if self.dissipation_history
            else 0.0,
            artificial_viscosity=self._artificial_viscosity_prev,
            energy=self.energy_history[-1] if self.energy_history else 0.0,
            residual_norm_first=float(res_step[0]),
            residual_norm_last=float(res_step[-1]),
            update_norm_last=float(upd_step[-1]),
            corrector_passes=len(res_step),
        )

    # ------------------------------------------------------------------ #
    #  nr_iteration — inject ANN correction into residual
    # ------------------------------------------------------------------ #

    def nr_iteration(self, solution: NDArray) -> NDArray:
        """NR iteration with frozen ANN correction added to the global residual."""
        sgsp_correction: NDArray | None = self.compute_sgsp_contribution()
        self._last_sgsp_correction = sgsp_correction  # cache for energy diagnostics

        solution_n = solution.copy()
        solution_k = solution.copy()
        residual_history_loop: list = []
        update_history_loop: list = []

        if self._step_count == self._sgsp_warmup_steps:
            print("\n--- SGSP prediction sample (step 0) ---")
            print(f"cross:    {sgsp_correction[:, 0]}")
            print(f"reynolds: {sgsp_correction[:, 1]}")
            print(f"viscous:  {sgsp_correction[:, 4]}")

        for _ in range(self.max_iterations):
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

            with self.timer("global_assembly"):
                global_residual, global_jacobian = self.global_assembly(
                    elemental_residuals, elemental_jacobians
                )

            if sgsp_correction is not None and not self.set_off_predictions:
                global_residual = self._add_sgsp_contribution_to_residual(
                    global_residual, sgsp_correction
                )

            global_residual, global_jacobian = self._apply_boundary_conditions(
                global_residual, global_jacobian, solution_k
            )
            residual_history_loop.append(np.linalg.norm(global_residual))

            with self.timer("linear_solve"):
                delta_u = np.linalg.solve(global_jacobian, -global_residual)
                if self.boundary_condition_type == "periodic":
                    delta_u_full = np.zeros_like(solution_k)
                    delta_u_full[:-1] = delta_u
                    delta_u_full[-1] = delta_u[0]
                    delta_u = delta_u_full

            update_history_loop.append(np.linalg.norm(delta_u))

            solution_k += delta_u

            if self.is_update_converged(delta_u) or self.is_residual_converged(
                global_residual
            ):
                break

        self.residual_history.append(residual_history_loop)
        self.update_history.append(update_history_loop)
        return solution_k

    # ------------------------------------------------------------------ #
    #  SGS_ANN helpers
    # ------------------------------------------------------------------ #

    def compute_sgsp_contribution(self) -> NDArray | None:
        """Build element input stencils and run a batched ANN forward pass.

        Returns (n_elements, 5) array of interaction terms, or None during warm-up.
        Columns: [cross, Reynolds, temporal_L, temporal_R, viscous].
        """
        if self.set_off_predictions:
            return None

        if self._step_count < self._sgsp_warmup_steps or len(self._u_bar_history) < 3:
            return None

        input_rows: list[NDArray] = []
        valid_element_indices: list[int] = []

        for elem_idx in range(self.n_elements):
            input_vec = build_input_stencil(
                u_bar_history=self._u_bar_history,
                du_bar_dt_history=self._du_bar_dt_history,
                forcing_history=self._forcing_history,
                element_idx=elem_idx,
                n_les_nodes=self.n_nodes,
            )
            if input_vec is not None:
                input_rows.append(input_vec)
                valid_element_indices.append(elem_idx)

        if not input_rows:
            return None

        x_batch = np.array(input_rows, dtype=np.float32)
        x_batch_norm = (x_batch - self._x_mean) / self._x_std

        with torch.no_grad():
            y_norm = self._predictor(torch.from_numpy(x_batch_norm)).numpy()

        y_phys = y_norm * self.y_std + self.y_mean

        if self.clip_pusuluri:
            y_phys = np.clip(y_phys, self._y_lower_bound, self._y_upper_bound)

        sgsp_correction_all = np.zeros(
            (self.n_elements, OUTPUT_UNITS), dtype=np.float64
        )
        for local_idx, elem_idx in enumerate(valid_element_indices):
            y_elem = y_phys[local_idx].copy()

            if self.clip_rajampeta:
                node_left, node_right = elem_idx, elem_idx + 1
                b0 = float(self._u_bar_history[-1][node_left])
                b1 = float(self._u_bar_history[-1][node_right])
                uet = -(
                    b0 * y_elem[2]
                    + b1 * y_elem[3]
                    - (b0 - b1) * y_elem[0]
                    - (b0 - b1) * y_elem[1]
                )
                if uet > 0:
                    y_elem[:] = 0.0

            sgsp_correction_all[elem_idx] = y_elem

        return sgsp_correction_all

    def _add_sgsp_contribution_to_residual(
        self,
        global_residual: NDArray,
        sgsp_correction: NDArray,
    ) -> NDArray:
        """Scatter per-element SGSP predictions into the global residual.

        Columns: 0=cross, 1=reynolds, 2=temporal_L, 3=temporal_R, 4=viscous.
        Spatial terms (cross, reynolds, viscous) integrate against w_x:
        opposite sign at left (-1/h) and right (+1/h) nodes.
        Temporal terms integrate against their respective shape functions.
        """
        if not np.all(np.isfinite(sgsp_correction)):
            return global_residual

        residual_modified = global_residual.copy()
        n_boundary_node: int = (
            int(self.nodes_les[-1]) if hasattr(self, "nodes_les") else self.n_nodes - 1
        )

        for elem_idx, element in enumerate(self.elements):
            node_left: int = int(element[0])
            node_right: int = int(element[1])

            cross_val: float = sgsp_correction[elem_idx, 0]
            reynolds_val: float = sgsp_correction[elem_idx, 1]
            temporal_left_val: float = sgsp_correction[elem_idx, 2]
            temporal_right_val: float = sgsp_correction[elem_idx, 3]
            viscous_val: float = sgsp_correction[elem_idx, 4]

            spatial_contribution: float = (
                cross_val + reynolds_val - self.viscosity * viscous_val
            )

            for global_node in [node_left, node_right]:
                if global_node in (0, n_boundary_node):
                    continue
                sign = 1.0 if global_node == node_right else -1.0
                residual_modified[global_node] -= sign * spatial_contribution

        return residual_modified

    def calc_sgsp_energy_injection(self) -> float:
        """Compute energy injected by the SGS predictor at the current step."""
        if self._last_sgsp_correction is None:
            return 0.0
        zero_residual = np.zeros(self.n_nodes, dtype=np.float64)
        sgsp_nodal_force = self._add_sgsp_contribution_to_residual(
            zero_residual, self._last_sgsp_correction
        )
        return float(np.dot(self.solution, sgsp_nodal_force))

    # ------------------------------------------------------------------ #
    #  Blow-up detection
    # ------------------------------------------------------------------ #

    def _detect_blowup(self, solution_amplitudes: NDArray) -> bool:
        """True if solution contains NaN, Inf, or exceeds the amplitude threshold."""
        return bool(
            np.any(np.isnan(solution_amplitudes))
            or np.any(np.isinf(solution_amplitudes))
            or np.max(np.abs(solution_amplitudes)) > self._blowup_threshold
        )

    def seed_history_from_projection(
        self,
        projected_solutions: NDArray,
        forcing_fn: Callable | None = None,
    ) -> None:
        """Pre-populate 3-level history from projected snapshots [t^{n-2}, t^{n-1}].

        Seeds history with 3 entries (duplicating t^{n-2} as t^{n-3}) so the
        3-level check in _compute_sgsp_contribution passes immediately.
        """
        assert len(projected_solutions) == 2, "Expected exactly 2 seed snapshots"

        u_nm2 = projected_solutions[0].copy()
        u_nm1 = projected_solutions[1].copy()

        du_dt_nm2 = np.zeros(self.n_nodes)
        du_dt_nm1 = (u_nm1 - u_nm2) / self.dt

        f_nm2 = (
            forcing_fn(self.mesh, -2 * self.dt)
            if forcing_fn
            else np.zeros(self.n_nodes)
        )
        f_nm1 = (
            forcing_fn(self.mesh, -self.dt) if forcing_fn else np.zeros(self.n_nodes)
        )

        # 3 entries needed: duplicate u_nm2 as the oldest level
        self._u_bar_history = [u_nm2.copy(), u_nm2, u_nm1]
        self._du_bar_dt_history = [du_dt_nm2.copy(), du_dt_nm2, du_dt_nm1]
        self._forcing_history = [f_nm2.copy(), f_nm2, f_nm1]
        self._step_count = self._sgsp_warmup_steps

    # ------------------------------------------------------------------ #
    #  Blow-up buffer: save to disk
    # ------------------------------------------------------------------ #

    def _save_blowup_buffer(self, label: str = "blowup") -> Path:
        """Flush blow-up buffer to .npz archive and .txt summary in master_path."""
        buf = self._blowup_buffer
        output_dir: Path = self.master_path
        termination_step_idx = len(buf) - 1
        termination_time = buf.time_values[-1] if buf.time_values else float("nan")

        npz_path = output_dir / f"buffer_{label}_{self.run_id}.npz"
        np.savez_compressed(
            npz_path,
            time_values=np.array(buf.time_values),
            amplitude_snapshots=np.array(buf.amplitude_snapshots),
            energy_spectra=np.array(buf.energy_spectra),
            dissipation_values=np.array(buf.dissipation_values),
            artificial_viscosity_values=np.array(buf.artificial_viscosity_values),
            energy_values=np.array(buf.energy_values),
            residual_norms_first=np.array(buf.residual_norms_first),
            residual_norms_last=np.array(buf.residual_norms_last),
            update_norms_last=np.array(buf.update_norms_last),
            corrector_pass_counts=np.array(buf.corrector_pass_counts),
            blowup_step_idx=termination_step_idx,
            blowup_time=termination_time,
        )
        logger.info(
            "Buffer saved to %s (%d steps, label=%s)", npz_path, len(buf), label
        )

        txt_path = output_dir / f"buffer_{label}_{self.run_id}.txt"
        final_amp = buf.amplitude_snapshots[-1] if buf.amplitude_snapshots else None
        lines = [
            f"Run ID             : {self.run_id}",
            f"Label              : {label}",
            f"Steps in buffer    : {len(buf)}",
            f"Termination step   : {termination_step_idx}  (t = {termination_time:.6f})",
            f"Max |u| at end     : {np.max(np.abs(final_amp)):.4e}"
            if final_amp is not None
            else "Max |u| at end     : N/A",
            f"NaN in u           : {bool(np.any(np.isnan(final_amp)))}"
            if final_amp is not None
            else "",
            f"Inf in u           : {bool(np.any(np.isinf(final_amp)))}"
            if final_amp is not None
            else "",
            f"Final residual norm: {buf.residual_norms_last[-1]:.4e}"
            if buf.residual_norms_last
            else "",
            f"Final dissipation  : {buf.dissipation_values[-1]:.4e}"
            if buf.dissipation_values
            else "",
            "",
            "--- Solver config ---",
            f"  n_nodes       : {self.n_nodes}",
            f"  viscosity     : {self.viscosity}",
            f"  dt            : {self.dt}",
            f"  clip_pusuluri : {self.clip_pusuluri}",
            f"  clip_rajampeta: {self.clip_rajampeta}",
            f"  blowup_thr    : {self._blowup_threshold:.2e}",
        ]
        txt_path.write_text("\n".join(line for line in lines if line is not None))
        logger.info("Buffer summary written to %s", txt_path)

        if label == "blowup":
            self._save_final_steps_as_csv(n_steps=5)

        return npz_path

    def _save_final_steps_as_csv(self, n_steps: int = 5) -> None:
        """Write the last n_steps buffered snapshots as individual CSVs for inspection."""
        buf = self._blowup_buffer
        n_to_write = min(n_steps, len(buf))

        for offset in range(n_to_write - 1, -1, -1):
            buf_idx = len(buf) - 1 - offset
            time_val = buf.time_values[buf_idx]
            amplitudes = buf.amplitude_snapshots[buf_idx]
            filepath = self.master_path / f"final_step_{offset:02d}_t{time_val:.6f}.csv"
            with filepath.open("w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        "node_index",
                        "x_coordinate",
                        "velocity",
                        "residual_norm_last",
                        "energy",
                    ]
                )
                for node_idx in range(self.n_nodes):
                    writer.writerow(
                        [
                            node_idx,
                            round(self.mesh[node_idx], 8),
                            amplitudes[node_idx],
                            buf.residual_norms_last[buf_idx],
                            buf.energy_values[buf_idx],
                        ]
                    )

        logger.info("Saved %d final-step CSVs to %s", n_to_write, self.master_path)

    def _write_blowup_solutions_to_csv(self) -> None:
        """Write only the snapshots collected before blow-up, skipping NaN padding."""
        if self.requested_snapshots is None:
            return
        self.write_solution_to_csv()
        print(
            f"wrote {len(self.requested_snapshots)} pre-blowup snapshots at {self.master_path}"
        )

    def print_configuration(self) -> None:
        """Print base config plus SGSP-specific settings."""
        super().print_configuration()
        W = 72
        COL = 30

        def _row(label: str, value: str) -> None:
            print(f"  {label:<{COL}} {value}")

        print()
        print("  SGS Predictor")
        print("─" * W)
        _row("model path", str(self._sgsp_model_path))
        _row("normalisation stats", str(self._normalization_path))
        _row("warmup steps", str(self._sgsp_warmup_steps))
        _row("blowup threshold", f"{self._blowup_threshold:.2e}")
        _row("blowup buffer size", str(self._blowup_buffer.max_steps))
        _row("blown_up path", str(self._blown_up_path))
        print()
        print("  Clipping")
        print("─" * W)
        _row("clip_pusuluri", str(self.clip_pusuluri))
        _row("clip_rajampeta", str(self.clip_rajampeta))
        print("═" * W)

    # ------------------------------------------------------------------ #
    #  Post-processing
    # ------------------------------------------------------------------ #

    def post_processing(self) -> None:
        """Standard post-processing plus blow-up diagnostics if blown up."""
        super().post_processing()
        if self.master_path == self._blown_up_path:
            self.plot_blowup_diagnostics()

    def plot_blowup_diagnostics(
        self,
        pre_blowup_window: int = 80,
        trim_tail: int = 30,
        show_plot: bool = False,
    ) -> None:
        """Generate two diagnostic plots centred on the blow-up region.

        Writes blowup_zoom_full_<run_id>.png and blowup_zoom_trim_<run_id>.png.
        """
        """Generate two diagnostic plots centred on the blow-up region."""
        if not self.residual_history:
            return
        n_total = len(self.energy_history)  # ← was len(self.time_steps)
        slice_start = max(0, n_total - pre_blowup_window)
        for plot_slice, tag in (
            (slice(slice_start, n_total), f"blowup_zoom_full_{self.run_id}"),
            (
                slice(slice_start, max(slice_start, n_total - trim_tail)),
                f"blowup_zoom_trim_{self.run_id}",
            ),
        ):
            self._plot_blowup_window(
                plot_slice=plot_slice, filename_tag=tag, show_plot=show_plot
            )

    def _plot_blowup_window(
        self,
        plot_slice: slice,
        filename_tag: str,
        show_plot: bool = False,
    ) -> None:
        """Render and save one diagnostic plot for a sliced history window."""
        n_steps_actual = len(self.energy_history)
        s_start, s_stop, _ = plot_slice.indices(n_steps_actual)

        time_steps_sliced = self.time_steps[s_start:s_stop]
        energy_sliced = self.energy_history[s_start:s_stop]
        dissipation_sliced = self.dissipation_history[s_start:s_stop]

        if len(time_steps_sliced) == 0:
            logger.warning(
                "_plot_blowup_window: empty window [%d:%d], skipping.", s_start, s_stop
            )
            return

        buf = self._blowup_buffer
        solution_snapshot = (
            buf.amplitude_snapshots[s_stop - 1]
            if buf.amplitude_snapshots and s_stop <= len(buf)
            else self.solution
        )
        wn, sp = self.get_positive_spectrum(
            *self.compute_energy_spectrum(solution_snapshot)
        )

        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 2)

        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(
            self.mesh,
            solution_snapshot,
            color="royalblue",
            linestyle="-",
            marker="o",
            label="Resolved solution",
        )
        ax0.plot(
            self.mesh,
            self.initial_condition,
            color="grey",
            linestyle="--",
            label="Initial solution",
        )
        ax0.set_xlabel(r"$x$")
        ax0.set_ylabel("Velocity")
        ax0.grid(True)
        ax0.legend()
        step_range = (
            f"steps {time_steps_sliced[0]}–{time_steps_sliced[-1]}"
            if len(time_steps_sliced) > 1
            else f"step {time_steps_sliced[0]}"
        )
        ax0.set_title(f"Solution at window end [{step_range}] [SGS: LES-ANN]")

        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(time_steps_sliced, energy_sliced, color="red", label="Total energy")
        ax1.plot(
            time_steps_sliced, dissipation_sliced, color="purple", label="Dissipation"
        )
        ax1.set_xlabel("Time step")
        ax1.set_title("Energy and dissipation (window)")
        ax1.grid(True)
        ax1.legend()

        ax2 = fig.add_subplot(gs[1, 1])
        ax2.loglog(wn[1:], sp[1:], marker="o")
        ax2.set_xlabel("Wavenumber k")
        ax2.set_ylabel("E(k)")
        ax2.set_title("Spectral analysis (window end)")
        ax2.grid(True)

        plt.tight_layout()
        save_path = self.master_path / f"{filename_tag}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Blow-up diagnostic plot saved to %s", save_path)
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
