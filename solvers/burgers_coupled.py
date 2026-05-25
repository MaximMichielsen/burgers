"""Coupled Burgers FEM solver: Galerkin + ANN-predicted SGS correction.

Inherits from BurgersPure and overrides only the parts that change:

    1. advance_time_step  — maintains the lagged solution history buffer
                            (u_bar at t^n, t^{n-1}, t^{n-2}) needed by the
                            LFS input stencil, and records per-step diagnostics
                            into the blow-up buffer.
    2. nr_iteration       — builds the ANN input stencil once per NR call
                            (outside the element loop; stencil is frozen at
                            lagged time levels, so it does not change during
                            NR iterations), then adds the predicted interaction
                            terms to the global residual after assembly.
    3. run_simulation     — wraps the parent loop in a try/finally so that
                            all buffered history is written to disk whenever
                            the solver terminates, whether by blow-up or by
                            a normal end-of-run.

The Jacobian is left unchanged (pure Galerkin).  This is correct because the
LFS ensures ANN inputs are strictly from previous time levels, meaning the ANN
output has no dependence on the current corrector-pass iterate u^k.  Adding
ANN terms to the Jacobian would be incorrect and would require a full
algorithmic differentiation of the network.

Residual modification (signs follow the VMS two-scale formulation):
    R_global += −(w_x, ū·u')_e    [cross]
    R_global += −(w_x, u'²/2)_e   [Reynolds]
    R_global[left]  += (w_l, u'_t)_e   [temporal, left node]
    R_global[right] += (w_r, u'_t)_e   [temporal, right node]
    R_global += −ν · (w_x, u'_x)_e    [viscous SGS]

The signs come from the interaction terms appearing on the LHS of the weak
form with a minus sign:
    (w, u_t) + (w, u·u_x) − ν(w, u_xx) − (interaction terms) − (w, f) = 0

Blow-up saving
--------------
A ``_BlowupBuffer`` (private inner dataclass) accumulates per-step snapshots
of every quantity needed for:

  * **Diagnosis** — full solution amplitudes, residual/update norms,
    corrector-pass counts, energy, dissipation.
  * **RL corrector training** — the exact state vector ``sn`` from the
    research proposal (Eq. 2.8):

        sn = [Ê_1, …, Ê_K, ε̄^n, α_{n-1}]

    where Ê_k = E(k,t) / E_DNS(k) are normalised spectral energies and ε̄^n
    is the instantaneous resolved dissipation rate.  Because the DNS target
    is not available at run-time, raw E(k,t) and ε̄^n are stored; the
    normalisation is left for the offline training pipeline.

The buffer is written to a compressed ``.npz`` archive (and a human-readable
``.txt`` summary) whenever the simulation ends, regardless of whether the
termination was clean or caused by a blow-up.

Output path structure
---------------------
The solver starts writing into ``stable/`` (set as ``master_path`` by
``create_ann_config``).  On blow-up, ``master_path`` is redirected to
``blown_up/`` before the buffer is flushed, so all output — CSVs, log,
archive — lands in the correct sub-directory automatically.

    solver_data/LES_ANN/
      unclipped/   (or pusuluri/ or rajampeta/)
        stable/      ← clean run output
        blown_up/    ← blow-up run output

References
----------
Robijns (2019), Section 3.2 / 5.
Pusuluri (2021), Section 3.3.
Rajampeta (2022), Section 4.3–4.4.
Research Proposal, Section 2.3 (state vector definition, Eq. 2.8).
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from constants import OUTPUT_UNITS
from solvers.burgers_pure import BurgersPure
from data_curation.training_data_assembly import (
    build_input_stencil,
    _gradient_basis_functions,
)
from ml_agents.predictor import SGSPredictor, load_predictor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blow-up threshold — tune to your problem's typical amplitude scale.
# For the forced 1-D Burgers equation with O(1) initial amplitudes this is
# conservative; lower it (e.g. 1e3) if you want earlier detection.
# ---------------------------------------------------------------------------
_BLOWUP_AMP_THRESHOLD: float = 1e4


# ---------------------------------------------------------------------------
# Blow-up history buffer (private — not part of the public API)
# ---------------------------------------------------------------------------


@dataclass
class _BlowupBuffer:
    """Per-step snapshot buffer used for post-blow-up diagnostics and RL data.

    Fields are kept as plain Python lists and converted to arrays only on
    save, avoiding repeated allocations during the hot time loop.

    Attributes
    ----------
    max_steps:
        Rolling window size.  Older entries are dropped once the buffer is
        full, keeping memory bounded for long runs.
    """

    max_steps: int = 5_000

    # ── simulation state ────────────────────────────────────────────────
    time_values: list[float] = field(default_factory=list)
    amplitude_snapshots: list[NDArray] = field(default_factory=list)

    # ── RL state components (Research Proposal Eq. 2.8) ─────────────────
    # raw E(k, t) per step — normalise by E_DNS offline
    energy_spectra: list[NDArray] = field(default_factory=list)
    # ε̄^n: instantaneous resolved dissipation rate
    dissipation_values: list[float] = field(default_factory=list)
    # α_{n-1}: artificial viscosity applied at this step (0 for unclipped)
    artificial_viscosity_values: list[float] = field(default_factory=list)

    # ── diagnostics ─────────────────────────────────────────────────────
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
        """Append one timestep; prune oldest entry when buffer is full."""
        if len(self.time_values) >= self.max_steps:
            self.time_values.pop(0)
            self.amplitude_snapshots.pop(0)
            self.energy_spectra.pop(0)
            self.dissipation_values.pop(0)
            self.artificial_viscosity_values.pop(0)
            self.energy_values.pop(0)
            self.residual_norms_first.pop(0)
            self.residual_norms_last.pop(0)
            self.update_norms_last.pop(0)
            self.corrector_pass_counts.pop(0)

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


# ---------------------------------------------------------------------------
# Coupled solver
# ---------------------------------------------------------------------------


class BurgersCoupled(BurgersPure):
    """Burgers FEM solver with ANN-predicted SGS closure.

    Extends BurgersPure by injecting the trained SGS predictor into the
    global residual at each Newton–Raphson call.

    Additional Parameters (passed via configuration)
    -------------------------------------------------
    ann_model_path : Path | str
        Path to the saved ``sgs_predictor.pt`` checkpoint.
    normalisation_stats_path : Path | str
        Path to ``normalisation_stats.npz`` produced during training.
    ann_warmup_steps : int, optional
        Number of time steps to run pure Galerkin before activating the ANN.
        During warm-up, the lagged buffer is populated but no correction is
        applied.  Default: 2 (minimum needed to fill the 3-level stencil).
    blowup_threshold : float, optional
        Amplitude magnitude above which blow-up is declared.  Default: 1e4.
    blowup_buffer_size : int, optional
        Number of past steps retained in the blow-up buffer.  Default: 5000.
    blown_up_path : str, optional
        Directory the solver redirects ``master_path`` to on blow-up.
        Falls back to ``master_path`` (i.e. stable/) if absent.
    """

    def __init__(
        self,
        configuration: dict,
        clip_pusuluri: bool = False,
        clip_rajampeta: bool = False,
        exclude_visc: bool = True,
        sigma_multiplier: float = 3.0,
    ) -> None:
        super().__init__(configuration)

        # --- Load ANN model ---
        ann_model_path = Path(configuration["ann_model_path"])
        self._predictor: SGSPredictor = load_predictor(ann_model_path)
        self._predictor.eval()

        # --- Load normalisation statistics ---
        norm_stats_path = Path(configuration["normalisation_stats_path"])
        norm_data = np.load(norm_stats_path)
        self._x_mean: NDArray = norm_data["X_mean"].astype(np.float32)
        self._x_std: NDArray = norm_data["X_std"].astype(np.float32)
        self._y_mean: NDArray = norm_data["y_mean"].astype(np.float32)
        self._y_std: NDArray = norm_data["y_std"].astype(np.float32)

        # --- Lagged solution history (LFS: 3 time levels) ---
        # Populated in advance_time_step; used to build the ANN input stencil.
        self._u_bar_history: list[NDArray] = []
        self._du_bar_dt_history: list[NDArray] = []
        self._forcing_history: list[NDArray] = []

        # Warm-up: run pure Galerkin for the first few steps to fill the buffer
        self._ann_warmup_steps: int = int(configuration.get("ann_warmup_steps", 2))
        self._step_count: int = 0

        # Gradient of basis functions (constant for uniform mesh)
        self._grad_basis: NDArray = _gradient_basis_functions(self.element_size)

        self.clip_pus: bool = clip_pusuluri
        self.clip_raj: bool = clip_rajampeta
        self.exclude_visc: bool = exclude_visc

        if self.clip_raj and not self.clip_pus:
            raise ValueError(
                "If clipping using Rajampeta's, set Pusuluri's also to True."
            )

        # Pusuluri clipping
        if self.clip_pus:
            self._y_lower_bound = self._y_mean - sigma_multiplier * self._y_std
            self._y_upper_bound = self._y_mean + sigma_multiplier * self._y_std

        # --- Blow-up detection & history buffer ---
        self._blowup_threshold: float = float(
            configuration.get("blowup_threshold", _BLOWUP_AMP_THRESHOLD)
        )
        self._blowup_buffer: _BlowupBuffer = _BlowupBuffer(
            max_steps=int(configuration.get("blowup_buffer_size", 5_000))
        )
        # Tracks artificial viscosity applied at the previous step.
        # Always 0.0 for unclipped runs; non-zero once the RL corrector is
        # active.  Stored so the RL state vector α_{n-1} is always available.
        self._artificial_viscosity_prev: float = 0.0

        # Output path the solver redirects to on blow-up (blown_up/ sibling of
        # stable/).  Falls back to master_path so the class is safe without it.
        blown_up_str: str | None = configuration.get("blown_up_path")
        self._blown_up_path: Path = (
            Path(blown_up_str) if blown_up_str is not None else self.master_path
        )

    # ------------------------------------------------------------------
    # Public factory helper — mirrors BurgersPure.create_config
    # ------------------------------------------------------------------

    @staticmethod
    def create_coupled_config(
        ann_model_path: str | Path,
        normalisation_stats_path: str | Path,
        ann_warmup_steps: int = 2,
        blowup_threshold: float = _BLOWUP_AMP_THRESHOLD,
        blowup_buffer_size: int = 5_000,
        blown_up_path: str | None = None,
        **base_config_kwargs,
    ) -> dict:
        """Build a coupled-solver configuration dict.

        Calls BurgersPure.create_config with *base_config_kwargs* and
        appends the ANN-specific keys.

        Parameters
        ----------
        ann_model_path:
            Path to the saved ``sgs_predictor.pt`` checkpoint.
        normalisation_stats_path:
            Path to ``normalisation_stats.npz``.
        ann_warmup_steps:
            Pure-Galerkin steps before ANN activation.
        blowup_threshold:
            Amplitude magnitude above which blow-up is declared.
        blowup_buffer_size:
            Rolling window size for the blow-up history buffer.
        blown_up_path:
            Directory the solver redirects output to on blow-up.
            Falls back to ``master_path`` if None.
        **base_config_kwargs:
            Forwarded verbatim to ``BurgersPure.create_config``.
        """
        base_config = BurgersPure.create_config(**base_config_kwargs)
        base_config["simulation_mode"] = "ann"
        base_config["ann_model_path"] = str(ann_model_path)
        base_config["normalisation_stats_path"] = str(normalisation_stats_path)
        base_config["ann_warmup_steps"] = ann_warmup_steps
        base_config["blowup_threshold"] = blowup_threshold
        base_config["blowup_buffer_size"] = blowup_buffer_size
        if blown_up_path is not None:
            base_config["blown_up_path"] = str(blown_up_path)
        return base_config

    # ------------------------------------------------------------------
    # Override: run_simulation — add blow-up guard around parent loop
    # ------------------------------------------------------------------

    def run_simulation(self) -> None:
        """Run the full simulation with guaranteed post-blow-up data saving.

        Wraps the parent time loop in a try/finally so that all buffered
        history is flushed to disk regardless of how the simulation ends
        (clean finish, detected blow-up, or unexpected exception).

        On blow-up, ``master_path`` is redirected to ``_blown_up_path``
        before the buffer is saved, so the archive, log, and CSV snapshots
        all land in ``blown_up/`` rather than ``stable/``.
        """
        blowup_detected: bool = False

        try:
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
                    for time_step_idx in range(total_steps):
                        step_start = perf_counter()
                        self.time_steps.append(time_step_idx)

                        # Advance Time Step
                        # -----------------
                        self.advance_time_step()
                        # -----------------

                        idx_extract = self._maybe_extract_solution(idx_extract)

                        step_time = perf_counter() - step_start
                        pbar.set_description(
                            f"Eating Burgers | {self.throbber(time_step_idx)}"
                        )
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "t": f"{self.simulation_time_elapsed:.3f}",
                                "dt": f"{self.dt:.3f}",
                                "step_time": f"{step_time:.3f}s",
                            }
                        )

                # End-of-run extraction flush (parent logic)
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

        except RuntimeError as exc:
            # RuntimeError is raised by advance_time_step when blow-up is
            # detected before the NR step.  This is the expected termination
            # path for diverging runs.
            # time_steps was appended before the raise, but energy_history /
            # dissipation_history were never updated (super() never ran) —
            # pop the dangling entry so all history lists stay the same length
            # and post_plotting does not raise a shape mismatch.
            if self.time_steps:
                self.time_steps.pop()
            blowup_detected = True
            logger.warning("Blow-up termination: %s", exc)

        except Exception as exc:
            # Unexpected crash (e.g. singular Jacobian that slipped through,
            # OOM, etc.).  Also redirect to blown_up/ since the run is invalid.
            blowup_detected = True
            logger.error(
                "Solver raised an unexpected exception: %s. "
                "Saving buffered data before re-raise.",
                exc,
            )
            raise

        finally:
            # Redirect master_path FIRST so every subsequent write — CSVs,
            # config JSON, buffer archive, log — lands in the correct folder.
            if blowup_detected:
                self.master_path = self._blown_up_path

            if self.write_solutions:
                self.write_config_to_json()
                self.write_solution_to_csv()

            if len(self._blowup_buffer) > 0:
                label = "blowup" if blowup_detected else "clean"
                self._save_blowup_buffer(label=label)

    # ------------------------------------------------------------------
    # Override: advance_time_step — maintain lagged history + record step
    # ------------------------------------------------------------------

    def advance_time_step(self) -> None:
        """Advance one time step, update the lagged solution history, and
        record per-step diagnostics into the blow-up buffer.

        Blow-up is checked against the *current* solution before handing off
        to the NR solver.  This is intentional: once amplitudes cross the
        threshold the Jacobian is typically ill-conditioned and
        ``np.linalg.solve`` will raise or return NaN, bypassing the
        post-step check in ``run_simulation`` entirely.  Detecting here
        raises ``RuntimeError`` so the ``except`` block in ``run_simulation``
        catches it cleanly and routes output to ``blown_up/``.
        """
        if self._detect_blowup(self.solution):
            raise RuntimeError(
                f"Blow-up before NR step at t={self.simulation_time_elapsed:.6f}: "
                f"max|u|={np.max(np.abs(self.solution)):.4e} "
                f"> threshold {self._blowup_threshold:.4e}"
            )

        super().advance_time_step()

        if len(self._u_bar_history) == 0:
            # First step: push IC twice to pre-fill t^{n-1} and t^{n-2} slots.
            self._u_bar_history.append(self.initial_condition.copy())
            self._u_bar_history.append(self.initial_condition.copy())
            self._du_bar_dt_history.append(np.zeros(self.n_nodes))
            self._du_bar_dt_history.append(np.zeros(self.n_nodes))

            forcing_append = (
                self.forcing_current.copy()
                if self.forcing_current is not None
                else np.zeros(self.n_nodes)
            )
            self._forcing_history.append(forcing_append)
            self._forcing_history.append(forcing_append)

        # Push the just-completed solution as the newest lagged level.
        u_bar_new = self.solution.copy()
        u_bar_prev = self._u_bar_history[-1]
        du_bar_dt_new = (u_bar_new - u_bar_prev) / self.dt

        self._u_bar_history.append(u_bar_new)
        self._du_bar_dt_history.append(du_bar_dt_new)
        self._forcing_history.append(
            self.forcing_current.copy()
            if self.forcing_current is not None
            else np.zeros(self.n_nodes)
        )

        # Keep only 3 levels.
        if len(self._u_bar_history) > 3:
            self._u_bar_history.pop(0)
            self._du_bar_dt_history.pop(0)
            self._forcing_history.pop(0)

        self._step_count += 1

        # ── Record step into blow-up buffer ────────────────────────────
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
            residual_norm_first=float(res_step[0]) if res_step else 0.0,
            residual_norm_last=float(res_step[-1]) if res_step else 0.0,
            update_norm_last=float(upd_step[-1]) if upd_step else 0.0,
            corrector_passes=len(res_step),
        )

    # ------------------------------------------------------------------
    # Override: nr_iteration — inject ANN correction into residual
    # ------------------------------------------------------------------

    def nr_iteration(self, solution: NDArray) -> NDArray:
        """Newton–Raphson with ANN correction added to the global residual.

        The ANN term is frozen at the lagged state (LFS) and does not change
        between NR iterations, so it is computed once and cached.
        """
        ann_correction_per_element: NDArray | None = self._compute_ann_correction()

        solution_n = solution.copy()
        solution_k = solution.copy()
        residual_history_loop: list = []
        update_history_loop: list = []

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

            if ann_correction_per_element is not None:
                global_residual = self._add_ann_correction_to_residual(
                    global_residual, ann_correction_per_element
                )

            global_residual, global_jacobian = self._apply_boundary_conditions(
                global_residual, global_jacobian, solution_k
            )

            residual_history_loop.append(np.linalg.norm(global_residual))

            with self.timer("linear_solve"):
                delta_u = np.linalg.solve(global_jacobian, -global_residual)
                if self.boundary_condition_type == "periodic":
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
    # ANN helpers
    # ------------------------------------------------------------------

    def _compute_ann_correction(self) -> NDArray | None:
        """Build input stencils for all elements and run a batched ANN forward pass.

        Returns
        -------
        ann_correction : NDArray of shape (n_elements, 5), or None during warm-up.
            Physical-space predicted interaction terms per element.
            Column ordering: [cross, Reynolds, u't_L, u't_R, viscous].
        """
        if self._step_count < self._ann_warmup_steps:
            return None
        if len(self._u_bar_history) < 3:
            return None

        n_elements = self.n_elements
        input_rows: list[NDArray] = []
        valid_element_indices: list[int] = []

        for elem_idx in range(n_elements):
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
            x_tensor = torch.from_numpy(x_batch_norm)
            y_norm_tensor = self._predictor(x_tensor)
            y_norm = y_norm_tensor.numpy()

        y_phys = y_norm * self._y_std + self._y_mean  # (n_valid, 5)

        if self.clip_pus:
            y_phys = np.clip(y_phys, self._y_lower_bound, self._y_upper_bound)

        ann_correction_all = np.zeros((n_elements, OUTPUT_UNITS), dtype=np.float64)
        for local_idx, elem_idx in enumerate(valid_element_indices):
            y_elem = y_phys[local_idx].copy()

            if self.clip_raj:
                node_left = elem_idx
                node_right = elem_idx + 1
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

            if self.exclude_visc:
                y_elem[4] = 0.0

            ann_correction_all[elem_idx] = y_elem

        return ann_correction_all

    def _add_ann_correction_to_residual(
        self,
        global_residual: NDArray,
        ann_correction: NDArray,
    ) -> NDArray:
        """Scatter per-element ANN predictions into the global residual.

        Sign convention (interaction terms move to LHS with minus):
            R[left]  -= cross_term + Reynolds_term - temporal_left  - viscous
            R[right] -= cross_term + Reynolds_term - temporal_right - viscous

        More precisely, for element e with left node i and right node i+1:

            The cross and Reynolds terms integrate against w_x (constant over
            element), contributing equally to both nodes with the sign of
            grad_basis[0] = -1/h (left) and grad_basis[1] = +1/h (right).
            However, since compute_element_output_terms stores the raw
            integral with w_x = +1/h (grad_basis[1]), we reconstruct the
            nodal contributions:

                left  contribution = -ann[e, col] * (-1) = +ann[e, col]
                right contribution = -ann[e, col] * (+1) = -ann[e, col]

            Wait — this needs care.  The stored integral is:
                (w_x, q)_e = ∫ (1/h) * q dx   [used grad_basis[1] = +1/h]

            But the actual weak form uses both shape functions:
                w_l: grad = -1/h,  w_r: grad = +1/h

            So the correct nodal residual contributions are:
                R[i]   += -(-1/h) * raw_integral * h  = +raw_integral
                R[i+1] += -(+1/h) * raw_integral * h  = -raw_integral

            where the outer minus comes from the interaction terms being
            subtracted from the residual (they appear as −IT on the LHS).

            For temporal terms (integrated against w_l, w_r directly):
                R[i]   -= temporal_left
                R[i+1] -= temporal_right

        Column indices: 0=cross, 1=Reynolds, 2=temporal_L, 3=temporal_R, 4=viscous
        """
        if not np.all(np.isfinite(ann_correction)):
            return global_residual  # skip correction entirely this step

        residual_modified = global_residual.copy()

        for elem_idx, element in enumerate(self.elements):
            node_left, node_right = int(element[0]), int(element[1])

            cross_val = ann_correction[elem_idx, 0]
            reynolds_val = ann_correction[elem_idx, 1]
            temporal_left_val = ann_correction[elem_idx, 2]
            temporal_right_val = ann_correction[elem_idx, 3]
            viscous_val = ann_correction[elem_idx, 4]

            residual_modified[node_left] += cross_val + reynolds_val + viscous_val
            residual_modified[node_right] -= cross_val + reynolds_val + viscous_val
            residual_modified[node_left] -= temporal_left_val
            residual_modified[node_right] -= temporal_right_val

        return residual_modified

    # ------------------------------------------------------------------
    # Blow-up detection
    # ------------------------------------------------------------------

    def _detect_blowup(self, solution_amplitudes: NDArray) -> bool:
        """Return True if the solution has blown up.

        Checks for NaN, Inf, or amplitude exceeding ``_blowup_threshold``.
        """
        return bool(
            np.any(np.isnan(solution_amplitudes))
            or np.any(np.isinf(solution_amplitudes))
            or np.max(np.abs(solution_amplitudes)) > self._blowup_threshold
        )

    # ------------------------------------------------------------------
    # Blow-up buffer: save to disk
    # ------------------------------------------------------------------

    def _save_blowup_buffer(self, label: str = "blowup") -> Path:
        """Flush the blow-up buffer to ``master_path/buffer_<label>_<run_id>.npz``.

        Two files are always written:
        - ``.npz`` — compressed archive with all array data; load with
          ``np.load(path, allow_pickle=False)``.
        - ``.txt`` — human-readable summary for quick diagnosis.

        The ``.npz`` keys directly match the RL state-vector definition
        (Research Proposal Eq. 2.8) so the offline training pipeline can
        consume the file without further transformation:

            time_values                 — t^n for each recorded step
            amplitude_snapshots         — ū_h(t^n)    [n_steps × n_nodes]
            energy_spectra              — E(k, t^n)   [n_steps × n_pos_modes]
            dissipation_values          — ε̄^n          [n_steps]
            artificial_viscosity_values — α_{n-1}     [n_steps]
            energy_values               — ½∫u² dx      [n_steps]
            residual_norms_first        — ||R||_0      [n_steps]
            residual_norms_last         — ||R||_final  [n_steps]
            update_norms_last           — ||Δu||_final [n_steps]
            corrector_pass_counts       — NR iters     [n_steps]
            blowup_step_idx             — index of termination step (scalar)
            blowup_time                 — t at termination (scalar)

        Returns the path to the ``.npz`` file.
        """
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
            "Buffer saved to %s (%d steps, label=%s)",
            npz_path,
            len(buf),
            label,
        )

        txt_path = output_dir / f"buffer_{label}_{self.run_id}.txt"
        with txt_path.open("w") as txt_file:
            txt_file.write(f"Run ID             : {self.run_id}\n")
            txt_file.write(f"Label              : {label}\n")
            txt_file.write(
                f"Steps in buffer    : {len(buf)}\n"
                f"  (all steps stored in .npz; CSVs are separate extraction\n"
                f"   checkpoints from extract_at_times, not one-per-step)\n"
            )
            txt_file.write(
                f"Termination step   : {termination_step_idx}"
                f"  (t = {termination_time:.6f})\n"
            )
            if buf.amplitude_snapshots:
                final_amp = buf.amplitude_snapshots[-1]
                txt_file.write(
                    f"Max |u| at end     : {np.max(np.abs(final_amp)):.4e}\n"
                )
                txt_file.write(f"NaN in u           : {np.any(np.isnan(final_amp))}\n")
                txt_file.write(f"Inf in u           : {np.any(np.isinf(final_amp))}\n")
            if buf.residual_norms_last:
                txt_file.write(
                    f"Final residual norm: {buf.residual_norms_last[-1]:.4e}\n"
                )
            if buf.dissipation_values:
                txt_file.write(
                    f"Final dissipation  : {buf.dissipation_values[-1]:.4e}\n"
                )
            txt_file.write("\n--- Solver config ---\n")
            txt_file.write(f"  n_nodes      : {self.n_nodes}\n")
            txt_file.write(f"  viscosity    : {self.viscosity}\n")
            txt_file.write(f"  dt           : {self.dt}\n")
            txt_file.write(f"  clip_pus     : {self.clip_pus}\n")
            txt_file.write(f"  clip_raj     : {self.clip_raj}\n")
            txt_file.write(f"  exclude_visc : {self.exclude_visc}\n")
            txt_file.write(f"  blowup_thr   : {self._blowup_threshold:.2e}\n")

        logger.info("Buffer summary written to %s", txt_path)

        if label == "blowup":
            self._save_final_steps_as_csv(n_steps=5)

        return npz_path

    def _save_final_steps_as_csv(self, n_steps: int = 5) -> None:
        """Write the last ``n_steps`` buffered snapshots as individual CSVs.

        These are always written on blow-up so the pre-blow-up solution
        trajectory is available for human inspection without loading the full
        ``.npz``.  Files are named ``final_step_<offset>_t<time>.csv`` where
        ``offset`` counts back from the last recorded step (0 = last, 1 =
        second-to-last, etc.).

        Format matches the standard ``write_solution_to_csv`` output:
            node_index, x_coordinate, velocity, energy_spectrum_magnitude
        """
        buf = self._blowup_buffer
        output_dir: Path = self.master_path
        n_available = len(buf)
        n_to_write = min(n_steps, n_available)

        for offset in range(n_to_write - 1, -1, -1):
            # offset=n_to_write-1 is the oldest of the last-N; offset=0 is last
            buf_idx = n_available - 1 - offset
            time_val = buf.time_values[buf_idx]
            amplitudes = buf.amplitude_snapshots[buf_idx]
            filename = f"final_step_{offset:02d}_t{time_val:.6f}.csv"
            filepath = output_dir / filename

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
                            round(self.node_coords[node_idx], 8),
                            amplitudes[node_idx],
                            buf.residual_norms_last[buf_idx],
                            buf.energy_values[buf_idx],
                        ]
                    )

        logger.info(
            "Saved %d final-step CSVs to %s (offset 00 = last recorded step)",
            n_to_write,
            output_dir,
        )

    # ------------------------------------------------------------------
    # Override: post_processing — add blow-up diagnostics on blown-up runs
    # ------------------------------------------------------------------

    def post_processing(self) -> None:
        """Run standard post-processing and, on blow-up, add diagnostic plots."""
        super().post_processing()
        if self.master_path == self._blown_up_path:
            self.plot_blowup_diagnostics()

    # ------------------------------------------------------------------
    # Blow-up diagnostic plots
    # ------------------------------------------------------------------

    def plot_blowup_diagnostics(
        self,
        pre_blowup_window: int = 80,
        trim_tail: int = 30,
        show_plot: bool = False,
    ) -> None:
        """Generate two diagnostic plots centred on the blow-up region.

        Only called when the run blew up (no-op otherwise).  Both plots use
        the same layout as ``BurgersPure.post_plotting`` but operate on a
        sliced view of the history so the relevant dynamics are visible.

        Parameters
        ----------
        pre_blowup_window:
            Number of time steps before the last recorded step to include.
            Default 100 — shows the onset of instability without the full run.
        trim_tail:
            Number of steps removed from the *end* for the second (clean) plot.
            Default 15 — removes the extreme blow-up spike so convergence
            history is readable.
        show_plot:
            Whether to call ``plt.show()``.  Default False.

        Output files (written to ``master_path``)
        ------------------------------------------
        ``blowup_zoom_full_<run_id>.png``   — last ``pre_blowup_window`` steps
        ``blowup_zoom_trim_<run_id>.png``   — same window, last ``trim_tail``
                                              steps removed
        """
        if not self.residual_history:
            return

        n_total = len(self.time_steps)
        # Slice indices into the per-timestep history lists
        slice_start = max(0, n_total - pre_blowup_window)
        slice_full = slice(slice_start, n_total)
        slice_trim = slice(slice_start, max(slice_start, n_total - trim_tail))

        for plot_slice, filename_tag in (
            (slice_full, f"blowup_zoom_full_{self.run_id}"),
            (slice_trim, f"blowup_zoom_trim_{self.run_id}"),
        ):
            self._plot_blowup_window(
                plot_slice=plot_slice,
                filename_tag=filename_tag,
                show_plot=show_plot,
            )

    def _plot_blowup_window(
        self,
        plot_slice: slice,
        filename_tag: str,
        show_plot: bool = False,
    ) -> None:
        """Render and save one diagnostic plot for a sliced history window.

        Reuses all the panel logic from ``BurgersPure.post_plotting`` but
        operates on ``plot_slice`` of the per-step lists so axes are scaled
        to the region of interest rather than the full run.
        """
        from itertools import chain as _chain
        from matplotlib import pyplot as plt

        # ── Slice per-step lists ────────────────────────────────────────
        # Plain Python lists don't support slice objects directly —
        # use slice.indices() to get concrete start/stop bounds.
        s_start, s_stop, _ = plot_slice.indices(len(self.time_steps))
        time_steps_sliced = self.time_steps[s_start:s_stop]
        energy_sliced = self.energy_history[s_start:s_stop]
        dissipation_sliced = self.dissipation_history[s_start:s_stop]
        residual_history_sliced = self.residual_history[s_start:s_stop]
        update_history_sliced = self.update_history[s_start:s_stop]

        if not residual_history_sliced:
            return

        # ── Smoothed convergence (mirrors BurgersPure.moving_stats) ────
        first_res = [r[0] for r in residual_history_sliced if r]
        last_res = [r[-1] for r in residual_history_sliced if r]
        first_upd = [u[0] for u in update_history_sliced if u]
        last_upd = [u[-1] for u in update_history_sliced if u]
        fr_mean, fr_std = self.moving_stats(first_res)
        lr_mean, lr_std = self.moving_stats(last_res)
        fu_mean, fu_std = self.moving_stats(first_upd)
        lu_mean, lu_std = self.moving_stats(last_upd)

        # ── Last NR iteration in this window ───────────────────────────
        last_res_iter = residual_history_sliced[-1]
        last_upd_iter = update_history_sliced[-1]

        # ── Solution snapshot at end of window ─────────────────────────
        # Use the buffer's amplitude snapshot at the corresponding step if
        # available, otherwise fall back to self.solution (end of full run).
        buf = self._blowup_buffer
        window_end_buf_idx = s_stop  # same index into buffer
        if buf.amplitude_snapshots and window_end_buf_idx <= len(buf):
            solution_snapshot = buf.amplitude_snapshots[window_end_buf_idx - 1]
        else:
            solution_snapshot = self.solution

        # ── Energy spectrum of snapshot ────────────────────────────────
        wn, sp = self.get_positive_spectrum(
            *self.compute_energy_spectrum(solution_snapshot)
        )

        t_axis = np.arange(len(first_res))  # relative x-axis within window

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(3, 2)

        # Panel 0 — solution snapshot at window end
        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(
            self.node_coords,
            solution_snapshot,
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
        step_range = (
            f"steps {time_steps_sliced[0]}–{time_steps_sliced[-1]}"
            if len(time_steps_sliced) > 1
            else f"step {time_steps_sliced[0]}"
        )
        ax0.set_title(f"Solution at window end  [{step_range}]  [SGS: LES-ANN]")

        # Panel 1 — global convergence (smoothed)
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(t_axis, fr_mean, color="royalblue", label="Residual (first)")
        ax1.fill_between(
            t_axis, fr_mean - fr_std, fr_mean + fr_std, color="royalblue", alpha=0.15
        )
        ax1.plot(t_axis, lr_mean, color="navy", linestyle="--", label="Residual (last)")
        ax1.fill_between(
            t_axis, lr_mean - lr_std, lr_mean + lr_std, color="navy", alpha=0.15
        )
        ax1.plot(t_axis, fu_mean, color="tab:orange", label="Update (first)")
        ax1.fill_between(
            t_axis, fu_mean - fu_std, fu_mean + fu_std, color="tab:orange", alpha=0.15
        )
        ax1.plot(
            t_axis, lu_mean, color="darkorange", linestyle="--", label="Update (last)"
        )
        ax1.fill_between(
            t_axis, lu_mean - lu_std, lu_mean + lu_std, color="darkorange", alpha=0.15
        )
        tol_r = self.configuration["convergence_tol_residual"]
        tol_u = self.configuration["convergence_tol_update"]
        ax1.axhline(y=tol_r, color="lightskyblue", linestyle="--")
        ax1.axhline(y=tol_u, color="lightsalmon", linestyle="--")
        ax1.set_yscale("log")
        ax1.set_xlabel(f"Time step (relative, window start = {time_steps_sliced[0]})")
        ax1.set_ylabel("Norm")
        ax1.set_title("Global convergence (smoothed, window)")
        ax1.grid(True)
        ax1.legend(fontsize=7)

        # Panel 2 — last NR iteration in window
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(last_res_iter, "o-", label="Residual", color="royalblue")
        if last_upd_iter:
            ax2.plot(last_upd_iter, "x--", label="Update", color="tab:orange")
        ax2.set_yscale("log")
        ax2.set_xlabel("Newton iteration")
        ax2.set_ylabel("Norm")
        ax2.set_title("Last Newton iteration (window end)")
        ax2.grid(True)
        ax2.legend()

        # Panel 3 — energy & dissipation in window
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(time_steps_sliced, energy_sliced, color="red", label="Total energy")
        ax3.plot(
            time_steps_sliced, dissipation_sliced, color="purple", label="Dissipation"
        )
        ax3.set_xlabel("Time step")
        ax3.set_title("Energy and dissipation (window)")
        ax3.grid(True)
        ax3.legend()

        # Panel 4 — spectral analysis at window end
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.loglog(wn[1:], sp[1:], marker="o")
        ax4.set_xlabel("Wavenumber k")
        ax4.set_ylabel("E(k)")
        ax4.set_title("Spectral analysis (window end)")
        ax4.grid(True)

        plt.tight_layout()

        save_path = self.master_path / f"{filename_tag}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Blow-up diagnostic plot saved to %s", save_path)

        if show_plot:
            plt.show()
        else:
            plt.close(fig)
