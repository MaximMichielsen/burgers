"""Coupled Burgers FEM solver: Galerkin + ANN-predicted SGS correction.

Inherits from BurgersPure and overrides only the parts that change:

    1. advance_time_step  — maintains the lagged solution history buffer
                            (u_bar at t^n, t^{n-1}, t^{n-2}) needed by the
                            LFS input stencil.
    2. nr_iteration       — builds the ANN input stencil once per NR call
                            (outside the element loop; stencil is frozen at
                            lagged time levels, so it does not change during
                            NR iterations), then adds the predicted interaction
                            terms to the global residual after assembly.

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

References
----------
Robijns (2019), Section 3.2 / 5.
Pusuluri (2021), Section 3.3.
Rajampeta (2022), Section 4.3–4.4.
Research Proposal, Section 2.3.
"""

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from constants import OUTPUT_UNITS
from data_curation.a_priori_verificiation import N_OUTPUT_TERMS
from solvers.burgers_pure import BurgersPure
from data_curation.training_data_assembly import (
    build_input_stencil,
    _gradient_basis_functions,
)
from ml_agents.predictor import SGSPredictor, load_predictor


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
    """

    def __init__(
        self,
        configuration: dict,
        clip_pusuluri: bool = False,
        clip_rajampeta: bool = False,
        exclude_visc: bool = True,
        sigma_multiplier: float = 1.0
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

    # ------------------------------------------------------------------
    # Public factory helper — mirrors BurgersPure.create_config
    # ------------------------------------------------------------------

    @staticmethod
    def create_coupled_config(
        ann_model_path: str | Path,
        normalisation_stats_path: str | Path,
        ann_warmup_steps: int = 2,
        **base_config_kwargs,
    ) -> dict:
        """Build a coupled-solver configuration dict.

        Calls BurgersPure.create_config with *base_config_kwargs* and
        appends the ANN-specific keys.
        """
        base_config = BurgersPure.create_config(**base_config_kwargs)
        base_config["simulation_mode"] = "ann"
        base_config["ann_model_path"] = str(ann_model_path)
        base_config["normalisation_stats_path"] = str(normalisation_stats_path)
        base_config["ann_warmup_steps"] = ann_warmup_steps
        return base_config

    # ------------------------------------------------------------------
    # Override: advance_time_step — maintain lagged history buffer
    # ------------------------------------------------------------------

    def advance_time_step(self) -> None:
        """Advance one time step and update the lagged solution history."""
        # Evaluate / store forcing before calling super (which also does this,
        # but we need forcing_current populated for the history buffer).
        super().advance_time_step()

        # After super() has advanced u^{n+1}, store the *pre-advance* state.
        # We record the solution that was current at the start of this step
        # (i.e. u^n), which is what the stencil references as the most recent
        # lagged level.  super().solution is now u^{n+1}, so we use the copy
        # that NR started from — stored as solution_n inside nr_iteration.
        # The cleanest approach is to snapshot before the super call; we do
        # that by keeping a rolling buffer updated here.

        # Store u^n (current solution before this step's advance).
        # Note: super().advance_time_step() has already updated self.solution
        # to u^{n+1}.  We push the *previous* solution, which we saved at the
        # top of the previous call.  On the very first call, push the IC.
        if len(self._u_bar_history) == 0:
            # First step: history is empty — push IC twice to pre-fill t^{n-1}
            # and t^{n-2} slots with the initial condition.
            self._u_bar_history.append(self.initial_condition.copy())
            self._u_bar_history.append(self.initial_condition.copy())
            self._du_bar_dt_history.append(np.zeros(self.n_nodes))
            self._du_bar_dt_history.append(np.zeros(self.n_nodes))
            self._forcing_history.append(
                self.forcing_current.copy()
                if self.forcing_current is not None
                else np.zeros(self.n_nodes)
            )
            self._forcing_history.append(
                self.forcing_current.copy()
                if self.forcing_current is not None
                else np.zeros(self.n_nodes)
            )

        # Push current (just-completed) solution as the newest lagged level
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

        # Keep only 3 levels
        if len(self._u_bar_history) > 3:
            self._u_bar_history.pop(0)
            self._du_bar_dt_history.pop(0)
            self._forcing_history.pop(0)

        self._step_count += 1

    # ------------------------------------------------------------------
    # Override: nr_iteration — inject ANN correction into residual
    # ------------------------------------------------------------------

    def nr_iteration(self, solution: NDArray) -> NDArray:
        """Newton–Raphson with ANN correction added to the global residual.

        The ANN term is frozen at the lagged state (LFS) and does not change
        between NR iterations, so it is computed once and cached.
        """
        # Build the per-element ANN correction vector (frozen for all NR iters)
        ann_correction_per_element: NDArray | None = self._compute_ann_correction()

        # Run the standard Galerkin NR but intercept the residual after assembly
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

            # --- Inject ANN correction into global residual ---
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

        # Normalise
        x_batch = np.array(input_rows, dtype=np.float32)
        x_batch_norm = (x_batch - self._x_mean) / self._x_std

        # Forward pass
        with torch.no_grad():
            x_tensor = torch.from_numpy(x_batch_norm)
            y_norm_tensor = self._predictor(x_tensor)
            y_norm = y_norm_tensor.numpy()

        # De-normalise back to physical space
        y_phys = y_norm * self._y_std + self._y_mean  # (n_valid, 5)

        # Pusuluri μ ± 3σ clipping (offline training bounds)
        if self.clip_pus:
            y_phys = np.clip(y_phys, self._y_lower_bound, self._y_upper_bound)

        # Scatter into full array, with optional Rajampeta backscatter limiting
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

            # Spatial terms (cross, Reynolds, viscous): stored with w_x = +1/h
            # Left node sees w_x = -1/h → contribution flips sign
            # Right node sees w_x = +1/h → same sign as stored
            # Outer sign: -IT (interaction terms subtracted from residual)
            residual_modified[node_left] += cross_val + reynolds_val + viscous_val
            residual_modified[node_right] -= cross_val + reynolds_val + viscous_val
            residual_modified[node_left] -= temporal_left_val
            residual_modified[node_right] -= temporal_right_val

        return residual_modified
