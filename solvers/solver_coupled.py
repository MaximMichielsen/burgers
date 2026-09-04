"""Machine-learning coupled finite element solver for Burgers' equation.

Extends SolverBase to dynamically scale subgrid-scale (SGS) model coefficients
using artificial neural networks (ANNs) trained via reinforcement learning (e.g., TD3):
- Constructs feature input stencils based on normalized LES energy spectra and
  historical coefficient values.
- Evaluates neural network policies to dynamically output scale corrections
  (c_1, c_2, ...) for active sub-grid tau models.
- Supports dual execution modes: training-mode (external parameter override)
  and inference-mode (direct PyTorch model evaluation).
"""

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ml.tau_ann import load_tau_ann, TauANN
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import SolverBase, SimulationMode, TauModel


class SolverCoupled(SolverBase):
    """Base solver coupled with ANN to adjust SGS model coefficients."""

    def __init__(
        self,
        problem: Problem,
        disc_config: DiscretizationConfig,
        master_path: Path,
        tau_model: TauModel,
        simulation_mode: SimulationMode = SimulationMode.TAU_BASED,
        ann_path: Path | None = None,
        snapshot_factor: int = 1,
        t_start: float = 0.0,
        training_mode: bool = False,
    ):
        super().__init__(
            problem,
            disc_config,
            simulation_mode,
            master_path,
            tau_model,
            snapshot_factor,
            t_start,
        )

        self._COEFFICIENT_NAMES: dict[str, tuple[str, ...]] = {
            "2": ("c_1", "c_2"),
            "3": ("c_1", "c_2", "c_3"),
            "3_dt_augmented": ("c_1", "c_2", "c_3", "c_4"),
        }

        self.tau_model = tau_model
        self.training_mode = training_mode

        self.n_correction_coefficients = self.get_output_dimensions()
        self.correction_coefficients: NDArray | None = None
        self.correction_coefficients_history: list = []

        # Load ANN only during inference/solver mode
        self.ann: TauANN | None = None
        if not self.training_mode:
            if ann_path is None:
                raise ValueError(
                    "ann_path must be provided when training_mode is False."
                )
            self.ann = load_tau_ann(ann_path)

        self._n_wavenumber_bins: int = (self.n_nodes + 1) // 2

    def advance_time_step(self) -> None:
        """Advance the solution by one time step: U^{n+1} ← U^n.

        Previous solutions are stored for BDF2 time-marching."""
        self.resolve_current_forcing()
        # Only evaluate the ANN if we are in inference/eval mode (not training mode)
        if not self.training_mode:
            self.correction_coefficients = self.get_ann_coefficients()

        new_solution = self.nr_iteration(self.solution, self.solution_previous)
        self.solution_previous = self.solution
        self.solution = new_solution

        self.energy_history.append(self._compute_energy(self.solution))
        self.dissipation_history.append(self._compute_dissipation(self.solution))
        self.correction_coefficients_history.append(self.correction_coefficients)
        self.simulation_time_elapsed += self.dt

    # ------------------------------------------------------------------ #
    #  ANN
    # ------------------------------------------------------------------ #

    def create_input_stencil(self) -> NDArray:
        """Build the MDP state s_n in R^(K+n_coefficients).

        s_n = (Ehat_1, ..., Ehat_K, c_1^{n-1}, c_...^{n-1})
        where Ehat_k = E_LES(k,t) / sum_k(E_LES(k,t)) is the ...
        """
        if not np.all(np.isfinite(self.solution)):
            raise ValueError(f"Error in the solution field.\n{self.solution}")

        wavenumbers_all, raw_spectrum_all = self._compute_energy_spectrum(self.solution)
        _, positive_spectrum = self.get_positive_spectrum(
            wavenumbers_all, raw_spectrum_all
        )
        spectrum_k = positive_spectrum.astype(np.float32)
        total_les_energy = float(spectrum_k.sum())
        normalised_spectrum = spectrum_k / max(total_les_energy, 1e-12)

        previous_coefficients = (
            self.correction_coefficients
            if self.correction_coefficients is not None
            else np.ones(self.n_correction_coefficients)
        )

        return np.concatenate([normalised_spectrum, previous_coefficients])

    def get_ann_coefficients(self) -> NDArray:
        """Call ANN and receive correction coefficients."""
        state_array = self.create_input_stencil()
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            alpha_tensor = self.ann(state_tensor).squeeze(0)  # (output_dim,)

        return alpha_tensor.numpy().astype(np.float64)

    def _coefficients_as_kwargs(self) -> dict[str, float]:
        """Map the raw correction_coefficients array to tau-model kwarg names."""
        assert self.tau_model is not None, "SolverCoupled requires a tau_model."
        if self.correction_coefficients is None:
            raise RuntimeError(
                "correction_coefficients is None — compute_tau() was called "
                "before get_ann_coefficients() set it for this time step."
            )
        names = self._COEFFICIENT_NAMES[self.tau_model]
        return dict(zip(names, self.correction_coefficients))

    def get_output_dimensions(self) -> int:
        match self.tau_model:
            case TauModel.TWO_PARAMS:
                return 2
            case TauModel.THREE_PARAMS:
                return 3
            case TauModel.FOUR_PARAMS:
                return 4
            case _:
                raise ValueError(f"Unknown tau_model {self.tau_model!r}")

    # ------------------------------------------------------------------ #
    #  Tau models
    # ------------------------------------------------------------------ #

    def compute_tau(self, u_e: NDArray, u_x_e: NDArray | None = None) -> float:
        # Default to ones (1.0 for each term) if correction_coefficients is None
        c = (
            self.correction_coefficients
            if self.correction_coefficients is not None
            else np.ones(self.get_output_dimensions())
        )

        if self.tau_model == TauModel.TWO_PARAMS:
            return self.tau_model_two_params(u_e, c)
        elif self.tau_model == TauModel.THREE_PARAMS and u_x_e is not None:
            return self.tau_model_three_params(u_e, u_x_e, c)
        elif self.tau_model == TauModel.FOUR_PARAMS and u_x_e is not None:
            return self.tau_model_three_dt_aug(u_e, u_x_e, c)
        raise ValueError(
            f"Invalid tau model selection or missing gradient: {self.tau_model}"
        )
