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
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from ml.tau_ann import load_tau_ann, TauANN, TauANNConfig, Scope
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
        ann_config: TauANNConfig,
        simulation_mode: SimulationMode = SimulationMode.TAU_BASED,
        ann_path: Path | None = None,
        snapshot_factor: int = 1,
        t_start: float = 0.0,
        training_mode: bool = False,
        prescribed_action_trajectory: list | None = None,
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
        self.ann_config: TauANNConfig = ann_config

        self.n_local_stencil_points = ann_config.n_local_stencil_points

        self.n_correction_coefficients = tau_model.output_dimensions
        self.correction_coefficients: NDArray | None = None
        self.correction_coefficients_history: list = []

        self.prescribed_action_trajectory = prescribed_action_trajectory

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

        if self.prescribed_action_trajectory is not None:
            action = self.prescribed_action_trajectory[self.current_time_step]
            self.correction_coefficients = np.asarray(action, dtype=np.float64)

        elif not self.training_mode:
            self.correction_coefficients = self.get_ann_coefficients()

        new_solution = self.nr_iteration(self.solution, self.solution_previous)
        self.solution_previous = self.solution
        self.solution = new_solution

        self.energy_history.append(self.compute_energy_(self.solution))
        self.dissipation_history.append(self.compute_dissipation_(self.solution))
        self.correction_coefficients_history.append(self.correction_coefficients)
        self.simulation_time_elapsed += self.dt

    def retrieve_local_corrections(self, element: int | None = None) -> NDArray:
        """Retrieve local or global coefficients for element evaluation."""
        if self.ann_config.output_scope == Scope.LOCAL and element is not None:
            reshaped = self.correction_coefficients.reshape(
                self.ann_config.n_local_groups, self.ann_config.n_coefficients
            )
            return reshaped[element]

        assert self.correction_coefficients is not None
        return self.correction_coefficients

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

        wavenumbers_all, raw_spectrum_all = self.compute_energy_spectrum_(self.solution)
        _, positive_spectrum = self.get_positive_spectrum(
            wavenumbers_all, raw_spectrum_all
        )
        spectrum_k = positive_spectrum.astype(np.float32)
        total_les_energy = float(spectrum_k.sum())
        normalised_spectrum = spectrum_k / max(total_les_energy, 1e-12)

        previous_coefficients = (
            self.correction_coefficients
            if self.correction_coefficients is not None
            else np.ones(self.ann_config.action_dimension)
        )

        if self.ann_config.input_scope == Scope.LOCAL:
            local_stencils = []
            for node in self.nodes:
                _, local_spectrum = self.compute_local_energy_spectrum(node)
                local_spectrum_32 = local_spectrum.astype(np.float32)
                local_total_energy = float(local_spectrum_32.sum())
                norm_local_spectrum = local_spectrum_32 / max(local_total_energy, 1e-12)

                local_stencils.append(norm_local_spectrum)

            flattened_local_features = np.concatenate(local_stencils)
            return np.concatenate(
                [normalised_spectrum, previous_coefficients, flattened_local_features]
            )

        return np.concatenate([normalised_spectrum, previous_coefficients])

    def compute_local_energy_spectrum(
        self, node: int, positive_only: bool = True
    ) -> tuple[NDArray, NDArray]:
        """Compute local energy spectrum across a 4-node stencil around a target node."""
        n_points = self.n_local_stencil_points
        half_stencil = (n_points - 1) // 2
        start_idx = node - half_stencil
        target_indices = np.arange(start_idx, start_idx + n_points, dtype=int)
        total_nodes = len(self.solution)
        valid_mask = (target_indices >= 0) & (target_indices < total_nodes)

        if not np.any(valid_mask):
            return np.array([]), np.array([])

        # Zero-pad out-of-bounds nodes outside domain boundaries
        u_local = np.zeros(n_points, dtype=np.float64)
        u_local[valid_mask] = self.solution[target_indices[valid_mask]]

        # Compute 1D FFT
        u_hat_local = np.fft.fft(u_local)

        # Wavenumbers using exact grid spacing d = element_size (dx)
        wavenumbers = np.fft.fftfreq(n_points, d=self.element_size) * 2.0 * np.pi

        # Spectrum matching global energy normalization (0.5 * |u_hat|^2 / N)
        spectrum = 0.5 * (np.abs(u_hat_local) ** 2) / n_points

        if positive_only:
            mask = wavenumbers > 0
            return wavenumbers[mask], spectrum[mask]

        return wavenumbers, spectrum

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

    # ------------------------------------------------------------------ #
    #  Tau models
    # ------------------------------------------------------------------ #

    def compute_tau(
        self, u_e: NDArray, u_x_e: NDArray | None = None, element: int | None = None
    ) -> float:
        # Default to ones (1.0 for each term) if correction_coefficients is None
        c = (
            self.retrieve_local_corrections(element=element)
            if self.correction_coefficients is not None
            else np.ones(self.tau_model.output_dimensions)
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

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #

    def plot_correction_coefficients(self, show_plot: bool = False):
        if self.ann_config.output_scope == Scope.GLOBAL:
            coefficients = [self.correction_coefficients]
        else:
            coefficients = self.correction_coefficients.reshape(
                self.ann_config.n_local_groups, self.ann_config.n_coefficients
            )

        x = [1, 2, 3, 4][: self.n_correction_coefficients]
        param_names = ["Advection", "Viscosity", "Diffusion", "Time"][
            : self.n_correction_coefficients
        ]

        fig, ax = plt.subplots(figsize=(7, 5))
        for coefficients_ in coefficients:
            ax.plot(x, coefficients_, "x", markersize=8, markeredgewidth=1.5)

        ax.axhline(
            self.ann_config.max_action / 2, color="gray", linestyle="--", linewidth=0.8
        )

        ax.set_ylim(-0.1, self.ann_config.max_action)

        for x_pos, name in zip(x, param_names):
            ax.text(
                x_pos,
                1.03,
                name,
                ha="center",
                va="bottom",
                fontweight="bold",
                clip_on=False,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlim(0.5, x[-1] + 0.5)

        ax.grid(
            True,
            which="both",
            axis="both",
            linestyle=":",
            linewidth=0.7,
            alpha=0.6,
            color="gray",
        )

        plt.suptitle("Correction Coefficients", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(
            self.master_path
            / f"post_plotting_{self.ann_config.output_scope.value}_corrections.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"Corrections plot saved to: {self.master_path / f'post_plotting_{self.ann_config.output_scope.value}_corrections.png'}"
        )

        if show_plot:
            plt.show()
        else:
            plt.close(fig)
