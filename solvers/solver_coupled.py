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
        if self.ann_config.output_scope != Scope.GLOBAL and element is not None:
            reshaped = self.correction_coefficients.reshape(
                self.ann_config.n_local_groups, self.ann_config.n_coefficients
            )
            group_id = self.ann_config.group_map[element]
            return reshaped[group_id]

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

    def post_processing(self) -> None:
        """Run post-plotting and post-logging."""
        self.post_plotting()
        self.post_logging()

        if not self.correction_coefficients_history:
            self.logger.warning("No correction coefficients recorded to plot.")
            return

        raw_history = np.array(self.correction_coefficients_history)

        n_steps = len(raw_history)
        n_groups = (
            1
            if self.ann_config.output_scope == Scope.GLOBAL
            else self.ann_config.n_local_groups
        )
        n_coeffs = self.n_correction_coefficients

        history_coeffs = raw_history.reshape(n_steps, n_groups, n_coeffs)
        history_time = self.time_steps[:n_steps]

        self.plot_correction_coefficients_snapshot(
            show_plot=False, save_suffix="final_step"
        )

        self.plot_correction_coefficients_evolution(
            history_time=history_time,
            history_coefficients=history_coeffs,
            style="both",
            show_plot=False,
        )

    def plot_correction_coefficients_snapshot(
        self, show_plot: bool = False, save_suffix: str = "snapshot"
    ) -> None:
        if self.ann_config.output_scope == Scope.GLOBAL:
            coefficients = np.atleast_2d(self.correction_coefficients)
        else:
            coefficients = self.correction_coefficients.reshape(
                self.ann_config.n_local_groups, self.ann_config.n_coefficients
            )

        n_groups, n_coeffs = coefficients.shape
        param_names = ["Advection", "Viscosity", "Diffusion", "Time"][:n_coeffs]
        x_base = np.arange(1, n_coeffs + 1)

        fig, ax = plt.subplots(figsize=(8, 5.5))

        for g in range(n_groups):
            offset = (g - (n_groups - 1) / 2) * 0.12 if n_groups > 1 else 0.0
            ax.plot(
                x_base + offset,
                coefficients[g],
                "x",
                markersize=9,
                markeredgewidth=2.0,
                alpha=0.85,  # Prevents hidden overlap
                label=f"Group {g}",
            )

        ax.axhline(
            self.ann_config.max_action, color="gray", linestyle="--", linewidth=0.8, label="max action"
        )
        ax.axhline(
            0.0, color="gray", linestyle="--", linewidth=0.8, label="min action"
        )

        # FIX 1: Set y-upper limit higher (e.g. 1.25) so markers at 1.0 are not clipped
        max_act = self.ann_config.max_action
        ax.set_ylim(-0.1, max_act * 1.25)

        # FIX 2: Raise label text positions above the highest possible marker
        for x_pos, name in zip(x_base, param_names):
            ax.text(
                x_pos,
                max_act * 1.12,
                name,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
                clip_on=False,
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlim(0.5, n_coeffs + 0.5)

        if n_groups > 1:
            ax.legend(title="Spatial Groups", loc="best", frameon=True)

        ax.grid(
            True, which="both", linestyle=":", linewidth=0.7, alpha=0.6, color="gray"
        )

        plt.suptitle("Correction Coefficients", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()

        save_path = (
            self.master_path
            / f"post_plotting_{self.ann_config.output_scope.value}_corrections_{save_suffix}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def plot_correction_coefficients_evolution(
        self,
        history_time: NDArray,
        history_coefficients: NDArray,
        style: str = "lines",  # Choose "lines", "heatmap", or "both"
        show_plot: bool = False,
    ) -> None:
        """Plots the temporal evolution of local correction coefficients over time."""
        if style not in ("lines", "heatmap", "both"):
            raise ValueError(
                f"Invalid style: {style!r}. Choose 'lines', 'heatmap', or 'both'."
            )

        n_steps, n_groups, n_coeffs = history_coefficients.shape
        param_names = ["Advection", "Viscosity", "Diffusion", "Time"][:n_coeffs]

        styles_to_render = ["lines", "heatmap"] if style == "both" else [style]

        for current_style in styles_to_render:
            if current_style == "lines":
                n_rows = 1 if n_coeffs <= 2 else 2
                n_cols = min(n_coeffs, 2)
                fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(5 * n_cols, 3.5 * n_rows),
                    sharex=True,
                    sharey=True,
                )
                axes_list = np.atleast_1d(axes).flatten()

                for c_idx, name in enumerate(param_names):
                    ax = axes_list[c_idx]
                    for g in range(n_groups):
                        ax.plot(
                            history_time,
                            history_coefficients[:, g, c_idx],
                            label=f"Group {g}",
                            linewidth=1.8,
                        )
                    ax.set_title(name, fontsize=11, fontweight="bold")
                    ax.set_ylabel("Coefficient Value")
                    ax.grid(True, linestyle=":", alpha=0.6)
                    if c_idx == 0 and n_groups > 1:
                        ax.legend(loc="best", frameon=True)

                for ax in axes_list[-n_cols:]:
                    ax.set_xlabel("Time (t)")

                fig.suptitle(
                    "Temporal Evolution of Local Corrections",
                    fontsize=13,
                    fontweight="bold",
                )
                # rect prevents suptitle from overlapping subplot titles
                fig.tight_layout(rect=[0, 0, 1, 0.95])

            elif current_style == "heatmap":
                # layout="constrained" handles colorbar layout without UserWarning
                n_rows = 1 if n_coeffs <= 2 else 2
                n_cols = min(n_coeffs, 2)
                fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(5 * n_cols, 3.8*n_rows),
                    sharex=True,
                    sharey=True,
                    layout="constrained",
                )
                axes_list = np.atleast_1d(axes).flatten()

                for c_idx, name in enumerate(param_names):
                    ax = axes_list[c_idx]
                    data = history_coefficients[:, :, c_idx].T

                    im = ax.imshow(
                        data,
                        aspect="auto",
                        origin="lower",
                        extent=[
                            history_time[0],
                            history_time[-1],
                            -0.5,
                            n_groups - 0.5,
                        ],
                        cmap="magma",
                        vmin=0.0,
                        vmax=self.ann_config.max_action,
                    )
                    ax.set_title(name, fontsize=11, fontweight="bold")
                    ax.set_yticks(range(n_groups))
                    ax.set_yticklabels([f"Group {g}" for g in range(n_groups)])

                    for g_line in range(n_groups - 1):
                        ax.axhline(
                            g_line + 0.5, color="white", linewidth=1.5, alpha=0.8
                        )

                for ax in axes_list[-n_cols:]:
                    ax.set_xlabel("Time (t)")

                fig.colorbar(
                    im,
                    ax=axes_list.tolist(),
                    orientation="vertical",
                    label="Correction Value",
                    shrink=0.85,
                )
                fig.suptitle(
                    "Spatio-Temporal Maps of Local Corrections",
                    fontsize=13,
                    fontweight="bold",
                )

            # Save each plot individually
            save_path = (
                self.master_path
                / f"post_plotting_{self.ann_config.output_scope.value}_corrections_{current_style}.png"
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Evolution plot ({current_style}) saved to: {save_path}")

            if show_plot:
                plt.show()
            else:
                plt.close(fig)
