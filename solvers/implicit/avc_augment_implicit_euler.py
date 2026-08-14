"""Coupled Burgers FEM solver: SGS predictor + AV corrector.

Inherits from BurgersSGSP (which handles the SGS predictor) and adds:
    _create_avc_input_stencil  — builds the MDP state sₙ ∈ ℝ^(K+2)
    _calc_avc_correction       — runs the AVCorrector forward pass
    _add_avc_correction        — writes αₙ into self.av_correction
    advance_time_step          — calls super() then appends AV/drain histories
    update_av_history          — records αₙ and energy drain per step
    calc_energy_drain          — computes ΔE_drain = α · ∫(∂u/∂x)² dx

The corrected coarse-scale problem (Research Proposal eq. 2.7) is:
    M·U̇ + A(U)·U + (ν + α(t))·K₀·U + C_fs(U) = f

α(t) enters the physics through the already-overridden _residual_integrand
and _jacobian_integrand methods, which read self.av_correction at
element-assembly time.  No further changes to BurgersBase's NR loop are
needed.

correction_is_fixed mode bypasses the policy and holds av_correction at
its initial value; useful for constant-α sweep data collection.

References
----------
Research Proposal §2.2.4, §2.3.2.
Robijns (2019), Pusuluri (2021), Rajampeta (2022).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from ml.ml_agents.before_rk2.corrector_implicit import AVController, load_corrector
from problems_and_configurations.disc_config import DiscretizationConfig
from problems_and_configurations.problems import Problem
from ml.ml_agents.before_rk2.solver_configs import SGSPConfig, AVCConfig
from solvers.implicit.sgsp_augment_implicit_euler import SGSPSolverImplicit

logger = logging.getLogger(__name__)


class AVCSolverImplicit(SGSPSolverImplicit):
    """Burgers FEM solver with SGS predictor and AV corrector.

    Extends BurgersSGSP by injecting a learned scalar artificial viscosity
    α(t) into the diffusion term at each control step.
    """

    def __init__(
        self,
        problem: Problem,
        disc_cfg: DiscretizationConfig,
        simulation_mode: str,
        master_path: Path,
        sgsp_cfg: SGSPConfig,
        avc_cfg: AVCConfig,
        snapshot_factor: int = 1,
    ) -> None:
        super().__init__(
            problem,
            disc_cfg,
            simulation_mode,
            master_path,
            sgsp_cfg,
            snapshot_factor,
        )

        self._avc_cfg = avc_cfg

        self._avc_model_path: Path = avc_cfg.avc_model_path
        self.corrector: AVController = load_corrector(avc_cfg.avc_model_path)
        self.corrector.eval()

        self.av_correction: float | NDArray = 0.0
        self.av_history: list[float | NDArray] = []
        self.energy_drain_history: list[float] = []
        self.sgsp_injection_history: list[float] = []

        self._n_wavenumber_bins: int = (self.n_nodes + 1) // 2
        self._current_element: tuple[int, int] = (0, 1)

        self._step_counter: int = 0

    def advance_time_step(self) -> bool:
        """Query policy every n_skip_steps, then advance one LES step.

        For AVCTrainerConfig, av_correction is driven externally by
        BurgersAVCEnvironment.step() and must not be overwritten here.
        """
        if not self._avc_cfg.externally_driven:
            if self._step_counter % self._avc_cfg.n_skip_steps == 0:
                self.av_correction = self._calc_avc_correction()

        self._step_counter += 1
        step_ok = super().advance_time_step()
        self.update_av_history()
        return step_ok

    def update_av_history(self) -> None:
        """Append current av_correction and energy drain to per-step histories."""
        self.av_history.append(self.av_correction)
        self.energy_drain_history.append(self.calc_energy_drain())
        self.sgsp_injection_history.append(self.calc_sgsp_energy_injection())

    # TODO: Handle local mode.
    def create_avc_input_stencil(self) -> NDArray:
        """Build the MDP state s_n in R^(K+2) per eq. (2.8), revised.

        s_n = (Ehat_1, ..., Ehat_K, eps^-n, alpha_{n-1})
        where Ehat_k = E_LES(k,t) / sum_k(E_LES(k,t)) is the normalized
        LES spectral energy fraction (shape of the spectrum, not absolute
        magnitude), bounded in [0, 1] by construction.
        """
        if not np.all(np.isfinite(self.solution)):
            return np.zeros(self._n_wavenumber_bins + 2, dtype=np.float64)

        wavenumbers_all, raw_spectrum_all = self.compute_energy_spectrum(self.solution)
        _, positive_spectrum = self.get_positive_spectrum(
            wavenumbers_all, raw_spectrum_all
        )
        spectrum_k = positive_spectrum.astype(np.float32)
        total_les_energy = float(spectrum_k.sum())
        normalised_spectrum = spectrum_k / max(total_les_energy, 1e-12)
        dissipation_val = np.float64(
            self.dissipation_history[-1] if self.dissipation_history else 0.0
        )
        if isinstance(self.av_correction, np.ndarray):
            alpha_prev_val = np.float64(float(np.mean(self.av_correction)))
        else:
            alpha_prev_val = np.float64(self.av_correction)

        return np.concatenate(
            [
                normalised_spectrum,
                np.array([dissipation_val, alpha_prev_val], dtype=np.float64),
            ]
        )

    def _calc_avc_correction(self) -> float | NDArray:
        """Run the AVCorrector forward pass and return αₙ.

        Returns
        -------
        alpha_n : scalar float for global mode, NDArray shape (N_nodes,) for local.
        """
        state_array = self.create_avc_input_stencil()
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            alpha_tensor = self.corrector(state_tensor).squeeze(0)  # (output_dim,)

        if self.corrector.correction_mode == "global":
            return float(alpha_tensor.item())
        else:
            return alpha_tensor.numpy().astype(np.float64)

    # ------------------------------------------------------------------ #
    #  AVC-adjusted elemental integrands (eq. 2.7)
    # ------------------------------------------------------------------ #

    def _residual_integrand(
        self,
        i: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
        mid: dict[str, float],
        f_interp: float = 0.0,
    ) -> float:
        """Weak-form residual integrand with effective viscosity νeff = ν + α."""
        if self.corrector.correction_mode == "global" or isinstance(
            self.av_correction, float
        ):
            av_local = float(self.av_correction)
        else:
            av_local = float(basis @ self.av_correction[list(self._current_element)])
        time_derivative = basis[i] * (f["u_k"] - f["u_n"]) / self.dt
        diffusion = (self.viscosity + av_local) * mid["du_mid"] * gradient_basis[i]
        advection = basis[i] * mid["u_mid"] * mid["du_mid"]
        forcing = basis[i] * f_interp
        return time_derivative + diffusion + advection - forcing

    def _jacobian_integrand(
        self,
        i: int,
        j: int,
        basis: NDArray,
        gradient_basis: NDArray,
        f: dict[str, float],
    ) -> float:
        """Jacobian integrand with effective viscosity νeff = ν + α."""
        if self.corrector.correction_mode == "global" or isinstance(
            self.av_correction, float
        ):
            av_local = float(self.av_correction)
        else:
            av_local = float(basis @ self.av_correction[list(self._current_element)])
        mass = basis[i] * basis[j] / self.dt
        stiffness = (self.viscosity + av_local) * gradient_basis[i] * gradient_basis[j]
        advection = basis[i] * (basis[j] * f["du_k"] + f["u_k"] * gradient_basis[j])
        return mass + 0.5 * (stiffness + advection)

    # ------------------------------------------------------------------ #
    #  Element context (needed for local AV interpolation)
    # ------------------------------------------------------------------ #

    def calculate_elemental_residual_jacobian(
        self,
        element: tuple[int, int],
        u_k: NDArray,
        u_n: NDArray,
        f_e: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Store current element for local AV interpolation, then delegate to super."""
        self._current_element = element
        return super().calculate_elemental_residual_jacobian(element, u_k, u_n, f_e)

    # ------------------------------------------------------------------ #
    #  Energy drain
    # ------------------------------------------------------------------ #

    def calc_energy_drain(self) -> float:
        """Compute energy drained by the AV correction at the current step.

        The AV contribution to dissipation is α · ∫(∂u/∂x)² dx, mirroring
        compute_dissipation in BurgersBase but with α instead of ν.
        For local mode α is interpolated to each Gauss point via the basis functions.

        Returns
        -------
        energy_drain : float  (zero when av_correction is everywhere zero).
        """
        av_corr = self.av_correction
        if isinstance(av_corr, np.ndarray):
            if not np.any(av_corr):
                return 0.0
        elif av_corr == 0.0:
            return 0.0

        drain = 0.0
        jacobian_val = self.element_size / 2.0
        dn_dx = self.reference_gradient_basis_functions()
        points, weights = self.gauss_legendre(2)

        for element in self.elements:
            u_element = self.solution[element]
            for gauss_point, gauss_weight in zip(points, weights):
                if isinstance(av_corr, np.ndarray):
                    basis_vals = self.reference_basis_functions(gauss_point)
                    av_local = float(basis_vals @ av_corr[list(element)])
                else:
                    av_local = float(av_corr)

                drain += (
                    av_local
                    * gauss_weight
                    * abs(jacobian_val)
                    * (dn_dx @ u_element) ** 2
                )
        return drain

    def print_configuration(self) -> None:
        """Print base + SGSP config plus AVC-specific settings."""
        super().print_configuration()
        W = 72
        COL = 30

        def _row(label: str, value: str) -> None:
            print(f"  {label:<{COL}} {value}")

        print()
        print("  AV Corrector")
        print("─" * W)
        _row("model path", str(self._avc_model_path))
        _row("correction mode", str(self.corrector.correction_mode))
        _row("output_scale", f"{self.corrector.output_scale:.4e}")
        _row("n_wavenumber_bins", str(self._n_wavenumber_bins))
        print("═" * W)

    # ------------------------------------------------------------------ #
    #  Post-processing
    # ------------------------------------------------------------------ #

    def post_processing(self) -> None:
        """Standard post-processing plus AVC contribution plots."""
        super().post_processing()
        if self.av_history:
            self.plot_avc_contributions()
            self.plot_local_avc_spatial()

    def post_logging(self) -> None:
        """Write run summary plus AVC-specific fields to the log file."""
        super().post_logging()
        if self.av_history:
            # For local mode each entry is an NDArray; flatten to a single array
            # of all per-node values across all steps for summary statistics.
            av_arr = np.array(
                [
                    np.mean(a) if isinstance(a, np.ndarray) else a
                    for a in self.av_history
                ],
                dtype=np.float64,
            )
            self.logger.info(
                "AV correction — mean: %.4e  min: %.4e  max: %.4e  final: %.4e",
                float(av_arr.mean()),
                float(av_arr.min()),
                float(av_arr.max()),
                float(av_arr[-1]),
            )

    def plot_avc_contributions(self) -> None:
        """Plot applied viscosity α(t) and accumulated energy drain over time."""
        if not self.av_history:
            logger.warning("plot_avc_contributions: no AV history to plot, skipping.")
            return

        n_plot_points = (
            min(len(self.time_steps), len(self.av_history)) - self._sgsp_warmup_steps
        )
        if n_plot_points == 0:
            logger.warning("plot_avc_contributions: no data points to plot, skipping.")
            return

        time_axis = self.time_steps[
            self._sgsp_warmup_steps + self._avc_cfg.n_skip_steps : n_plot_points
        ]
        av_raw = self.av_history[:n_plot_points]
        av_array = np.array(
            [np.mean(a) if isinstance(a, np.ndarray) else a for a in av_raw],
            dtype=np.float64,
        )
        av_array = av_array[self._sgsp_warmup_steps + self._avc_cfg.n_skip_steps :]
        drain_array = np.cumsum(self.energy_drain_history[:n_plot_points])
        drain_array = drain_array[
            self._sgsp_warmup_steps + self._avc_cfg.n_skip_steps :
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(time_axis, av_array, color="tab:orange", linewidth=1.5)
        axes[0].set_xlabel("Time")
        y_label = (
            r"$\alpha(t)$"
            if self._avc_cfg.output_scope == "global"
            else r"$\alpha_{\text{mean}}(t)$"
        )
        axes[0].set_ylabel(y_label)
        title = (
            "Global AV correction applied by corrector policy"
            if self._avc_cfg.output_scope == "global"
            else "Mean AV correction applied by local corrector policy"
        )
        axes[0].set_title(title)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time_axis, drain_array, color="royalblue", linewidth=1.5)
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Cumulative energy drain")
        axes[1].set_title("Cumulative energy drained by AV correction")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.master_path / f"avc_contributions_{self.run_id}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("AVC contribution plot saved to %s", save_path)
        plt.close(fig)

    def plot_local_avc_spatial(self, n_snapshots: int = 3) -> None:
        """Plot α(x) spatial profile at n_snapshots time indices for local mode.

        Skipped silently if correction_mode is global or av_history is empty.
        """
        if self._avc_cfg.output_scope == "global":
            return
        if not self.av_history:
            logger.warning("plot_local_avc_spatial: no AV history to plot, skipping.")
            return

        local_av_history = [a for a in self.av_history if isinstance(a, np.ndarray)]
        if not local_av_history:
            logger.warning(
                "plot_local_avc_spatial: no local NDArray entries in av_history, skipping."
            )
            return

        n_available = len(local_av_history)
        n_snapshots = min(n_snapshots, n_available)
        snapshot_indices = np.linspace(0, n_available - 1, n_snapshots, dtype=int)

        colors = ["royalblue", "tab:orange", "lightgreen"]
        fig, ax = plt.subplots(figsize=(8, 4))

        for plot_idx, history_idx in enumerate(snapshot_indices):
            alpha_spatial = local_av_history[history_idx]
            time_val = (
                self.time_steps[history_idx + 1]
                if history_idx + 1 < len(self.time_steps)
                else history_idx * self.dt
            )
            ax.plot(
                self.mesh,
                alpha_spatial,
                color=colors[plot_idx % len(colors)],
                linewidth=1.5,
                linestyle=":",
                marker="o",
                markersize=3,
                label=f"t = {time_val:.3f}",
            )

        ax.set_xlabel("x")
        ax.set_ylabel(r"$\alpha(x, t)$")
        ax.set_title("Local AV correction spatial profile")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.master_path / f"avc_local_spatial_{self.run_id}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Local AVC spatial plot saved to %s", save_path)
        plt.close(fig)
