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

from ml.ml_agents.corrector import AVCorrector, load_corrector
from solvers.burgers_sgsp import BurgersSGSP

logger = logging.getLogger(__name__)


class BurgersAVC(BurgersSGSP):
    """Burgers FEM solver with SGS predictor and AV corrector.

    Extends BurgersSGSP by injecting a learned scalar artificial viscosity
    α(t) into the diffusion term at each control step.

    Additional configuration keys
    ------------------------------
    avc_model_path      : Path to av_corrector.pt checkpoint.
    avc_alpha_max       : Upper bound αₘₐₓ = Cα·ν (must match trained model).
    dns_energy_spectrum : Array of DNS target spectral energies E_DNS(k), shape (K,).
    dns_dissipation     : Scalar DNS target dissipation rate ε_DNS.

    Parameters
    ----------
    configuration:
        Config dict from create_avc_config.
    correction_is_global:
        If True the scalar α is applied uniformly (current formulation).
        False is reserved for the future spatially-varying extension.
    correction_is_fixed:
        If True the corrector policy is bypassed; av_correction holds its
        initial value throughout.  Used for constant-α sweep data collection.
    """

    def __init__(
        self,
        configuration: dict,
        correction_is_fixed: bool = False,
        clip_pusuluri: bool = False,
        clip_rajampeta: bool = False,
        exclude_visc: bool = False,
    ) -> None:
        super().__init__(
            configuration,
            clip_pusuluri=clip_pusuluri,
            clip_rajampeta=clip_rajampeta,
            exclude_visc=exclude_visc,
        )

        avc_model_path = Path(configuration["avc_model_path"])
        self._corrector: AVCorrector = load_corrector(avc_model_path)
        self._corrector.eval()

        # Scalar AV correction α(t); written by _add_avc_correction each step.
        self.av_correction: float | NDArray = 0.0

        # Per-step history lists for diagnostics and RL data collection.
        self.av_history: list[float] = []
        self.energy_drain_history: list[float] = []

        self.correction_is_fixed: bool = correction_is_fixed

        # DNS targets for state normalization and reward computation.
        self._dns_energy_spectrum: NDArray = np.asarray(
            configuration["dns_energy_spectrum"], dtype=np.float32
        )
        self._dns_dissipation: float = float(configuration["dns_dissipation"])

        # K = number of positive wavenumber bins used in the MDP state.
        self._n_wavenumber_bins: int = len(self._dns_energy_spectrum)

        self._current_element: tuple[int, int] = (0, 1)  # overwritten each element loop

    # ------------------------------------------------------------------ #
    #  Configuration
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_avc_config(
        avc_model_path: str | Path,
        dns_energy_spectrum: NDArray,
        dns_dissipation: float,
        **sgsp_config_kwargs,
    ) -> dict:
        """Build a config dict for BurgersAVC.

        Forwards kwargs to BurgersSGSP.create_sgsp_config, then appends the
        AVC-specific keys.

        Parameters
        ----------
        avc_model_path:
            Path to the saved AVCorrector .pt checkpoint.
        dns_energy_spectrum:
            1-D array of DNS target spectral energies E_DNS(k), shape (K,).
        dns_dissipation:
            Scalar DNS target dissipation rate ε_DNS.
        **sgsp_config_kwargs:
            Forwarded to BurgersSGSP.create_sgsp_config / BurgersBase.create_config.
        """
        base_config = BurgersSGSP.create_sgsp_config(**sgsp_config_kwargs)
        base_config.update(
            {
                "simulation_mode": "avc",
                "avc_model_path": str(avc_model_path),
                "dns_energy_spectrum": dns_energy_spectrum.tolist(),
                "dns_dissipation": float(dns_dissipation),
            }
        )
        return base_config

    # ------------------------------------------------------------------ #
    #  advance_time_step
    # ------------------------------------------------------------------ #

    def advance_time_step(self) -> None:
        """Compute αₙ, inject into diffusion term, advance one LES step.

        Order of operations:
        1. Predict αₙ from current state sₙ (unless fixed mode).
        2. Write αₙ into self.av_correction (read by _residual_integrand).
        3. Call super().advance_time_step() — runs NR with SGS + AV.
        4. Append αₙ and energy drain to history lists.
        """
        if not self.correction_is_fixed:
            self._add_avc_correction()

        super().advance_time_step()
        self.update_av_history()

    # ------------------------------------------------------------------ #
    #  History update
    # ------------------------------------------------------------------ #

    def update_av_history(self) -> None:
        """Append current av_correction and energy drain to per-step histories."""
        self.av_history.append(self.av_correction)
        self.energy_drain_history.append(self.calc_energy_drain())

    # ------------------------------------------------------------------ #
    #  AVC input / inference helpers
    # ------------------------------------------------------------------ #

    def _create_avc_input_stencil(self) -> NDArray:
        """Build the MDP state sₙ ∈ ℝ^(K+2) per eq. (2.8).

        sₙ = (Ê₁, …, Êₖ, ε⁻ⁿ, αₙ₋₁)
        where Êₖ = E(k, t) / E_DNS(k) is the DNS-normalised spectral energy.

        Returns
        -------
        state_vector : float32 array of shape (K+2,).
        """
        wavenumbers_all, raw_spectrum_all = self.compute_energy_spectrum(self.solution)
        _, positive_spectrum = self.get_positive_spectrum(
            wavenumbers_all, raw_spectrum_all
        )
        # Trim to K bins to match DNS target length.
        spectrum_k = positive_spectrum[: self._n_wavenumber_bins].astype(np.float32)

        # Normalise: Êₖ = E(k) / E_DNS(k); avoid division by zero.
        dns_safe = np.where(
            self._dns_energy_spectrum > 0.0, self._dns_energy_spectrum, 1.0
        )
        normalised_spectrum = spectrum_k / dns_safe

        dissipation_val = np.float32(
            self.dissipation_history[-1] if self.dissipation_history else 0.0
        )
        alpha_prev_val = np.float32(self.av_correction)

        return np.concatenate([normalised_spectrum, [dissipation_val, alpha_prev_val]])

    def _calc_avc_correction(self) -> float:
        """Run the AVCorrector forward pass and return αₙ ∈ [0, αₘₐₓ].

        Returns
        -------
        alpha_n : scalar float.
        """
        state_array = self._create_avc_input_stencil()
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            alpha_tensor = self._corrector(state_tensor)

        return float(alpha_tensor.squeeze().item())

    def _add_avc_correction(self) -> None:
        """Predict αₙ and write it into self.av_correction.

        self.av_correction is read by the overridden _residual_integrand and
        _jacobian_integrand at element-assembly time, so it must be set before
        super().advance_time_step() is called.
        """
        alpha_output = self._calc_avc_correction()

        if self._corrector.correction_mode == "global":
            self.av_correction = float(alpha_output)  # scalar float
        else:
            self.av_correction = alpha_output  # NDArray shape (N,)

        logger.debug(
            "AVC correction applied: α=%.6e  (t=%.4f)",
            self.av_correction,
            self.simulation_time_elapsed,
        )

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
        if self._corrector.correction_mode == "global":
            av_local = self.av_correction
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
        if self._corrector.correction_mode == "global":
            av_local = self.av_correction
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

        Returns
        -------
        energy_drain : float  (zero when av_correction is zero).
        """
        if self.av_correction == 0.0:
            return 0.0

        drain = 0.0
        jacobian_val = self.element_size / 2.0
        dn_dx = self.reference_gradient_basis_functions()
        points, weights = self.gauss_legendre(2)

        for element in self.elements:
            u_element = self.solution[element]
            for gauss_point, gauss_weight in zip(points, weights):
                drain += (
                    self.av_correction
                    * gauss_weight
                    * abs(jacobian_val)
                    * (dn_dx @ u_element) ** 2
                )
        return drain

    # ------------------------------------------------------------------ #
    #  Post-processing
    # ------------------------------------------------------------------ #

    def post_processing(self) -> None:
        """Standard post-processing plus AVC contribution plots."""
        super().post_processing()
        print("AV history:", self.av_history)
        if self.av_history:
            self.plot_avc_contributions()

    def plot_avc_contributions(self) -> None:
        """Plot applied viscosity α(t) and accumulated energy drain over time."""
        if not self.av_history:
            logger.warning("No AV history to plot.")
            return

        time_axis = (
            np.array(self.time_steps[: len(self.av_history)], dtype=float) * self.dt
        )
        av_array = np.array(self.av_history)
        drain_array = np.cumsum(self.energy_drain_history)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(time_axis, av_array, color="tab:orange", linewidth=1.5)
        axes[0].axhline(
            y=self._corrector.alpha_max,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=r"$\alpha_\mathrm{max}$",
        )
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel(r"$\alpha(t)$")
        axes[0].set_title("AV correction applied by corrector policy")
        axes[0].legend()
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
