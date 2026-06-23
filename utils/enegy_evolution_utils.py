"""Utility functions for energy evolution diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from utils.io_utils import read_data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_energy_series(
    solutions: list[NDArray], coords: list[NDArray]
) -> list[float]:
    """Compute ½∫u² dx per snapshot via trapezoidal integration."""
    return [
        float(np.trapezoid(0.5 * solution**2, coord))
        for solution, coord in zip(solutions, coords)
    ]


def _compute_dissipation_series(
    solutions: list[NDArray],
    coords: list[NDArray],
    viscosity: float,
) -> list[float]:
    """Compute ν∫(∂u/∂x)² dx per snapshot via trapezoidal integration."""
    result: list[float] = []
    for solution, coord in zip(solutions, coords):
        du_dx = np.gradient(solution, coord)
        result.append(float(viscosity * np.trapezoid(du_dx**2, coord)))
    return result


def _compute_energy_spectrum(
    solution: NDArray, domain_length: float
) -> tuple[NDArray, NDArray]:
    """Return positive wavenumbers and spectral energy of a solution snapshot."""
    n_points = len(solution)
    u_hat = np.fft.fft(solution)
    wavenumbers = np.fft.fftfreq(n_points, d=domain_length / n_points) * 2 * np.pi
    spectrum = 0.5 * np.abs(u_hat) ** 2 / n_points
    mask = wavenumbers > 0
    return wavenumbers[mask], spectrum[mask]


def _plot_series(
    ax: plt.Axes,
    data: dict,
    x_key: str,
    y_key: str,
) -> None:
    """Plot a time series for all entries in data onto ax."""
    for label, entry in data.items():
        ax.plot(
            entry[x_key],
            entry[y_key],
            color=entry["color"],
            linestyle=entry["ls"],
            linewidth=entry["lw"],
            label=label,
        )


def _load_solver_data(directory: Path) -> dict:
    """Load CSV snapshots and compute energy diagnostics for one solver."""
    mesh, times, solutions, _ = read_data(directory)
    coords = [mesh] * len(solutions)
    return {
        "times": times,
        "solutions": solutions,
        "coords": coords,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_energy_comparison(
    dns_dir: Path,
    les_a_dir: Path,
    les_nm_dir: Path | None,
    les_sgsp_dir: Path | None,
    les_avcg_dir: Path | None,
    output_path: Path,
    viscosity: float = 0.01,
    domain_length: float = 1.0,
    projection_dir: Path | None = None,
    les_avcl_dir: Path | None = None,
) -> None:
    """Read CSV solver outputs and produce a 3-panel energy comparison figure."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    solver_configs: dict[str, tuple[Path | None, str, str, float]] = {
        "DNS": (dns_dir, "gray", "-", 1.8),
        "LES-A": (les_a_dir, "tab:orange", "--", 1.4),
        "LES-NM": (les_nm_dir, "gold", "-.", 1.4),
        "LES-SGSP": (les_sgsp_dir, "crimson", "-", 1.8),
        "LES-AVCG": (les_avcg_dir, "royalblue", "-", 1.8),
        "LES-AVCL": (les_avcl_dir, "blueviolet", "--", 1.8),
        "Projection": (projection_dir, "lightgreen", "-", 1.2),
    }

    data: dict = {}
    for label, (directory, color, ls, lw) in solver_configs.items():
        if directory is None:
            continue
        directory = Path(directory)
        if not directory.exists():
            print(f"  Skipping {label}: directory not found at {directory}")
            continue
        try:
            entry = _load_solver_data(directory)
            entry["color"] = color
            entry["ls"] = ls
            entry["lw"] = lw
            data[label] = entry
            print(f"  Loaded {label}: {len(entry['times'])} snapshots")
        except FileNotFoundError as err:
            print(f"  Skipping {label}: {err}")

    if len(data) < 2:
        print("Not enough data to produce comparison plot.")
        return

    for label, entry in data.items():
        entry["energy"] = _compute_energy_series(entry["solutions"], entry["coords"])
        entry["dissipation"] = _compute_dissipation_series(
            entry["solutions"], entry["coords"], viscosity
        )
        wavenumbers, spectrum = _compute_energy_spectrum(
            entry["solutions"][-1], domain_length
        )
        entry["wavenumbers"] = wavenumbers
        entry["spectrum"] = spectrum

    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.32)
    ax_energy = fig.add_subplot(gs[0, 0])
    ax_spectrum = fig.add_subplot(gs[0, 1])
    ax_dissipation = fig.add_subplot(gs[0, 2])

    _plot_series(ax_energy, data, "times", "energy")
    ax_energy.set_xlabel("Time $t$")
    ax_energy.set_ylabel(r"$\frac{1}{2}\int u^2\,dx$")
    ax_energy.set_title("Total kinetic energy")
    ax_energy.legend()
    ax_energy.grid(True, alpha=0.25)

    for label, entry in data.items():
        ax_spectrum.loglog(
            entry["wavenumbers"],
            entry["spectrum"],
            color=entry["color"],
            linestyle=entry["ls"],
            linewidth=entry["lw"],
            label=label,
        )
    first_entry = next(iter(data.values()))
    wn_ref = first_entry["wavenumbers"]
    mid_idx = len(wn_ref) // 3
    slope_line = first_entry["spectrum"][mid_idx] * (wn_ref / wn_ref[mid_idx]) ** (-5 / 3)
    ax_spectrum.loglog(
        wn_ref, slope_line, color="lightgray", linestyle="--", linewidth=1.0, label=r"$k^{-5/3}$"
    )
    ax_spectrum.set_xlabel("Wavenumber $k$")
    ax_spectrum.set_ylabel("$E(k)$")
    ax_spectrum.set_title("Energy spectrum at $t_{\\mathrm{final}}$")
    ax_spectrum.legend()
    ax_spectrum.grid(True, which="both", alpha=0.2)

    _plot_series(ax_dissipation, data, "times", "dissipation")
    ax_dissipation.set_xlabel("Time $t$")
    ax_dissipation.set_ylabel(r"$\nu\int\left(\partial u/\partial x\right)^2\,dx$")
    ax_dissipation.set_title("Viscous dissipation")
    ax_dissipation.legend()
    ax_dissipation.grid(True, alpha=0.25)

    fig.suptitle("Energy diagnostics: DNS vs LES variants", fontsize=13, fontweight="bold", y=1.02)

    save_path = output_path / "energy_comparison.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved energy comparison plot to '{save_path}'.")
    plt.close(fig)
