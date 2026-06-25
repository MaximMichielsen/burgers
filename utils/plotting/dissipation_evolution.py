"""Utility functions for dissipation evolution diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from utils.io_utils import read_data
from utils.pipeline_utils import RunPaths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _compute_dissipation_spectrum(
    solution: NDArray,
    domain_length: float,
    viscosity: float,
) -> tuple[NDArray, NDArray]:
    """Return positive wavenumbers and spectral dissipation D(k) = 2ν k² E(k)."""
    n_points = len(solution)
    u_hat = np.fft.fft(solution)
    wavenumbers_all = np.fft.fftfreq(n_points, d=domain_length / n_points) * 2.0 * np.pi
    energy_spectrum = 0.5 * np.abs(u_hat) ** 2 / n_points
    dissipation_spectrum = 2.0 * viscosity * wavenumbers_all**2 * energy_spectrum
    mask = wavenumbers_all > 0
    return wavenumbers_all[mask], dissipation_spectrum[mask]


def _load_solver_data(directory: Path) -> dict:
    """Load CSV snapshots for one solver."""
    mesh, times, solutions, _ = read_data(directory)
    coords = [mesh] * len(solutions)
    return {
        "times": times,
        "solutions": solutions,
        "coords": coords,
    }


def _trim_to_reference_length(
    data: dict, reference_label: str, labels_to_trim: set[str]
) -> None:
    """Trim entries in-place to match snapshot count of reference_label."""
    if reference_label not in data:
        return
    n_ref = len(data[reference_label]["times"])
    for label in labels_to_trim:
        if label in data:
            entry = data[label]
            entry["times"] = entry["times"][:n_ref]
            entry["solutions"] = entry["solutions"][:n_ref]
            entry["coords"] = entry["coords"][:n_ref]


# ---------------------------------------------------------------------------
# Public plot function
# ---------------------------------------------------------------------------


def plot_dissipation_comparison(
    paths: RunPaths,
    output_path: Path,
    viscosity: float,
    domain_length: float,
) -> None:
    """Read CSV solver outputs and produce a 3-panel dissipation comparison figure.

    Panels: full dissipation evolution, windowed dissipation evolution,
    dissipation spectrum D(k) = 2ν k² E(k) at t_final.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    _all_configs: list[tuple[str, Path | None, str, str, float]] = [
        ("DNS", paths.dns_data, "gray", "-", 1.8),
        ("Projection", paths.projection, "lightgreen", "-", 1.2),
        ("LES - A", paths.les_a_data, "tab:orange", "--", 1.4),
        ("LES - NM", paths.les_nm_data, "gold", "-.", 1.4),
        ("LES - SGSP", paths.les_sgsp_data, "crimson", "-", 1.8),
        ("LES - AVCG", paths.les_avcg_data, "royalblue", "-", 1.8),
        ("LES - AVCL", paths.avcl_model, "blueviolet", "--", 1.8),
    ]

    solver_configs: dict[str, tuple[Path, str, str, float]] = {
        label: (path, color, ls, lw)
        for label, path, color, ls, lw in _all_configs
        if path is not None
    }

    data: dict = {}
    for label, (directory, color, ls, lw) in solver_configs.items():
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
        print("Not enough data to produce dissipation comparison plot.")
        return

    _trim_to_reference_length(data, "LES - SGSP", {"DNS", "Projection"})

    for label, entry in data.items():
        entry["dissipation"] = _compute_dissipation_series(
            entry["solutions"], entry["coords"], viscosity
        )
        wavenumbers, diss_spectrum = _compute_dissipation_spectrum(
            entry["solutions"][-1], domain_length, viscosity
        )
        entry["wavenumbers"] = wavenumbers
        entry["diss_spectrum"] = diss_spectrum

    all_times = next(iter(data.values()))["times"]
    t_end = float(all_times[-1])
    t_window_start = max(t_end - 1.0, t_end * 0.8)

    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.32)
    ax_diss = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_spectrum = fig.add_subplot(gs[0, 2])

    # Panel 1: full dissipation evolution
    for label, entry in data.items():
        ax_diss.plot(
            entry["times"],
            entry["dissipation"],
            color=entry["color"],
            linestyle=entry["ls"],
            linewidth=entry["lw"],
            label=label,
        )
    ax_diss.set_xlabel("Time $t$")
    ax_diss.set_ylabel(r"$\nu\int\left(\frac{\partial u}{\partial x}\right)^2 dx$")
    ax_diss.set_title("Dissipation rate")
    ax_diss.legend()
    ax_diss.grid(True, alpha=0.25)

    # Panel 2: windowed dissipation evolution
    for label, entry in data.items():
        times_arr = np.array(entry["times"])
        diss_arr = np.array(entry["dissipation"])
        mask = times_arr >= t_window_start
        ax_zoom.plot(
            times_arr[mask],
            diss_arr[mask],
            color=entry["color"],
            linestyle=entry["ls"],
            linewidth=entry["lw"],
            label=label,
        )
    ax_zoom.set_xlabel("Time $t$")
    ax_zoom.set_ylabel(r"$\nu\int\left(\frac{\partial u}{\partial x}\right)^2 dx$")
    ax_zoom.set_title(f"Dissipation rate ($t \\geq {t_window_start:.2f}$)")
    ax_zoom.legend()
    ax_zoom.grid(True, alpha=0.25)

    # Panel 3: dissipation spectrum D(k) = 2ν k² E(k) at t_final
    for label, entry in data.items():
        ax_spectrum.loglog(
            entry["wavenumbers"],
            entry["diss_spectrum"],
            color=entry["color"],
            linestyle=entry["ls"],
            linewidth=entry["lw"],
            label=label,
        )
    ax_spectrum.set_xlabel("Wavenumber $k$")
    ax_spectrum.set_ylabel("$D(k) = 2\\nu k^2 E(k)$")
    ax_spectrum.set_title("Dissipation spectrum at $t_{\\mathrm{final}}$")
    ax_spectrum.legend()
    ax_spectrum.grid(True, which="both", alpha=0.2)

    fig.suptitle(
        "Dissipation diagnostics: DNS vs LES variants",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    save_path = output_path / "dissipation_comparison.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved dissipation comparison plot to '{save_path}'.")
    plt.close(fig)
