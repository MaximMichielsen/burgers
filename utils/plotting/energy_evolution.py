"""Utility functions for energy evolution diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from old.utils.io_utils import read_data
from utils.pipeline_utils import RunPaths
from utils.plotting.configs_energy_and_dissipation import plotting_configs


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


def _trim_to_reference_length(
    data: dict, reference_label: str, labels_to_trim: set[str]
) -> None:
    """Trim entries in-place to match the snapshot count of the reference label."""
    if reference_label not in data:
        return
    n_ref = len(data[reference_label]["times"])
    for label in labels_to_trim:
        if label in data:
            entry = data[label]
            entry["times"] = entry["times"][:n_ref]
            entry["solutions"] = entry["solutions"][:n_ref]
            entry["coords"] = entry["coords"][:n_ref]


def plot_energy_comparison(
    paths: RunPaths,
    output_path: Path,
    domain_length: float,
) -> None:
    """Read CSV solver outputs and produce a 3-panel energy comparison figure."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    solver_configs: dict[str, tuple[Path, str, str, float]] = {}

    _all_configs = plotting_configs(paths)

    for label, path, color, linestyle, linewidth in _all_configs:
        if path is not None:
            solver_configs[label] = (path, color, linestyle, linewidth)

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
        print("Not enough data to produce comparison plot.")
        return

    _trim_to_reference_length(data, "LES - SGSP", {"DNS", "Projection"})

    for label, entry in data.items():
        entry["energy"] = _compute_energy_series(entry["solutions"], entry["coords"])
        wavenumbers, spectrum = _compute_energy_spectrum(
            entry["solutions"][-1], domain_length
        )
        entry["wavenumbers"] = wavenumbers
        entry["spectrum"] = spectrum

    # Determine window start for the zoom panel
    all_times = next(iter(data.values()))["times"]
    t_end = float(all_times[-1])
    t_window_start = max(t_end - 1.0, t_end * 0.8)

    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.32)
    ax_energy = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_spectrum = fig.add_subplot(gs[0, 2])

    # Panel 1: full energy evolution
    _plot_series(ax_energy, data, "times", "energy")
    ax_energy.set_xlabel("Time $t$")
    ax_energy.set_ylabel(r"$\frac{1}{2}\int u^2\,dx$")
    ax_energy.set_title("Total kinetic energy")
    ax_energy.legend()
    ax_energy.grid(True, alpha=0.25)

    # Panel 2: windowed energy evolution
    for label, entry in data.items():
        times_arr = np.array(entry["times"])
        energy_arr = np.array(entry["energy"])
        mask = times_arr >= t_window_start
        ax_zoom.plot(
            times_arr[mask],
            energy_arr[mask],
            color=entry["color"],
            linestyle=entry["ls"],
            linewidth=entry["lw"],
            label=label,
        )
    ax_zoom.set_xlabel("Time $t$")
    ax_zoom.set_ylabel(r"$\frac{1}{2}\int u^2\,dx$")
    ax_zoom.set_title(f"Total kinetic energy ($t \\geq {t_window_start:.2f}$)")
    ax_zoom.legend()
    ax_zoom.grid(True, alpha=0.25)

    # Panel 3: energy spectrum at t_final
    for label, entry in data.items():
        ax_spectrum.loglog(
            entry["wavenumbers"],
            entry["spectrum"],
            color=entry["color"],
            linewidth=entry["lw"],
            label=label,
            linestyle="--" if label in ("LES - SGSP", "LES - AVCG") else "-",
        )
    first_entry = next(iter(data.values()))
    wn_ref = first_entry["wavenumbers"]
    mid_idx = len(wn_ref) // 3
    slope_line = first_entry["spectrum"][mid_idx] * (wn_ref / wn_ref[mid_idx]) ** (
        -5 / 3
    )
    ax_spectrum.loglog(
        wn_ref,
        slope_line,
        color="lightgray",
        linestyle="--",
        linewidth=1.0,
        label=r"$k^{-5/3}$",
    )
    ax_spectrum.set_xlabel("Wavenumber $k$")
    ax_spectrum.set_ylabel("$E(k)$")
    ax_spectrum.set_title("Energy spectrum at $t_{\\mathrm{final}}$")
    ax_spectrum.legend()
    ax_spectrum.grid(True, which="both", alpha=0.2)

    fig.suptitle(
        "Energy diagnostics: DNS vs LES variants",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    save_path = output_path / "energy_comparison.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved energy comparison plot to '{save_path}'.")
    plt.close(fig)
