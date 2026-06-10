"""Energy comparison plots: DNS vs LES-A vs LES-NM vs LES-ANN.

Produces a 3-panel figure: total kinetic energy, energy spectrum at t_final,
and viscous dissipation over time. Can be called from the pipeline or run
as a standalone script.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_snapshots(
    directory: Path,
) -> tuple[list[float], list[NDArray], list[NDArray]]:
    """Read all sol_t*.csv files; return (times, solutions, coords) sorted by time."""
    files = sorted(directory.glob("sol_t*.csv"))
    if not files:
        raise FileNotFoundError(f"No sol_t*.csv found in {directory}")

    times, solutions, coords = [], [], []
    for file_path in files:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        times.append(float(file_path.stem.split("t")[-1]))
        coords.append(data[:, 1])
        solutions.append(data[:, 2])

    order = np.argsort(times)
    return (
        [times[i] for i in order],
        [solutions[i] for i in order],
        [coords[i] for i in order],
    )


def _compute_energy_series(solutions: list[NDArray], coords: list[NDArray]) -> NDArray:
    """Compute ½∫u² dx per snapshot via trapezoidal rule."""
    return np.array([0.5 * np.trapezoid(u**2, x=x) for u, x in zip(solutions, coords)])


def _compute_dissipation_series(
    solutions: list[NDArray],
    coords: list[NDArray],
    viscosity: float,
) -> NDArray:
    """Compute ν∫(∂u/∂x)² dx per snapshot via central differences."""
    return np.array(
        [
            viscosity * np.trapezoid(np.gradient(u, x[1] - x[0]) ** 2, x=x)
            for u, x in zip(solutions, coords)
        ]
    )


def _compute_energy_spectrum(
    solution: NDArray, domain_length: float
) -> tuple[NDArray, NDArray]:
    """Return positive wavenumbers and spectral energy via rfft."""
    n_pts = len(solution)
    u_hat = np.fft.rfft(solution)
    wavenumbers = 2.0 * np.pi * np.fft.rfftfreq(n_pts, d=domain_length / n_pts)
    spectrum = 0.5 * np.abs(u_hat) ** 2 / n_pts
    return wavenumbers, spectrum


def _smooth_series(values: NDArray, window: int = 15) -> NDArray:
    """Rolling mean with edge-replication padding to avoid boundary artifacts."""
    pad_width = window // 2
    padded = np.pad(values, pad_width, mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(values)]


# Labels for solvers that exhibit oscillatory SGSP-coupled behaviour.
_OSCILLATORY_LABELS = frozenset({"LES-SGSP", "LES-AVCG", "LES-AVCL"})


def _plot_series(
    ax: plt.Axes,
    data: dict,
    x_key: str,
    y_key: str,
    smooth_window: int = 15,
) -> None:
    """Plot x_key vs y_key for all entries in data.

    Oscillatory solvers (SGSP-coupled) are drawn with a faded raw line and
    an opaque smoothed overlay so the trend remains readable.
    """
    for label, entry in data.items():
        x_values = entry[x_key]
        y_values = entry[y_key]
        color = entry["color"]
        ls = entry["ls"]
        lw = entry["lw"]

        if label in _OSCILLATORY_LABELS:
            # Raw oscillatory line, faded
            ax.plot(
                x_values,
                y_values,
                color=color,
                linestyle=ls,
                linewidth=lw * 0.6,
                alpha=0.25,
            )
            # Smoothed trend overlay, full opacity
            ax.plot(
                x_values,
                _smooth_series(y_values, window=smooth_window),
                color=color,
                linestyle=ls,
                linewidth=lw,
                alpha=1.0,
                label=label,
            )
        else:
            ax.plot(
                x_values,
                y_values,
                color=color,
                linestyle=ls,
                linewidth=lw,
                label=label,
            )


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------


def plot_energy_comparison(
    dns_dir: Path,
    les_a_dir: Path,
    les_nm_dir: Path | None,
    les_sgsp_dir: Path | None,
    les_avcg_dir: Path | None,
    les_avcl_dir: Path | None,
    output_path: Path,
    viscosity: float = 0.01,
    domain_length: float = 1.0,
    projection_dir: Path | None = None,
) -> None:
    """Read solver outputs and produce a 3-panel energy comparison figure."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    solver_configs = {
        "DNS": (dns_dir, "gray", "-", 1.8),
        "LES-A": (les_a_dir, "tab:orange", "--", 1.4),
        "LES-NM": (les_nm_dir, "gold", "-.", 1.4),
        "LES-SGSP": (les_sgsp_dir, "salmon", "-", 1.8),
        "LES-AVCG": (les_avcg_dir, "royalblue", "-", 1.8),
        "LES-AVCL": (les_avcl_dir, "blueviolet", "--", 1.8),
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
            times_read, solutions_read, coords_read = _read_snapshots(directory)
            data[label] = {
                "times": np.array(times_read),
                "solutions": solutions_read,
                "coords": coords_read,
                "color": color,
                "ls": ls,
                "lw": lw,
            }
            print(f"  Loaded {label}: {len(times_read)} snapshots")
        except FileNotFoundError as err:
            print(f"  Skipping {label}: {err}")

    if projection_dir is not None:
        projection_dir = Path(projection_dir)
        solutions_proj = np.load(projection_dir / "solutions_projection.npy")
        times_proj = np.load(projection_dir / "times.npy")
        coords_proj = np.linspace(0, domain_length, solutions_proj.shape[1])

        solutions_proj_list = [solutions_proj[i] for i in range(len(solutions_proj))]
        coords_proj_list = [coords_proj] * len(solutions_proj)

        data["Projection"] = {
            "times": times_proj,
            "solutions": solutions_proj_list,
            "coords": coords_proj_list,
            "color": "lightgreen",
            "ls": "-",
            "lw": 1.2,
        }
        data["Projection"]["energy"] = _compute_energy_series(
            solutions_proj_list, coords_proj_list
        )
        data["Projection"]["dissipation"] = _compute_dissipation_series(
            solutions_proj_list, coords_proj_list, viscosity
        )
        wn, sp = _compute_energy_spectrum(solutions_proj[-1], domain_length)
        mask = wn > 0
        data["Projection"]["wavenumbers"] = wn[mask]
        data["Projection"]["spectrum"] = sp[mask]

    ordered_keys = []
    for key in data:
        ordered_keys.append(key)
        if key == "DNS" and "Projection" in data:
            ordered_keys.append("Projection")

    data = {k: data[k] for k in ordered_keys if k in data}

    if len(data) < 2:
        print("Not enough data to produce comparison plot.")
        return

    for label, entry in data.items():
        if "energy" not in entry:
            entry["energy"] = _compute_energy_series(
                entry["solutions"], entry["coords"]
            )
        if "dissipation" not in entry:
            entry["dissipation"] = _compute_dissipation_series(
                entry["solutions"], entry["coords"], viscosity
            )
        if "wavenumbers" not in entry:
            wn, sp = _compute_energy_spectrum(entry["solutions"][-1], domain_length)
            mask = wn > 0
            entry["wavenumbers"] = wn[mask]
            entry["spectrum"] = sp[mask]

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
    first_entry = data[next(iter(data))]
    wn_ref = first_entry["wavenumbers"]
    mid = len(wn_ref) // 3
    slope_line = first_entry["spectrum"][mid] * (wn_ref / wn_ref[mid]) ** (-5 / 3)
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
    ax_spectrum.set_title("Energy spectrum at $t_{final}$")
    ax_spectrum.legend()
    ax_spectrum.grid(True, which="both", alpha=0.2)

    _plot_series(ax_dissipation, data, "times", "dissipation")
    ax_dissipation.set_xlabel("Time $t$")
    ax_dissipation.set_ylabel(r"$\nu\int\left(\partial u/\partial x\right)^2\,dx$")
    ax_dissipation.set_title("Viscous dissipation")
    ax_dissipation.legend()
    ax_dissipation.grid(True, alpha=0.25)

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
