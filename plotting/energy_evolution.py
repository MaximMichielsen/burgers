"""Energy comparison plot: DNS vs LES-A vs LES-NM vs LES-ANN.

Reads directly from solver CSV outputs and produces:
    1. Total kinetic energy over time  (½∫u² dx)
    2. Energy spectra at t_final       (log-log E(k) vs k)
    3. Energy dissipation over time    (ν∫(∂u/∂x)² dx)

Usage
-----
Set RUN_DIR to an existing run folder, then run as a standalone script
or call plot_energy_comparison() from main.py after all solver runs.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Helpers: read solver output
# ---------------------------------------------------------------------------


def _read_snapshots(
    directory: Path,
) -> tuple[list[float], list[NDArray], list[NDArray]]:
    """Read all sol_t*.csv files from a solver output directory.

    Returns
    -------
    times : sorted list of snapshot times
    solutions : list of velocity arrays (one per snapshot)
    x_coords : list of coordinate arrays (one per snapshot, usually identical)
    """
    files = sorted(directory.glob("sol_t*.csv"))
    if not files:
        raise FileNotFoundError(f"No sol_t*.csv found in {directory}")

    times, solutions, coords = [], [], []
    for file_path in files:
        time_val = float(file_path.stem.split("t")[-1])
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        times.append(time_val)
        coords.append(data[:, 1])
        solutions.append(data[:, 2])

    order = np.argsort(times)
    times = [times[i] for i in order]
    solutions = [solutions[i] for i in order]
    coords = [coords[i] for i in order]
    return times, solutions, coords


def _compute_energy_series(
    solutions: list[NDArray],
    coords: list[NDArray],
) -> NDArray:
    """Compute ½∫u² dx via trapezoidal rule for each snapshot."""
    return np.array([0.5 * np.trapezoid(u**2, x=x) for u, x in zip(solutions, coords)])


def _compute_dissipation_series(
    solutions: list[NDArray],
    coords: list[NDArray],
    viscosity: float,
) -> NDArray:
    """Compute ν∫(∂u/∂x)² dx via central differences + trapezoidal rule."""
    dissipation_vals = []
    for u_sol, x_coord in zip(solutions, coords):
        dx_val = x_coord[1] - x_coord[0]
        du_dx = np.gradient(u_sol, dx_val)
        dissipation_vals.append(viscosity * np.trapezoid(du_dx**2, x=x_coord))
    return np.array(dissipation_vals)


def _compute_energy_spectrum(
    solution: NDArray,
    domain_length: float,
) -> tuple[NDArray, NDArray]:
    """Return positive wavenumbers and spectral energy."""
    n_pts = len(solution)
    u_hat = np.fft.rfft(solution)
    freqs = np.fft.rfftfreq(n_pts, d=domain_length / n_pts)
    wavenumbers = 2.0 * np.pi * freqs
    spectrum = 0.5 * np.abs(u_hat) ** 2 / n_pts
    return wavenumbers, spectrum


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------


def plot_energy_comparison(
    dns_dir: Path,
    les_a_dir: Path,
    les_nm_dir: Path,
    les_ann_dir: Path,
    output_path: Path,
    viscosity: float = 0.01,
    domain_length: float = 1.0,
) -> None:
    """Read solver outputs and produce a 3-panel energy comparison figure.

    Parameters
    ----------
    dns_dir, les_a_dir, les_nm_dir, les_ann_dir:
        Paths to solver output directories (containing sol_t*.csv).
    output_path:
        Directory where the figure is saved.
    viscosity:
        Kinematic viscosity ν (for dissipation computation).
    domain_length:
        Physical domain length L.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    configs = {
        "DNS": (dns_dir, "dimgray", "-", 1.8),
        "LES-A": (les_a_dir, "royalblue", "--", 1.4),
        "LES-NM": (les_nm_dir, "tab:orange", "-.", 1.4),
        "LES-ANN": (les_ann_dir, "crimson", ":", 1.8),
    }

    data: dict = {}
    for label, (directory, color, ls, lw) in configs.items():
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

    if len(data) < 2:
        print("Not enough data to produce comparison plot.")
        return

    # --- Compute diagnostics ---
    for label, entry in data.items():
        entry["energy"] = _compute_energy_series(entry["solutions"], entry["coords"])
        entry["dissipation"] = _compute_dissipation_series(
            entry["solutions"], entry["coords"], viscosity
        )
        # Spectrum at final snapshot
        final_u = entry["solutions"][-1]
        final_x = entry["coords"][-1]
        wn, sp = _compute_energy_spectrum(final_u, domain_length)
        # Keep only positive non-zero wavenumbers
        mask = wn > 0
        entry["wavenumbers"] = wn[mask]
        entry["spectrum"] = sp[mask]

    # --- Figure layout ---
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.32)

    ax_energy = fig.add_subplot(gs[0, 0])
    ax_spectrum = fig.add_subplot(gs[0, 1])
    ax_dissipation = fig.add_subplot(gs[0, 2])

    # --- Panel 1: Total energy over time ---
    for label, entry in data.items():
        ax_energy.plot(
            entry["times"],
            entry["energy"],
            color=entry["color"],
            linestyle=entry["ls"],
            linewidth=entry["lw"],
            label=label,
        )
    ax_energy.set_xlabel("Time $t$", fontsize=11)
    ax_energy.set_ylabel(r"$\frac{1}{2}\int u^2\,dx$", fontsize=11)
    ax_energy.set_title("Total kinetic energy", fontsize=11)
    ax_energy.legend(fontsize=9)
    ax_energy.grid(True, alpha=0.25)

    # --- Panel 2: Energy spectrum at t_final ---
    for label, entry in data.items():
        ax_spectrum.loglog(
            entry["wavenumbers"],
            entry["spectrum"],
            color=entry["color"],
            linestyle=entry["ls"],
            linewidth=entry["lw"],
            label=label,
        )

    # Reference -5/3 slope
    wn_ref = data[next(iter(data))]["wavenumbers"]
    k_mid = wn_ref[len(wn_ref) // 3]
    e_mid = data[next(iter(data))]["spectrum"][len(wn_ref) // 3]
    slope_line = e_mid * (wn_ref / k_mid) ** (-5 / 3)
    ax_spectrum.loglog(
        wn_ref,
        slope_line,
        color="lightgray",
        linestyle="--",
        linewidth=1.0,
        label=r"$k^{-5/3}$",
    )

    ax_spectrum.set_xlabel("Wavenumber $k$", fontsize=11)
    ax_spectrum.set_ylabel("$E(k)$", fontsize=11)
    ax_spectrum.set_title("Energy spectrum at $t_{final}$", fontsize=11)
    ax_spectrum.legend(fontsize=9)
    ax_spectrum.grid(True, which="both", alpha=0.2)

    # --- Panel 3: Dissipation over time ---
    for label, entry in data.items():
        ax_dissipation.plot(
            entry["times"],
            entry["dissipation"],
            color=entry["color"],
            linestyle=entry["ls"],
            linewidth=entry["lw"],
            label=label,
        )
    ax_dissipation.set_xlabel("Time $t$", fontsize=11)
    ax_dissipation.set_ylabel(
        r"$\nu\int\left(\partial u/\partial x\right)^2\,dx$", fontsize=11
    )
    ax_dissipation.set_title("Viscous dissipation", fontsize=11)
    ax_dissipation.legend(fontsize=9)
    ax_dissipation.grid(True, alpha=0.25)

    fig.suptitle(
        "Energy diagnostics: DNS vs LES variants",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    save_path = output_path / "energy_comparison.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved energy comparison plot to '{save_path}'.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Edit this to point to your run directory
    RUN_DIR = Path("runs/run_robijns_one_0520_141710")

    plot_energy_comparison(
        dns_dir=RUN_DIR / "solver_data/DNS",
        les_a_dir=RUN_DIR / "solver_data/LES_A",
        les_nm_dir=RUN_DIR / "solver_data/LES_NM",
        les_ann_dir=RUN_DIR / "solver_data/LES_ANN",
        output_path=RUN_DIR,
        viscosity=0.01,
        domain_length=1.0,
    )
