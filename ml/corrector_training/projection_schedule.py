"""Time-varying DNS reference for the AV corrector reward.

Loads all per-snapshot CSV files written by BurgersBase.write_solution_to_csv
(format: ``sol_t{time:.6f}.csv``, columns: node_index, x_coordinate, velocity,
forcing) from a DNS run directory.  At query time it returns the energy spectrum
and dissipation rate interpolated to the requested simulation time, so the RL
reward always compares the LES state against the DNS state at the *same* physical
time rather than a fixed terminal snapshot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Needed for _load_snapshot_csv — import at module level.
import csv

logger = logging.getLogger(__name__)


class ProjectionReferenceSchedule:
    """Pre-computed Projection spectra and dissipation rates indexed by simulation time.

    Parameters
    ----------
    snapshot_times:
        Sorted 1-D array of DNS snapshot times, shape (T,).
    spectra_array:
        DNS energy spectra at each snapshot, shape (T, K).
        Column k holds E_DNS(k+1, t) (positive wavenumbers, DC skipped).
    n_wavenumber_bins:
        K — number of positive wavenumber bins kept (must match LES state dim).
    """

    def __init__(
        self,
        snapshot_times: NDArray,
        spectra_array: NDArray,
        n_wavenumber_bins: int,
    ) -> None:
        if snapshot_times.ndim != 1:
            raise ValueError("snapshot_times must be a 1-D array.")
        if spectra_array.shape != (len(snapshot_times), n_wavenumber_bins):
            raise ValueError(
                f"spectra_array shape {spectra_array.shape} does not match "
                f"(T={len(snapshot_times)}, K={n_wavenumber_bins})."
            )


        self._snapshot_times = snapshot_times
        self._spectra_array = spectra_array
        self.n_wavenumber_bins = n_wavenumber_bins
        self.t_min: float = float(snapshot_times[0])
        self.t_max: float = float(snapshot_times[-1])

    @classmethod
    def from_projection_directory(
        cls,
        projection_dir: Path,
        domain_length: float,
        n_wavenumber_bins: int,
    ) -> "ProjectionReferenceSchedule":
        """Build a schedule from projected LES-grid snapshots stored as CSV files."""
        projection_dir = Path(projection_dir)

        csv_files = sorted(projection_dir.glob("sol_t*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No projected solution CSVs found in {projection_dir}"
            )

        snapshot_times_list: list[float] = []
        spectra_list: list[NDArray] = []

        for csv_path in csv_files:
            time_value = float(csv_path.stem.replace("sol_t", ""))
            velocity_array = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=2)
            snapshot_times_list.append(time_value)
            spectra_list.append(
                _compute_spectrum_bins(
                    velocity_array=velocity_array,
                    domain_length=domain_length,
                    n_wavenumber_bins=n_wavenumber_bins,
                )
            )

        snapshot_times_array = np.array(snapshot_times_list, dtype=np.float64)
        spectra_array = np.stack(spectra_list, axis=0).astype(np.float64)

        sort_indices = np.argsort(snapshot_times_array)
        snapshot_times_array = snapshot_times_array[sort_indices]
        spectra_array = spectra_array[sort_indices]

        logger.info(
            "ProjectionReferenceSchedule loaded %d snapshots from %s (t=[%.4f, %.4f], K=%d).",
            len(snapshot_times_array),
            projection_dir,
            snapshot_times_array[0],
            snapshot_times_array[-1],
            n_wavenumber_bins,
        )

        return cls(
            snapshot_times=snapshot_times_array,
            spectra_array=spectra_array,
            n_wavenumber_bins=n_wavenumber_bins,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, t: float) -> NDArray:
        """Return DNS spectrum and dissipation interpolated to time *t*.

        Linear interpolation between the two nearest snapshots.  Clamps to
        the boundary values outside [t_min, t_max].

        Parameters
        ----------
        t:
            Simulation time at which to evaluate the DNS reference.

        Returns
        -------
        dns_spectrum_k : float64 array, shape (K,)
            Interpolated DNS energy spectrum E_DNS(k, t).
        dns_dissipation : float
            Interpolated DNS dissipation rate ε_DNS(t).
        """
        t_clamped = float(np.clip(t, self.t_min, self.t_max))

        # np.interp handles scalar interpolation; for the 2-D spectrum we
        # interpolate each wavenumber bin independently via broadcasting.
        dns_spectrum_k = np.array(
            [
                np.interp(t_clamped, self._snapshot_times, self._spectra_array[:, k])
                for k in range(self.n_wavenumber_bins)
            ],
            dtype=np.float64,
        )
        return dns_spectrum_k

    def plot_schedule(
        self, query_times: NDArray | None = None, output_path: Path | None = None
    ) -> None:
        """Visualize all snapshot spectra and optionally overlay queried interpolations."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # --- spectrum panel ---
        for snap_idx, snap_time in enumerate(self._snapshot_times):
            axes[0].loglog(
                np.arange(1, self.n_wavenumber_bins + 1),
                self._spectra_array[snap_idx],
                color="steelblue",
                alpha=0.2,
                linewidth=0.8,
            )
        if query_times is not None:
            for query_t in query_times:
                queried_spectrum = self.query(query_t)
                axes[0].loglog(
                    np.arange(1, self.n_wavenumber_bins + 1),
                    queried_spectrum,
                    color="coral",
                    linewidth=1.5,
                    linestyle="--",
                    label=f"t={query_t:.2f}",
                )
        axes[0].set_xlabel("wavenumber k")
        axes[0].set_ylabel("E(k)")
        axes[0].set_title("Projected spectra (all snapshots + queries)")
        axes[0].legend(fontsize=8)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _load_snapshot_csv(csv_path: Path) -> tuple[NDArray, float]:
    """Read a ``sol_t{time:.6f}.csv`` snapshot and return (velocity, time).

    The time value is parsed from the filename rather than the file contents
    to avoid floating-point round-trip issues.
    """
    time_val = float(csv_path.stem.removeprefix("sol_t"))
    velocity_list: list[float] = []

    with csv_path.open(newline="") as file_handle:
        reader = csv.reader(file_handle)
        next(reader)  # skip header row
        for row in reader:
            # Columns: node_index, x_coordinate, velocity, forcing
            velocity_list.append(float(row[2]))

    return np.array(velocity_list, dtype=np.float64), time_val


def _compute_spectrum_bins(
    velocity_array: NDArray,
    domain_length: float,
    n_wavenumber_bins: int,
) -> NDArray:
    """Positive-wavenumber spectral energies, mirroring get_positive_spectrum.

    Index i corresponds to wavenumber k=i (DC included at i=0), matching the
    LES state vector exactly — n_wavenumber_bins must equal (n_nodes+1)//2.
    """
    n_nodes = len(velocity_array)
    u_hat = np.fft.fft(velocity_array)
    spectrum_full = 0.5 * np.abs(u_hat) ** 2 / n_nodes

    positive_indices = np.arange(0, (n_nodes + 1) // 2)
    spectrum_k = spectrum_full[positive_indices].astype(np.float64)

    if len(spectrum_k) != n_wavenumber_bins:
        raise ValueError(
            f"Computed {len(spectrum_k)} bins but expected n_wavenumber_bins="
            f"{n_wavenumber_bins} for n_nodes={n_nodes}."
        )
    return spectrum_k
