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


class DNSReferenceSchedule:
    """Pre-computed DNS spectra and dissipation rates indexed by simulation time.

    Parameters
    ----------
    snapshot_times:
        Sorted 1-D array of DNS snapshot times, shape (T,).
    spectra_array:
        DNS energy spectra at each snapshot, shape (T, K).
        Column k holds E_DNS(k+1, t) (positive wavenumbers, DC skipped).
    dissipation_array:
        DNS dissipation rates at each snapshot, shape (T,).
    n_wavenumber_bins:
        K — number of positive wavenumber bins kept (must match LES state dim).
    """

    def __init__(
        self,
        snapshot_times: NDArray,
        spectra_array: NDArray,
        dissipation_array: NDArray,
        n_wavenumber_bins: int,
    ) -> None:
        if snapshot_times.ndim != 1:
            raise ValueError("snapshot_times must be a 1-D array.")
        if spectra_array.shape != (len(snapshot_times), n_wavenumber_bins):
            raise ValueError(
                f"spectra_array shape {spectra_array.shape} does not match "
                f"(T={len(snapshot_times)}, K={n_wavenumber_bins})."
            )
        if dissipation_array.shape != (len(snapshot_times),):
            raise ValueError(
                f"dissipation_array shape {dissipation_array.shape} does not match "
                f"(T={len(snapshot_times)},)."
            )

        self._snapshot_times = snapshot_times
        self._spectra_array = spectra_array
        self._dissipation_array = dissipation_array
        self.n_wavenumber_bins = n_wavenumber_bins
        self.t_min: float = float(snapshot_times[0])
        self.t_max: float = float(snapshot_times[-1])

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        dns_dir: Path,
        domain_length: float,
        viscosity: float,
        n_wavenumber_bins: int,
    ) -> "DNSReferenceSchedule":
        """Build a schedule by reading all ``sol_t*.csv`` files in *dns_dir*.

        Parameters
        ----------
        dns_dir:
            Directory containing the DNS solver output CSVs.
        domain_length:
            Physical domain length L (needed for wavenumber computation).
        viscosity:
            Physical viscosity ν (needed for dissipation computation).
        n_wavenumber_bins:
            K — number of positive wavenumber bins to keep.  Should match
            ``N_LES // 2`` so the schedule aligns with the LES state vector.
        """
        dns_dir = Path(dns_dir)
        snapshot_csv_files = sorted(dns_dir.glob("sol_t*.csv"))
        if not snapshot_csv_files:
            raise FileNotFoundError(
                f"No 'sol_t*.csv' snapshot files found in {dns_dir}."
            )

        snapshot_times_list: list[float] = []
        spectra_list: list[NDArray] = []
        dissipation_list: list[float] = []

        for csv_path in snapshot_csv_files:
            velocity_array, time_val = _load_snapshot_csv(csv_path)
            spectrum_k = _compute_spectrum_bins(
                velocity_array=velocity_array,
                domain_length=domain_length,
                n_wavenumber_bins=n_wavenumber_bins,
            )
            dissipation_val = _compute_dissipation(
                velocity_array=velocity_array,
                domain_length=domain_length,
                viscosity=viscosity,
            )
            snapshot_times_list.append(time_val)
            spectra_list.append(spectrum_k)
            dissipation_list.append(dissipation_val)

        snapshot_times_array = np.array(snapshot_times_list, dtype=np.float64)
        spectra_array = np.stack(spectra_list, axis=0).astype(np.float64)
        dissipation_array = np.array(dissipation_list, dtype=np.float64)

        # Sort by time (glob order may not be strictly chronological on all OS).
        sort_indices = np.argsort(snapshot_times_array)
        snapshot_times_array = snapshot_times_array[sort_indices]
        spectra_array = spectra_array[sort_indices]
        dissipation_array = dissipation_array[sort_indices]

        logger.info(
            "DNSReferenceSchedule loaded %d snapshots from %s (t=[%.4f, %.4f], K=%d).",
            len(snapshot_times_array),
            dns_dir,
            snapshot_times_array[0],
            snapshot_times_array[-1],
            n_wavenumber_bins,
        )

        return cls(
            snapshot_times=snapshot_times_array,
            spectra_array=spectra_array,
            dissipation_array=dissipation_array,
            n_wavenumber_bins=n_wavenumber_bins,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, t: float) -> tuple[NDArray, float]:
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
        dns_dissipation = float(
            np.interp(t_clamped, self._snapshot_times, self._dissipation_array)
        )
        return dns_spectrum_k, dns_dissipation

    def plot_schedule(
        self, query_times: NDArray | None = None, output_path: Path | None = None
    ) -> None:
        """Visualise all snapshot spectra and optionally overlay queried interpolations."""
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
                queried_spectrum, _ = self.query(query_t)
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
        axes[0].set_title("DNS spectra (all snapshots + queries)")
        axes[0].legend(fontsize=8)

        # --- dissipation panel ---
        axes[1].plot(
            self._snapshot_times,
            self._dissipation_array,
            color="steelblue",
            linewidth=1.5,
        )
        if query_times is not None:
            for query_t in query_times:
                _, queried_diss = self.query(query_t)
                axes[1].axvline(query_t, color="coral", linestyle="--", linewidth=1.0)
                axes[1].scatter([query_t], [queried_diss], color="coral", zorder=5)
        axes[1].set_xlabel("time t")
        axes[1].set_ylabel("dissipation ε")
        axes[1].set_title("DNS dissipation over time")

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
    """Return the first *n_wavenumber_bins* positive-wavenumber spectral energies.

    Mirrors BurgersBase.compute_energy_spectrum + get_positive_spectrum,
    keeping only bins k = 1 … K (DC component at index 0 is dropped).

    Returns
    -------
    spectrum_k : float64 array, shape (n_wavenumber_bins,)
    """
    n_nodes = len(velocity_array)
    u_hat = np.fft.fft(velocity_array)
    spectrum_full = 0.5 * np.abs(u_hat) ** 2 / n_nodes

    # Positive wavenumber indices (excluding DC at index 0).
    positive_indices = np.arange(1, n_nodes // 2 + 1)
    positive_spectrum = spectrum_full[positive_indices]

    # Trim or zero-pad to exactly n_wavenumber_bins.
    spectrum_k = np.zeros(n_wavenumber_bins, dtype=np.float64)
    n_available = min(len(positive_spectrum), n_wavenumber_bins)
    spectrum_k[:n_available] = positive_spectrum[:n_available]
    return spectrum_k


def _compute_dissipation(
    velocity_array: NDArray,
    domain_length: float,
    viscosity: float,
) -> float:
    """Approximate ν · ∫(∂u/∂x)² dx via a spectral estimate.

    Uses Parseval's theorem: ∫(∂u/∂x)² dx = Σ_k k² |û_k|² / N²,
    which is exact for the periodic Fourier representation of the DNS
    solution and avoids re-implementing the FEM quadrature loop here.

    Parameters
    ----------
    velocity_array:
        Nodal velocity values from one DNS snapshot.
    domain_length:
        Physical domain length L.
    viscosity:
        Physical kinematic viscosity ν.
    """
    n_nodes = len(velocity_array)
    u_hat = np.fft.fft(velocity_array)
    wavenumbers_all = np.fft.fftfreq(n_nodes, d=domain_length / n_nodes) * 2.0 * np.pi
    dissipation_spectral = float(
        viscosity * np.sum(wavenumbers_all**2 * np.abs(u_hat) ** 2) / (n_nodes**2)
    )
    return dissipation_spectral
