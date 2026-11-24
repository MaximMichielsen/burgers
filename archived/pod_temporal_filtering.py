"""Spatial-POD reduction and temporal coarsening of DNS forcing.

Automatically retain the smallest POD rank capturing at least 99% of the
forcing energy. For each LES timestep, save:
  - spatially reduced forcing at the original DNS timestep (no temporal filter)
  - the same reduced forcing after boxcar filtering and temporal subsampling

Also generate:
  1. POD energy spectrum and cumulative energy
  3. Temporal POD coefficient comparisons
  5. Relative forcing reconstruction errors

Input array convention: (time, spatial_nodes).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from setup.constants import T_ZERO


def determine_pod_rank(
    singular_values: np.ndarray,
    energy_fraction: float = 0.99,
) -> tuple[int, np.ndarray]:
    """Return the smallest POD rank capturing at least energy_fraction."""
    if not 0.0 < energy_fraction <= 1.0:
        raise ValueError("energy_fraction must be in (0, 1].")

    modal_energy = singular_values**2
    total_energy = np.sum(modal_energy)
    if total_energy <= 0.0:
        raise ValueError("The forcing has zero total POD energy.")

    cumulative_energy = np.cumsum(modal_energy) / total_energy
    n_modes = int(np.searchsorted(cumulative_energy, energy_fraction) + 1)
    return n_modes, cumulative_energy


def boxcar_filter_and_subsample(
    coefficients: np.ndarray,
    stride: int,
) -> np.ndarray:
    """Low-pass filter the time-dependent POD coefficients, then decimate."""
    if stride < 1:
        raise ValueError("The LES/DNS timestep ratio must be at least one.")
    if stride == 1:
        return coefficients.copy()

    kernel = np.ones(stride, dtype=float) / stride
    filtered = np.apply_along_axis(
        lambda signal: np.convolve(signal, kernel, mode="same"),
        axis=0,
        arr=coefficients,
    )
    return filtered[::stride, :]


def _save_energy_spectrum(
    singular_values: np.ndarray,
    cumulative_energy: np.ndarray,
    n_modes: int,
    energy_fraction: float,
    output_path: Path,
) -> None:
    """Plot individual and cumulative POD energy."""
    modal_energy = singular_values**2
    modal_energy_fraction = modal_energy / np.sum(modal_energy)
    mode_numbers = np.arange(1, len(singular_values) + 1)

    fig, ax_energy = plt.subplots(figsize=(9, 5))
    ax_energy.semilogy(
        mode_numbers,
        modal_energy_fraction,
        "o-",
        markersize=3,
        linewidth=1,
        label="Individual modal energy",
    )
    ax_energy.set_xlabel("POD mode")
    ax_energy.set_ylabel("Fraction of total energy (individual)")
    ax_energy.grid(True, which="both", alpha=0.3)

    ax_cumulative = ax_energy.twinx()
    ax_cumulative.plot(
        mode_numbers,
        100.0 * cumulative_energy,
        color="tab:orange",
        linewidth=2,
        label="Cumulative energy",
    )
    ax_cumulative.axhline(
        100.0 * energy_fraction,
        color="tab:green",
        linestyle="--",
        label=f"{100.0 * energy_fraction:g}% target",
    )
    ax_cumulative.axvline(
        n_modes,
        color="tab:red",
        linestyle=":",
        label=f"Selected rank: {n_modes}",
    )
    ax_cumulative.set_ylabel("Cumulative energy (%)")
    ax_cumulative.set_ylim(0, 101)

    handles_1, labels_1 = ax_energy.get_legend_handles_labels()
    handles_2, labels_2 = ax_cumulative.get_legend_handles_labels()
    ax_energy.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")
    ax_energy.set_title("POD energy spectrum")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_temporal_coefficients(
    coefficients_dns: np.ndarray,
    coefficients_les: np.ndarray,
    times_dns: np.ndarray,
    times_les: np.ndarray,
    dt_les: float,
    output_path: Path,
    n_plot_modes: int = 6,
) -> None:
    """Compare original DNS coefficients with filtered/subsampled LES samples."""
    n_plot = min(n_plot_modes, coefficients_dns.shape[1])
    fig, axes = plt.subplots(n_plot, 1, figsize=(11, max(2.5 * n_plot, 4)), sharex=True)
    if n_plot == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        ax.plot(
            times_dns,
            coefficients_dns[:, k],
            linewidth=0.9,
            label="Original DNS coefficient",
        )
        ax.plot(
            times_les,
            coefficients_les[:, k],
            "o-",
            markersize=2.5,
            linewidth=1,
            label="Filtered + subsampled coefficient",
        )
        ax.set_ylabel(f"$a_{k + 1}(t)$")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel("Time")
    fig.suptitle(
        f"Temporal POD coefficients (first {n_plot} modes), "
        f"$\\Delta t_{{LES}}={dt_les:g}$"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _relative_error_over_time(
    reference: np.ndarray,
    approximation: np.ndarray,
) -> np.ndarray:
    """Spatial relative L2 error at each time, with safe zero handling."""
    numerator = np.linalg.norm(reference - approximation, axis=1)
    denominator = np.linalg.norm(reference, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > np.finfo(float).eps,
    )


def _save_forcing_errors(
    forcing_dns: np.ndarray,
    forcing_reduced_dns_dt: np.ndarray,
    forcing_reduced_les_dt: np.ndarray,
    times_dns: np.ndarray,
    times_les: np.ndarray,
    dt_les: float,
    output_path: Path,
) -> tuple[float, float]:
    """Plot per-time relative forcing errors and return global relative errors."""
    # Compare the LES forcing on the DNS grid by interpolating its temporal
    # coefficients implicitly through the reconstructed spatial field.
    # np.interp is applied independently to each spatial degree of freedom.
    forcing_les_on_dns_grid = np.empty_like(forcing_dns, dtype=float)
    for j in range(forcing_dns.shape[1]):
        coarse_signal = forcing_reduced_les_dt[:, j]
        fine_signal = np.interp(times_dns, times_les, coarse_signal)
        if len(times_les) >= 2:
            after_last = times_dns > times_les[-1]
            slope = (coarse_signal[-1] - coarse_signal[-2]) / (
                times_les[-1] - times_les[-2]
            )
            fine_signal[after_last] = coarse_signal[-1] + slope * (
                times_dns[after_last] - times_les[-1]
            )
        forcing_les_on_dns_grid[:, j] = fine_signal

    error_spatial_reduction = _relative_error_over_time(
        forcing_dns, forcing_reduced_dns_dt
    )
    error_total_reduction = _relative_error_over_time(
        forcing_dns, forcing_les_on_dns_grid
    )

    global_error_dns_dt = np.linalg.norm(
        forcing_dns - forcing_reduced_dns_dt
    ) / np.linalg.norm(forcing_dns)
    global_error_les_dt = np.linalg.norm(
        forcing_dns - forcing_les_on_dns_grid
    ) / np.linalg.norm(forcing_dns)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        times_dns,
        100.0 * error_spatial_reduction,
        label=f"Spatial POD truncation (global: {100 * global_error_dns_dt:.3f}%)",
        linewidth=1,
    )
    ax.plot(
        times_dns,
        100.0 * error_total_reduction,
        label=f"POD + temporal coarsening (global: {100 * global_error_les_dt:.3f}%)",
        linewidth=1,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative spatial forcing error (%)")
    ax.set_title(f"Forcing reconstruction error, $\\Delta t_{{LES}}={dt_les:g}$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return float(global_error_dns_dt), float(global_error_les_dt)


def process_forcing(
    forcing_file: Path,
    output_dir: Path,
    dt_dns: float = 1e-4,
    dt_les: float = 1e-2,
    energy_fraction: float = 0.99,
    energy_plot_path: Path | None = None,
) -> tuple[int, float]:
    """Write both forcing datasets and their diagnostic plots."""
    forcing_file = Path(forcing_file)
    output_dir = Path(output_dir)

    forcing_dns = np.load(forcing_file)
    if forcing_dns.ndim != 2:
        raise ValueError(
            f"Expected forcing with shape (T, spatial_nodes), got {forcing_dns.shape}."
        )

    stride_float = dt_les / dt_dns
    stride = int(round(stride_float))
    if stride < 1 or not np.isclose(stride_float, stride, rtol=1e-9, atol=1e-9):
        raise ValueError(
            f"dt_les/dt_dns must be a positive integer; got {stride_float:g}."
        )

    print(f"Loaded forcing: {forcing_dns.shape}")
    print("Computing temporal POD (SVD)...")
    temporal_modes, singular_values, spatial_modes = np.linalg.svd(
        forcing_dns, full_matrices=False
    )

    n_modes, cumulative_energy = determine_pod_rank(singular_values, energy_fraction)
    captured = cumulative_energy[n_modes - 1]
    print(
        f"Retaining {n_modes}/{len(singular_values)} modes "
        f"({100.0 * captured:.6f}% energy)."
    )

    # POD temporal coefficients are U*S. Reconstruct using the first N modes.
    coefficients_dns = temporal_modes[:, :n_modes] * singular_values[:n_modes]
    spatial_modes_reduced = spatial_modes[:n_modes, :]

    # DNS-dt output: spatially reduced only; no temporal filtering.
    forcing_reduced_dns_dt = coefficients_dns @ spatial_modes_reduced
    times_dns = T_ZERO + np.arange(forcing_dns.shape[0]) * dt_dns

    # LES-dt output: filter and subsample the same retained POD coefficients.
    coefficients_les = boxcar_filter_and_subsample(coefficients_dns, stride)
    forcing_reduced_les_dt = coefficients_les @ spatial_modes_reduced
    times_les = T_ZERO + np.arange(forcing_reduced_les_dt.shape[0]) * stride * dt_dns

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "forcing_dns_dt.npy", forcing_reduced_dns_dt)
    np.save(output_dir / "times_dns_dt.npy", times_dns)
    np.save(output_dir / "forcing_les_dt.npy", forcing_reduced_les_dt)
    np.save(output_dir / "times_les_dt.npy", times_les)
    np.save(output_dir / "pod_singular_values.npy", singular_values)
    np.save(output_dir / "pod_cumulative_energy.npy", cumulative_energy)

    if energy_plot_path is not None:
        energy_plot_path.parent.mkdir(parents=True, exist_ok=True)
        if not energy_plot_path.exists():
            _save_energy_spectrum(
                singular_values,
                cumulative_energy,
                n_modes,
                energy_fraction,
                energy_plot_path,
            )

    _save_temporal_coefficients(
        coefficients_dns,
        coefficients_les,
        times_dns,
        times_les,
        dt_les,
        plots_dir / "03_temporal_coefficients.png",
    )

    global_error_dns_dt, global_error_les_dt = _save_forcing_errors(
        forcing_dns,
        forcing_reduced_dns_dt,
        forcing_reduced_les_dt,
        times_dns,
        times_les,
        dt_les,
        plots_dir / "05_forcing_errors.png",
    )

    print(f"Saved DNS-dt forcing: {forcing_reduced_dns_dt.shape}")
    print(f"Saved LES-dt forcing: {forcing_reduced_les_dt.shape}")
    print(f"Global relative forcing error (POD only): {100 * global_error_dns_dt:.6f}%")
    print(
        f"Global relative forcing error (POD + temporal): {100 * global_error_les_dt:.6f}%"
    )
    print(f"Saved plots to: {plots_dir}")

    return n_modes, float(captured)


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.resolve()

    # Change this path if fz.npy is stored elsewhere.
    forcing_file = ROOT_DIR / "dns_data" / "partitioned" / "dat" / "f.npy"
    output_dir = ROOT_DIR / "dns_data" / "temporal" / "pod_99_percent"

    dt_dns = 1e-4
    dt_les_values = [1e-1, 5e-2, 1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 1e-4]
    energy_plot_path = output_dir / "plots" / "01_energy_spectrum.png"

    for dt_les in dt_les_values:
        print(f"\nProcessing dt_les={dt_les:g} (dt_dns={dt_dns:g})")
        case_dir = output_dir / f"dt_les_{dt_les:g}"
        process_forcing(
            forcing_file=forcing_file,
            output_dir=case_dir,
            dt_dns=dt_dns,
            dt_les=dt_les,
            energy_fraction=0.99,
            energy_plot_path=energy_plot_path,
        )
