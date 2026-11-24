from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from setup.constants import T_ZERO, DT_DNS, DT_LES

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.8,
    }
)

style = {
    "color": "#0072B2",
}


def boxcar_filter(
    forcing: np.ndarray,
    stride: int,
) -> np.ndarray:
    """Apply a temporal moving average while retaining DNS time resolution."""
    if stride < 1:
        raise ValueError("The LES/DNS timestep ratio must be at least one.")

    if stride == 1:
        return forcing.copy()

    kernel = np.ones(stride, dtype=float) / stride
    return np.apply_along_axis(
        lambda signal: np.convolve(signal, kernel, mode="same"),
        axis=0,
        arr=forcing,
    )


def relative_error_over_time(
    reference: np.ndarray,
    approximation: np.ndarray,
) -> np.ndarray:
    """Spatial relative L2 error at each time."""
    numerator = np.linalg.norm(reference - approximation, axis=1)
    denominator = np.linalg.norm(reference, axis=1)

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(float).eps,
    )


def process_forcing(
    forcing_file: Path,
    output_dir: Path,
    dt_dns: float = 1e-4,
    dt_les: float = 1e-3,
) -> float:
    """Filter DNS forcing, subsample to LES timestep, and generate diagnostics."""

    forcing_file = Path(forcing_file)
    output_dir = Path(output_dir)

    # Load DNS forcing.
    forcing_dns = np.load(forcing_file)

    if forcing_dns.ndim != 2:
        raise ValueError(
            f"Expected forcing with shape (T, spatial_nodes), got {forcing_dns.shape}."
        )

    # Determine temporal subsampling ratio.
    stride_float = dt_les / dt_dns
    stride = int(round(stride_float))

    if stride < 1 or not np.isclose(stride_float, stride, rtol=1e-9, atol=1e-9):
        raise ValueError(
            f"dt_les/dt_dns must be a positive integer; got {stride_float:g}."
        )

    print(f"Loaded DNS forcing: {forcing_dns.shape}")
    print(f"Filtering with boxcar width: {stride}")

    forcing_dns_filtered = boxcar_filter(forcing_dns, stride)
    times_dns = T_ZERO + np.arange(forcing_dns.shape[0]) * dt_dns

    # Create output directories.
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save the forcing variants for the four-run comparison.
    # The original DNS forcing remains available at forcing_file.
    np.save(output_dir / "f_dns_filtered.npy", forcing_dns_filtered)
    np.save(output_dir / "times.npy", times_dns)

    # Compare original and filtered forcing on the DNS time grid.
    # This isolates the effect of filtering, without subsampling/interpolation.
    errors = relative_error_over_time(forcing_dns, forcing_dns_filtered)

    denominator = np.linalg.norm(forcing_dns)
    global_error = (
        np.linalg.norm(forcing_dns - forcing_dns_filtered) / denominator
        if denominator > np.finfo(float).eps
        else 0.0
    )

    # Plot temporal forcing reconstruction error.
    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(
        times_dns,
        100.0 * errors,
        linewidth=1,
        label=f"DNS-grid boxcar filtering (global: {100 * global_error:.3f}%)",
        color=style["color"],
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Relative spatial forcing error (%)")
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(
        plots_dir / "01_forcing_error.png",
        dpi=300,
    )
    fig.savefig(
        plots_dir / "01_forcing_error.pdf",
        dpi=300,
    )
    plt.close(fig)

    # Print results.
    print(f"Saved DNS-grid filtered forcing: {forcing_dns_filtered.shape}")
    print(f"Saved DNS timestamps: {times_dns.shape}")
    print(
        "Global relative error (DNS vs. filtered DNS-grid forcing): "
        f"{100 * global_error:.6f}%"
    )
    print(f"Saved plots to: {plots_dir}")

    return float(global_error)


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.resolve()

    forcing_file = ROOT_DIR / "dns_data" / "partitioned" / "dat" / "f.npy"

    dt_dns = DT_DNS
    dt_les = DT_LES

    output_dir = ROOT_DIR / "dns_data" / "temporal" / f"boxcar_dt_{dt_les}"

    process_forcing(
        forcing_file=forcing_file,
        output_dir=output_dir,
        dt_dns=dt_dns,
        dt_les=dt_les,
    )
