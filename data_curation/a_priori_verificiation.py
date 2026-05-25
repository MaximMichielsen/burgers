"""A priori verification suite for the VMS-ANN SGS predictor.

Diagnostics (Robijns 2019, Pusuluri 2021, Rajampeta 2022, Krochak 2023):
    1. Scatter plots       — prediction vs truth per output term (ρ, E_truth, E_pred)
    2. Metrics table       — ρ, MAE, MSE, E_truth, E_pred per term
    3. Time-series plots   — predicted vs exact for a single element
    4. Spatial profiles    — predicted vs exact over all elements at one snapshot
    5. Interaction-term spectrum — FFT of summed terms (spectral accuracy check)

Output terms (Rajampeta / Research Proposal convention):
    [0] (w_x, ū·u')_e       cross
    [1] (w_x, u'²/2)_e      Reynolds
    [2] (w_l, u'_t)_e       temporal left
    [3] (w_r, u'_t)_e       temporal right
    [4] (w_x, u'_x)_e       viscous SGS
"""

from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.stats import pearsonr

from constants import OUTPUT_UNITS

# ---------------------------------------------------------------------------
# Output term labels
# ---------------------------------------------------------------------------

OUTPUT_TERM_LABELS: list[str] = [
    r"$(w_x,\,\bar{u}u')_e$  [cross]",
    r"$(w_x,\,u'^2/2)_e$  [Reynolds]",
    r"$(w_l,\,u'_t)_e$  [temporal L]",
    r"$(w_r,\,u'_t)_e$  [temporal R]",
    r"viscous",
]
OUTPUT_TERM_SHORT_LABELS: list[str] = [
    "cross",
    "Reynolds",
    "temporal L",
    "temporal R",
    "viscous",
]
N_OUTPUT_TERMS: int = OUTPUT_UNITS

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_per_term_metrics(y_true: NDArray, y_pred: NDArray) -> dict[str, NDArray]:
    """Compute ρ, MAE, MSE, E_truth, E_pred for each output term.

    Parameters: y_true, y_pred of shape (n_samples, N_OUTPUT_TERMS).
    Returns dict with keys rho, mae, mse, e_truth, e_pred, each (N_OUTPUT_TERMS,).
    """
    diff = y_true - y_pred
    rho_vals = np.array(
        [float(pearsonr(y_true[:, i], y_pred[:, i])[0]) for i in range(y_true.shape[1])]
    )
    return {
        "rho": rho_vals,
        "mae": np.mean(np.abs(diff), axis=0),
        "mse": np.mean(diff**2, axis=0),
        "e_truth": y_true.mean(axis=0),
        "e_pred": y_pred.mean(axis=0),
    }


def print_metrics_table(
    metrics: dict[str, NDArray], dataset_label: str = "Validation"
) -> None:
    """Print a formatted per-term metrics table to stdout."""
    col_w = 14
    print(f"\n{'─' * 72}\n  A priori metrics — {dataset_label} set\n{'─' * 72}")
    print(
        f"  {'Term':<32} {'ρ':>{col_w}} {'MAE':>{col_w}} {'MSE':>{col_w}} {'E_truth':>{col_w}} {'E_pred':>{col_w}}"
    )
    print(
        f"  {'─' * 32} {'─' * col_w} {'─' * col_w} {'─' * col_w} {'─' * col_w} {'─' * col_w}"
    )
    for i, label in enumerate(OUTPUT_TERM_SHORT_LABELS):
        print(
            f"  {label:<32} "
            f"{metrics['rho'][i]:>{col_w}.4f} "
            f"{metrics['mae'][i]:>{col_w}.4e} "
            f"{metrics['mse'][i]:>{col_w}.4e} "
            f"{metrics['e_truth'][i]:>{col_w}.4e} "
            f"{metrics['e_pred'][i]:>{col_w}.4e}"
        )
    print(f"{'─' * 72}\n")


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------


def plot_scatter_per_term(
    y_true: NDArray,
    y_pred: NDArray,
    metrics: dict[str, NDArray],
    output_path: Path,
    dataset_label: str = "Validation",
    max_scatter_points: int = 50_000,
) -> None:
    """5-panel scatter: prediction vs truth per output term.

    Follows Krochak (2023) Fig. 4.7: red perfect line, yellow trend line,
    E values top-left, ρ bottom-right. Subsampled for large datasets.
    """
    n_samples = y_true.shape[0]
    if n_samples > max_scatter_points:
        idx = np.random.default_rng(0).choice(
            n_samples, size=max_scatter_points, replace=False
        )
        y_true, y_pred = y_true[idx], y_pred[idx]

    fig, axes = plt.subplots(1, N_OUTPUT_TERMS, figsize=(4 * N_OUTPUT_TERMS, 4.5))
    fig.suptitle(
        f"A priori: prediction vs truth — {dataset_label} set", fontsize=13, y=1.01
    )

    for term_idx, ax in enumerate(axes):
        truth_col = y_true[:, term_idx]
        pred_col = y_pred[:, term_idx]

        ax.scatter(
            truth_col, pred_col, s=1, alpha=0.15, color="dimgray", rasterized=True
        )

        axis_lim = max(abs(truth_col).max(), abs(pred_col).max()) * 1.05
        axis_lim = max(axis_lim, 1e-12)
        lim_range = [-axis_lim, axis_lim]
        ax.plot(lim_range, lim_range, color="red", linewidth=1.5, label="Perfect")

        slope = float(np.dot(truth_col, pred_col) / np.dot(truth_col, truth_col))
        trend_x = np.linspace(-axis_lim, axis_lim, 100)
        ax.plot(
            trend_x,
            slope * trend_x,
            color="gold",
            linewidth=1.5,
            linestyle="--",
            label="Trend",
        )

        ax.set_xlim(lim_range)
        ax.set_ylim(lim_range)
        ax.set_xlabel("Truth [−]", fontsize=9)
        ax.set_ylabel("Prediction [−]", fontsize=9)
        ax.set_title(OUTPUT_TERM_LABELS[term_idx], fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.text(
            0.04,
            0.97,
            f"$E_{{truth}}$ = {metrics['e_truth'][term_idx]:.3e}\n$E_{{pred}}$ = {metrics['e_pred'][term_idx]:.3e}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
        )
        ax.text(
            0.97,
            0.04,
            f"$\\rho$ = {metrics['rho'][term_idx]:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            horizontalalignment="right",
            color="navy",
            fontweight="bold",
        )

    plt.tight_layout()
    save_path = output_path / f"apriori_scatter_{dataset_label.lower()}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved scatter plot to '{save_path}'.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Time-series and spatial plots — shared panel builder
# ---------------------------------------------------------------------------


def _plot_term_panels(
    ax_list: list[plt.Axes],
    y_true: NDArray,
    y_pred: NDArray,
    x_axis: NDArray,
    x_label: str,
) -> None:
    """Plot predicted vs exact for each output term on pre-created axes."""
    for term_idx, ax in enumerate(ax_list):
        ax.plot(x_axis, y_true[:, term_idx], color="black", lw=1.5, label="Exact")
        ax.plot(
            x_axis,
            y_pred[:, term_idx],
            color="royalblue",
            lw=1.0,
            linestyle="--",
            label="Predicted",
        )
        ax.set_ylabel(OUTPUT_TERM_LABELS[term_idx], fontsize=8)
        ax.grid(True, alpha=0.2)
        if term_idx == 0:
            ax.legend(fontsize=8)
    ax_list[-1].set_xlabel(x_label, fontsize=10)


def plot_time_series_single_element(
    y_true_seq: NDArray,
    y_pred_seq: NDArray,
    times: NDArray,
    element_idx: int,
    output_path: Path,
    dataset_label: str = "Validation",
) -> None:
    """Predicted vs exact interaction term time series for one element."""
    fig, axes = plt.subplots(
        N_OUTPUT_TERMS, 1, figsize=(10, 3 * N_OUTPUT_TERMS), sharex=True
    )
    fig.suptitle(
        f"A priori time series — element {element_idx} ({dataset_label})", fontsize=12
    )
    _plot_term_panels(list(axes), y_true_seq, y_pred_seq, times, "Time $t$")
    plt.tight_layout()
    save_path = (
        output_path
        / f"apriori_timeseries_elem{element_idx}_{dataset_label.lower()}.png"
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved time-series plot to '{save_path}'.")
    plt.close(fig)


def plot_spatial_profile_single_snapshot(
    y_true_snapshot: NDArray,
    y_pred_snapshot: NDArray,
    element_coords: NDArray,
    snapshot_time: float,
    output_path: Path,
    dataset_label: str = "Validation",
) -> None:
    """Predicted vs exact spatial profile at one snapshot."""
    fig, axes = plt.subplots(
        N_OUTPUT_TERMS, 1, figsize=(10, 3 * N_OUTPUT_TERMS), sharex=True
    )
    fig.suptitle(
        f"A priori spatial profile — t = {snapshot_time:.4f} ({dataset_label})",
        fontsize=12,
    )
    _plot_term_panels(
        list(axes), y_true_snapshot, y_pred_snapshot, element_coords, "$x$"
    )
    plt.tight_layout()
    save_path = (
        output_path
        / f"apriori_spatial_t{snapshot_time:.3f}_{dataset_label.lower()}.png"
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved spatial profile plot to '{save_path}'.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interaction-term energy spectrum
# ---------------------------------------------------------------------------


def plot_interaction_term_spectrum(
    y_true_snapshot: NDArray,
    y_pred_snapshot: NDArray,
    domain_length: float,
    snapshot_time: float,
    output_path: Path,
    dataset_label: str = "Validation",
) -> None:
    """Log-log energy spectrum of the summed interaction terms at one snapshot."""
    interaction_true = y_true_snapshot.sum(axis=1)
    interaction_pred = y_pred_snapshot.sum(axis=1)
    n_elements = len(interaction_true)
    wavenumber = 2.0 * np.pi * np.fft.rfftfreq(n_elements, d=domain_length / n_elements)
    spectrum_true = 0.5 * np.abs(np.fft.rfft(interaction_true)) ** 2 / n_elements
    spectrum_pred = 0.5 * np.abs(np.fft.rfft(interaction_pred)) ** 2 / n_elements

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(wavenumber[1:], spectrum_true[1:], color="black", lw=1.5, label="Exact")
    ax.loglog(
        wavenumber[1:],
        spectrum_pred[1:],
        color="royalblue",
        lw=1.0,
        linestyle="--",
        label="Predicted",
    )
    ax.set_xlabel("Wavenumber $k$", fontsize=11)
    ax.set_ylabel("$E(k)$", fontsize=11)
    ax.set_title(
        f"Interaction term spectrum — t = {snapshot_time:.4f} ({dataset_label})",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    save_path = (
        output_path
        / f"apriori_spectrum_t{snapshot_time:.3f}_{dataset_label.lower()}.png"
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved spectrum plot to '{save_path}'.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_apriori_verification(
    model_predict_fn: Callable[[NDArray], NDArray],
    data_dir: Path,
    output_dir: Path,
    domain_length: float,
    dt: float,
    n_elements: int,
    dataset_label: str = "Validation",
    element_idx_timeseries: int = 4,
) -> dict[str, NDArray]:
    """Run the full a priori verification suite; return per-term metrics dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    prefix = "val" if dataset_label.lower() == "validation" else "test"
    x_normalised = np.load(data_dir / f"X_{prefix}.npy")
    y_true_normalised = np.load(data_dir / f"y_{prefix}.npy")

    norm_stats = np.load(data_dir / "normalisation_stats.npz")
    y_mean = norm_stats["y_mean"]
    y_std = norm_stats["y_std"]

    y_pred_normalised = model_predict_fn(x_normalised)
    y_true_phys = y_true_normalised * y_std + y_mean
    y_pred_phys = y_pred_normalised * y_std + y_mean

    metrics = compute_per_term_metrics(y_true_phys, y_pred_phys)
    print_metrics_table(metrics, dataset_label=dataset_label)

    plot_scatter_per_term(
        y_true=y_true_phys,
        y_pred=y_pred_phys,
        metrics=metrics,
        output_path=output_dir,
        dataset_label=dataset_label,
    )

    sequential_x_path = data_dir / f"X_{prefix}_sequential.npy"
    sequential_y_path = data_dir / f"y_{prefix}_sequential.npy"

    if not (sequential_x_path.exists() and sequential_y_path.exists()):
        print(
            f"Sequential data not found at '{sequential_x_path}'. Skipping time-series, spatial, and spectrum plots."
        )
        return metrics

    y_seq_true_phys = np.load(sequential_y_path) * y_std + y_mean
    y_seq_pred_phys = model_predict_fn(np.load(sequential_x_path)) * y_std + y_mean
    n_seq_samples = y_seq_true_phys.shape[0]

    element_rows = np.arange(element_idx_timeseries, n_seq_samples, n_elements)
    if len(element_rows) > 1:
        plot_time_series_single_element(
            y_true_seq=y_seq_true_phys[element_rows],
            y_pred_seq=y_seq_pred_phys[element_rows],
            times=np.arange(len(element_rows)) * dt,
            element_idx=element_idx_timeseries,
            output_path=output_dir,
            dataset_label=dataset_label,
        )

    snap_rows = np.arange(n_seq_samples - n_elements, n_seq_samples)
    snapshot_time_val = float((n_seq_samples // n_elements) * dt)
    element_coords = np.linspace(0.0, domain_length, n_elements)

    plot_spatial_profile_single_snapshot(
        y_true_snapshot=y_seq_true_phys[snap_rows],
        y_pred_snapshot=y_seq_pred_phys[snap_rows],
        element_coords=element_coords,
        snapshot_time=snapshot_time_val,
        output_path=output_dir,
        dataset_label=dataset_label,
    )
    plot_interaction_term_spectrum(
        y_true_snapshot=y_seq_true_phys[snap_rows],
        y_pred_snapshot=y_seq_pred_phys[snap_rows],
        domain_length=domain_length,
        snapshot_time=snapshot_time_val,
        output_path=output_dir,
        dataset_label=dataset_label,
    )

    return metrics
