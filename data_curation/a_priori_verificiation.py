"""A priori verification of the VMS-ANN predictor model.

Produces the standard suite of diagnostics used across the TU Delft VMS-ANN
lineage (Robijns 2019, Pusuluri 2021, Rajampeta 2022):

    1. Scatter plots       — prediction vs. truth, per output term, with
                             Pearson ρ, perfect line, and trend line.
    2. MAE / MSE table     — per-term and overall, on val and test sets.
    3. Mean deviation      — E_truth vs E_pred per term (catches bias even
                             when ρ is high, as noted by Krochak 2023).
    4. Time-series plots   — predicted vs. exact term value over time for a
                             single representative spatial element.
    5. Spatial profile     — predicted vs. exact term value over all elements
                             at a single snapshot.
    6. Energy-spectrum of  — FFT of the summed interaction terms (predicted
       interaction terms     vs. exact) to check spectral accuracy.

Output terms (Rajampeta / Research Proposal convention):
    [0] cross term      (w_x, ū·u')_e
    [1] Reynolds term   (w_x, u'²/2)_e
    [2] u't left        (w_l, u'_t)_e
    [3] u't right       (w_r, u'_t)_e
    [4] viscous SGS     (w_x, u'_x)_e
"""

from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_TERM_LABELS: list[str] = [
    r"$(w_x,\,\bar{u}u')_e$  [cross]",
    r"$(w_x,\,u'^2/2)_e$  [Reynolds]",
    r"$(w_l,\,u'_t)_e$  [temporal L]",
    r"$(w_r,\,u'_t)_e$  [temporal R]",
    r"$(w_x,\,u'_x)_e$  [viscous]",
]
N_OUTPUT_TERMS = 5

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def pearson_correlation(
    y_true: NDArray,
    y_pred: NDArray,
) -> float:
    """Pearson correlation coefficient between two 1-D arrays."""
    rho, _ = pearsonr(y_true.ravel(), y_pred.ravel())
    return float(rho)


def mean_absolute_error(
    y_true: NDArray,
    y_pred: NDArray,
) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(
    y_true: NDArray,
    y_pred: NDArray,
) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def compute_per_term_metrics(
    y_true: NDArray,
    y_pred: NDArray,
) -> dict[str, NDArray]:
    """Compute ρ, MAE, MSE, E_truth, E_pred for each output term.

    Parameters
    ----------
    y_true, y_pred : NDArray of shape (n_samples, N_OUTPUT_TERMS)

    Returns
    -------
    metrics : dict with keys rho, mae, mse, e_truth, e_pred,
              each an array of shape (N_OUTPUT_TERMS,).
    """
    n_terms = y_true.shape[1]
    rho_vals = np.empty(n_terms)
    mae_vals = np.empty(n_terms)
    mse_vals = np.empty(n_terms)
    e_truth_vals = np.empty(n_terms)
    e_pred_vals = np.empty(n_terms)

    for term_idx in range(n_terms):
        truth_col = y_true[:, term_idx]
        pred_col = y_pred[:, term_idx]
        rho_vals[term_idx] = pearson_correlation(truth_col, pred_col)
        mae_vals[term_idx] = mean_absolute_error(truth_col, pred_col)
        mse_vals[term_idx] = mean_squared_error(truth_col, pred_col)
        e_truth_vals[term_idx] = float(np.mean(truth_col))
        e_pred_vals[term_idx] = float(np.mean(pred_col))

    return {
        "rho": rho_vals,
        "mae": mae_vals,
        "mse": mse_vals,
        "e_truth": e_truth_vals,
        "e_pred": e_pred_vals,
    }


def print_metrics_table(
    metrics: dict[str, NDArray],
    dataset_label: str = "Validation",
) -> None:
    """Print a formatted metrics table to stdout."""
    header = f"\n{'─' * 72}\n  A priori metrics — {dataset_label} set\n{'─' * 72}"
    print(header)
    col_w = 14
    print(
        f"  {'Term':<32} {'ρ':>{col_w}} {'MAE':>{col_w}} "
        f"{'MSE':>{col_w}} {'E_truth':>{col_w}} {'E_pred':>{col_w}}"
    )
    print(
        f"  {'─' * 32} {'─' * col_w} {'─' * col_w} {'─' * col_w} {'─' * col_w} {'─' * col_w}"
    )
    for term_idx, label in enumerate(OUTPUT_TERM_LABELS):
        short_label = label.split("[")[1].rstrip("]").strip()
        print(
            f"  {short_label:<32} "
            f"{metrics['rho'][term_idx]:>{col_w}.4f} "
            f"{metrics['mae'][term_idx]:>{col_w}.4e} "
            f"{metrics['mse'][term_idx]:>{col_w}.4e} "
            f"{metrics['e_truth'][term_idx]:>{col_w}.4e} "
            f"{metrics['e_pred'][term_idx]:>{col_w}.4e}"
        )
    print(f"{'─' * 72}\n")


# ---------------------------------------------------------------------------
# Plot 1: Scatter plots (prediction vs truth), one panel per output term
# ---------------------------------------------------------------------------


def plot_scatter_per_term(
    y_true: NDArray,
    y_pred: NDArray,
    metrics: dict[str, NDArray],
    output_path: Path,
    dataset_label: str = "Validation",
    max_scatter_points: int = 50_000,
) -> None:
    """5-panel scatter: prediction vs truth, with ρ, E_truth, E_pred annotations.

    Follows Krochak (2023) Fig. 4.7 / Bettini (2023) Fig. 4.2 convention:
    red perfect line, yellow trend line (linear regression), E values top-left,
    ρ bottom-right.  Data is randomly subsampled for large datasets.
    """
    n_samples = y_true.shape[0]
    if n_samples > max_scatter_points:
        rng = np.random.default_rng(0)
        subset_indices = rng.choice(n_samples, size=max_scatter_points, replace=False)
        y_true_plot = y_true[subset_indices]
        y_pred_plot = y_pred[subset_indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

    fig, axes = plt.subplots(1, N_OUTPUT_TERMS, figsize=(4 * N_OUTPUT_TERMS, 4.5))
    fig.suptitle(
        f"A priori: prediction vs truth — {dataset_label} set", fontsize=13, y=1.01
    )

    for term_idx, ax in enumerate(axes):
        truth_col = y_true_plot[:, term_idx]
        pred_col = y_pred_plot[:, term_idx]

        ax.scatter(
            truth_col, pred_col, s=1, alpha=0.15, color="dimgray", rasterized=True
        )

        # Perfect line (red)
        axis_lim = max(abs(truth_col).max(), abs(pred_col).max()) * 1.05
        axis_lim = max(axis_lim, 1e-12)
        lim_range = [-axis_lim, axis_lim]
        ax.plot(lim_range, lim_range, color="red", linewidth=1.5, label="Perfect")

        # Trend line (yellow) — linear regression through origin
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

        rho_val = metrics["rho"][term_idx]
        e_truth_val = metrics["e_truth"][term_idx]
        e_pred_val = metrics["e_pred"][term_idx]

        ax.text(
            0.04,
            0.97,
            f"$E_{{truth}}$ = {e_truth_val:.3e}\n$E_{{pred}}$ = {e_pred_val:.3e}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
        )
        ax.text(
            0.97,
            0.04,
            f"$\\rho$ = {rho_val:.4f}",
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
# Plot 2: Time-series of predicted vs exact for a single element
# ---------------------------------------------------------------------------


def plot_time_series_single_element(
    y_true_seq: NDArray,
    y_pred_seq: NDArray,
    times: NDArray,
    element_idx: int,
    output_path: Path,
    dataset_label: str = "Validation",
) -> None:
    """Plot predicted vs exact interaction term time series for one element.

    Parameters
    ----------
    y_true_seq, y_pred_seq : NDArray of shape (T, N_OUTPUT_TERMS)
        Sequential (ordered-in-time) predictions for a single spatial element.
    times : NDArray of shape (T,)
        Corresponding simulation times.
    element_idx : int
        Element index (for the plot title only).
    """
    fig, axes = plt.subplots(
        N_OUTPUT_TERMS, 1, figsize=(10, 3 * N_OUTPUT_TERMS), sharex=True
    )
    fig.suptitle(
        f"A priori time series — element {element_idx} ({dataset_label})", fontsize=12
    )

    for term_idx, ax in enumerate(axes):
        ax.plot(times, y_true_seq[:, term_idx], color="black", lw=1.5, label="Exact")
        ax.plot(
            times,
            y_pred_seq[:, term_idx],
            color="royalblue",
            lw=1.0,
            linestyle="--",
            label="Predicted",
        )
        ax.set_ylabel(OUTPUT_TERM_LABELS[term_idx], fontsize=8)
        ax.grid(True, alpha=0.2)
        if term_idx == 0:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time $t$", fontsize=10)
    plt.tight_layout()

    save_path = (
        output_path
        / f"apriori_timeseries_elem{element_idx}_{dataset_label.lower()}.png"
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved time-series plot to '{save_path}'.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Spatial profile at a single snapshot
# ---------------------------------------------------------------------------


def plot_spatial_profile_single_snapshot(
    y_true_snapshot: NDArray,
    y_pred_snapshot: NDArray,
    element_coords: NDArray,
    snapshot_time: float,
    output_path: Path,
    dataset_label: str = "Validation",
) -> None:
    """Spatial profile of predicted vs exact interaction terms at one snapshot.

    Parameters
    ----------
    y_true_snapshot, y_pred_snapshot : NDArray of shape (N_elements, N_OUTPUT_TERMS)
        Values over all elements at a single time level.
    element_coords : NDArray of shape (N_elements,)
        x-coordinate of each element midpoint.
    snapshot_time : float
        Simulation time of the snapshot (for the title).
    """
    fig, axes = plt.subplots(
        N_OUTPUT_TERMS, 1, figsize=(10, 3 * N_OUTPUT_TERMS), sharex=True
    )
    fig.suptitle(
        f"A priori spatial profile — t = {snapshot_time:.4f} ({dataset_label})",
        fontsize=12,
    )

    for term_idx, ax in enumerate(axes):
        ax.plot(
            element_coords,
            y_true_snapshot[:, term_idx],
            color="black",
            lw=1.5,
            label="Exact",
        )
        ax.plot(
            element_coords,
            y_pred_snapshot[:, term_idx],
            color="royalblue",
            lw=1.0,
            linestyle="--",
            label="Predicted",
        )
        ax.set_ylabel(OUTPUT_TERM_LABELS[term_idx], fontsize=8)
        ax.grid(True, alpha=0.2)
        if term_idx == 0:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel("$x$", fontsize=10)
    plt.tight_layout()

    save_path = (
        output_path
        / f"apriori_spatial_t{snapshot_time:.3f}_{dataset_label.lower()}.png"
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved spatial profile plot to '{save_path}'.")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Energy spectrum of summed interaction terms
# ---------------------------------------------------------------------------


def plot_interaction_term_spectrum(
    y_true_snapshot: NDArray,
    y_pred_snapshot: NDArray,
    domain_length: float,
    snapshot_time: float,
    output_path: Path,
    dataset_label: str = "Validation",
) -> None:
    """Log-log energy spectrum of the summed interaction terms.

    Sums all 5 output terms per element into one net interaction signal,
    then FFTs to show spectral accuracy.

    Parameters
    ----------
    y_true_snapshot, y_pred_snapshot : NDArray of shape (N_elements, N_OUTPUT_TERMS)
    domain_length : float
    snapshot_time : float
    """
    interaction_true = y_true_snapshot.sum(axis=1)
    interaction_pred = y_pred_snapshot.sum(axis=1)

    n_elements = len(interaction_true)
    freq = np.fft.rfftfreq(n_elements, d=domain_length / n_elements)
    wavenumber = 2.0 * np.pi * freq

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
        f"Interaction term energy spectrum — t = {snapshot_time:.4f} ({dataset_label})",
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
    snapshot_idx_spatial: int = -1,
) -> dict[str, NDArray]:
    """Run the full a priori verification suite.

    Parameters
    ----------
    model_predict_fn:
        Callable that takes a normalised X array (n_samples, 20) and returns
        a normalised y_pred array (n_samples, 5).  Wrap your model's forward
        pass here.
    data_dir:
        Directory containing X_val.npy / y_val.npy (or X_test / y_test).
        Also expects normalisation_stats.npz for de-normalisation.
    output_dir:
        Where to save all plots.
    domain_length:
        Physical domain length L (for spectrum wavenumbers).
    dt:
        LES time step (for reconstructing time axis on time-series plots).
    dataset_label:
        "Validation" or "Test" — used in titles and file names.
    element_idx_timeseries:
        Which spatial element to use for the time-series plot.
    snapshot_idx_spatial:
        Which time snapshot to use for spatial profile / spectrum (-1 = last).

    Returns
    -------
    metrics : dict with keys rho, mae, mse, e_truth, e_pred per output term.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    prefix = "val" if dataset_label.lower() == "validation" else "test"
    x_normalised = np.load(data_dir / f"X_{prefix}.npy")
    y_true_normalised = np.load(data_dir / f"y_{prefix}.npy")

    norm_stats = np.load(data_dir / "normalisation_stats.npz")
    y_mean = norm_stats["y_mean"]
    y_std = norm_stats["y_std"]

    # --- Predict ---
    y_pred_normalised = model_predict_fn(x_normalised)

    # --- De-normalise for physical-space metrics ---
    y_true_phys = y_true_normalised * y_std + y_mean
    y_pred_phys = y_pred_normalised * y_std + y_mean

    # --- Metrics ---
    metrics = compute_per_term_metrics(y_true_phys, y_pred_phys)
    print_metrics_table(metrics, dataset_label=dataset_label)

    # --- Scatter plots ---
    plot_scatter_per_term(
        y_true=y_true_phys,
        y_pred=y_pred_phys,
        metrics=metrics,
        output_path=output_dir,
        dataset_label=dataset_label,
    )

    # --- Time-series for a single element ---
    # Re-order rows back to time×space order for sequential plots.
    # This requires the original (unshuffled) sequential data; if only the
    # shuffled val/test set is available, skip with a warning.
    # --- Time-series, spatial profile, and spectrum ---
    # Requires unshuffled sequential data saved by split_and_save.
    sequential_path = data_dir / f"X_{prefix}_sequential.npy"
    sequential_y_path = data_dir / f"y_{prefix}_sequential.npy"

    if not (sequential_path.exists() and sequential_y_path.exists()):
        print(
            f"Sequential data not found at '{sequential_path}'. "
            "Skipping time-series, spatial profile, and spectrum plots. "
            "Add sequential saves to split_and_save to enable these."
        )
        return metrics

    x_seq = np.load(sequential_path)
    y_seq_true_norm = np.load(sequential_y_path)
    y_seq_pred_norm = model_predict_fn(x_seq)

    y_seq_true_phys = y_seq_true_norm * y_std + y_mean
    y_seq_pred_phys = y_seq_pred_norm * y_std + y_mean

    n_seq_samples = y_seq_true_phys.shape[0]

    # Rows are ordered (t0,e0),(t0,e1),...,(t0,e_{N-1}),(t1,e0),...
    # Row k → time step k // n_elements, element k % n_elements.

    # Time-series: every n_elements-th row from offset element_idx_timeseries
    element_rows = np.arange(element_idx_timeseries, n_seq_samples, n_elements)
    if len(element_rows) > 1:
        times_seq = np.arange(len(element_rows)) * dt
        plot_time_series_single_element(
            y_true_seq=y_seq_true_phys[element_rows],
            y_pred_seq=y_seq_pred_phys[element_rows],
            times=times_seq,
            element_idx=element_idx_timeseries,
            output_path=output_dir,
            dataset_label=dataset_label,
        )

    # Spatial profile + spectrum: last n_elements rows = final snapshot
    snap_rows = np.arange(n_seq_samples - n_elements, n_seq_samples)
    element_coords = np.linspace(0.0, domain_length, n_elements)
    snapshot_time_val = float((n_seq_samples // n_elements) * dt)

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
