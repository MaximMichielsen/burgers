"""A priori verification suite for the VMS-ANN SGS predictor.

Diagnostics (Robijns 2019, Pusuluri 2021, Rajampeta 2022, Krochak 2023):
    1. Scatter plots       — prediction vs truth per output term (ρ, E_truth, E_pred)
    2. Metrics table       — ρ, MAE, MSE, E_truth, E_pred per term
    3. Spatial profiles    — predicted vs exact over all elements at one snapshot

Output terms (Rajampeta / Research Proposal convention):
    [0] (w_x, ū·u')_e       cross
    [1] (w_x, u'²/2)_e      Reynolds
    [2] (w_l, u'_t)_e       temporal left
    [3] (w_r, u'_t)_e       temporal right
    [4] (w_x, u'_x)_e       viscous SGS
"""

from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.stats import pearsonr

from constants import OUTPUT_UNITS

from solvers.sgsp_training_data_generator_stash import load_normalisation_stats_csv

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
# Inference helper
# ---------------------------------------------------------------------------


def predict(model: torch.nn.Module, x_normalised: NDArray) -> NDArray:
    """Run normalised numpy input through the model; return normalised numpy output."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_normalised, dtype=torch.float32)
        return model(x_tensor).numpy()


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
        [
            float(pearsonr(y_true[:, term_idx], y_pred[:, term_idx])[0])
            for term_idx in range(y_true.shape[1])
        ]
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
        f"  {'Term':<32} {'ρ':>{col_w}} {'MAE':>{col_w}} {'MSE':>{col_w}} "
        f"{'E_truth':>{col_w}} {'E_pred':>{col_w}}"
    )
    print(
        f"  {'─' * 32} {'─' * col_w} {'─' * col_w} {'─' * col_w} "
        f"{'─' * col_w} {'─' * col_w}"
    )
    for term_idx, label in enumerate(OUTPUT_TERM_SHORT_LABELS):
        print(
            f"  {label:<32} "
            f"{metrics['rho'][term_idx]:>{col_w}.4f} "
            f"{metrics['mae'][term_idx]:>{col_w}.4e} "
            f"{metrics['mse'][term_idx]:>{col_w}.4e} "
            f"{metrics['e_truth'][term_idx]:>{col_w}.4e} "
            f"{metrics['e_pred'][term_idx]:>{col_w}.4e}"
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
    """5-panel scatter: prediction vs truth per output term."""
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
        axis_lim = (
            max(float(np.abs(truth_col).max()), float(np.abs(pred_col).max())) * 1.05
        )
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
            f"$E_{{truth}}$ = {metrics['e_truth'][term_idx]:.3e}\n"
            f"$E_{{pred}}$ = {metrics['e_pred'][term_idx]:.3e}",
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
# Spatial profile
# ---------------------------------------------------------------------------


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
# Main verification runner
# ---------------------------------------------------------------------------


def run_apriori_verification(
    model: torch.nn.Module,
    data_dir: Path,
    output_dir: Path,
    domain_length: float,
    n_elements: int,
    dataset_label: str = "Validation",
) -> dict[str, NDArray]:
    """Run the full a priori verification suite; return per-term metrics dict.

    Loads X_val.csv / y_val.csv, denormalizes, computes metrics, saves plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    x_normalised = np.loadtxt(data_dir / "X_val.csv", delimiter=",", skiprows=1)
    y_true_normalised = np.loadtxt(data_dir / "y_val.csv", delimiter=",", skiprows=1)

    norm_stats = load_normalisation_stats_csv(data_dir)
    y_mean = norm_stats["y_mean"]
    y_std = norm_stats["y_std"]

    y_pred_normalised = predict(model, x_normalised)
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

    # spatial profile at the last n_elements rows (last snapshot)
    if len(y_true_phys) >= n_elements:
        element_coords = np.linspace(0.0, domain_length, n_elements)
        plot_spatial_profile_single_snapshot(
            y_true_snapshot=y_true_phys[-n_elements:],
            y_pred_snapshot=y_pred_phys[-n_elements:],
            element_coords=element_coords,
            snapshot_time=float(len(y_true_phys) / n_elements),
            output_path=output_dir,
            dataset_label=dataset_label,
        )

    return metrics
