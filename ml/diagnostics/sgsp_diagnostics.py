"""Diagnostic utilities for investigating BurgersSGSP correction magnitudes.

Two entry points:

diagnose_training_label_scale
    Loads the saved training data, denormalises the labels back to physical
    space, and prints per-column statistics.  This immediately reveals
    whether the labels are genuinely small (Cause A: weak SGS forcing for
    this problem) or whether the normalisation stats were accidentally
    computed on a different run (Cause B: stats mismatch).

diagnose_sgsp_predictions
    Runs a fresh BurgersSGSP solver for n_steps and at each step captures
    the raw ANN correction array before it is scattered into the global
    residual.  Prints per-column mean absolute values and the global
    Frobenius norm so you can see how the correction evolves and whether
    any columns dominate or are consistently zeroed.

Place this file at:  ml/diagnostics/sgsp_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_COL_NAMES: list[str] = ["cross", "reynolds", "temporal_L", "temporal_R", "viscous"]


# ---------------------------------------------------------------------------
# Training label scale
# ---------------------------------------------------------------------------


def diagnose_training_label_scale(training_data_path: Path) -> None:
    """Print physical-space statistics of the SGS training labels.

    Compares y_mean / y_std from the normalisation stats against the
    actual denormalised label distribution so you can tell whether the
    corrections are intrinsically small for this problem or whether the
    stats file is mismatched.

    Parameters
    ----------
    training_data_path:
        Directory containing X_train.npy, y_train.npy, and
        normalisation_stats.npz.
    """
    training_data_path = Path(training_data_path)

    norm_stats = np.load(training_data_path / "normalisation_stats.npz")
    y_mean_val: NDArray = norm_stats["y_mean"]
    y_std_val: NDArray = norm_stats["y_std"]
    x_mean_val: NDArray = norm_stats["X_mean"]
    x_std_val: NDArray = norm_stats["X_std"]

    y_train_norm: NDArray = np.load(training_data_path / "y_train.npy")
    x_train_norm: NDArray = np.load(training_data_path / "X_train.npy")

    y_train_physical = y_train_norm * y_std_val + y_mean_val
    x_train_physical = x_train_norm * x_std_val + x_mean_val

    n_samples = y_train_physical.shape[0]
    print(f"  Training samples : {n_samples}")
    print(
        f"  Input range      : [{x_train_physical.min():.3e}, {x_train_physical.max():.3e}]"
    )
    print()
    print(f"  {'Column':<14} {'mean':>12} {'std':>12} {'max|val|':>12}  {'y_std':>12}")
    print("  " + "-" * 66)

    for col_idx, col_name in enumerate(_COL_NAMES):
        col_data = y_train_physical[:, col_idx]
        print(
            f"  {col_name:<14} "
            f"{col_data.mean():>12.3e} "
            f"{col_data.std():>12.3e} "
            f"{np.abs(col_data).max():>12.3e}  "
            f"{y_std_val[col_idx]:>12.3e}"
        )

    print()
    print("  Interpretation:")
    galerkin_scale_estimate = float(x_train_physical.std())
    correction_scale_estimate = float(np.abs(y_train_physical).mean())
    ratio = correction_scale_estimate / (galerkin_scale_estimate + 1e-30)
    print(f"    Galerkin input scale (std)    : {galerkin_scale_estimate:.3e}")
    print(f"    Correction output scale (mean): {correction_scale_estimate:.3e}")
    print(f"    Ratio correction/input        : {ratio:.3e}")
    if ratio < 1e-2:
        print(
            "    *** Labels are O(100x) smaller than inputs — "
            "SGS forcing is intrinsically weak for this problem. ***"
        )
    else:
        print(
            "    Ratio looks reasonable; check for stats file mismatch if online corrections are still small."
        )


# ---------------------------------------------------------------------------
# Online correction magnitudes
# ---------------------------------------------------------------------------


def diagnose_sgsp_predictions(
    solver: "BurgersSGSP",  # noqa: F821  (avoid circular import at module level)
    n_steps: int = 10,
) -> None:
    """Advance solver n_steps and print per-step ANN correction statistics.

    Captures the raw correction array from _compute_ann_contribution at
    each step — before clipping, before scatter into the global residual —
    so the output is not affected by clip_pusuluri / clip_rajampeta flags.

    Stops gracefully if a blow-up is detected mid-run, printing all steps
    collected up to that point.

    Parameters
    ----------
    solver:
        A freshly constructed BurgersSGSP instance (not yet run).
    n_steps:
        Number of time steps to advance and inspect.
    """
    print(
        f"  {'Step':>4}  {'||corr||_F':>12}  "
        + "  ".join(f"{name:>12}" for name in _COL_NAMES)
    )
    print("  " + "-" * (18 + 14 * len(_COL_NAMES)))

    for step_idx in range(n_steps):
        try:
            solver.advance_time_step()
        except RuntimeError as exc:
            print(f"  {step_idx:>4}  (blow-up: {exc})")
            break

        ann_correction_array = solver.compute_sgsp_contribution()

        if ann_correction_array is None:
            print(f"  {step_idx:>4}  {'(warmup)':>12}")
            continue

        global_frob_norm = float(np.linalg.norm(ann_correction_array))
        col_mean_abs_values = np.abs(ann_correction_array).mean(axis=0)

        print(
            f"  {step_idx:>4}  {global_frob_norm:>12.3e}  "
            + "  ".join(f"{val:>12.3e}" for val in col_mean_abs_values)
        )

    print()
    print("  Normalisation stats from loaded model:")
    print(f"    y_mean : {solver.y_mean}")
    print(f"    y_std  : {solver.y_std}")
    print()
    print("  Solver flags:")
    print(f"    clip_pusuluri  : {solver.clip_pusuluri}")
    print(f"    clip_rajampeta : {solver.clip_rajampeta}")
    print(f"    exclude_visc   : {solver.exclude_visc}")
    print(f"    n_nodes        : {solver.n_nodes}")
    print(f"    viscosity      : {solver.viscosity}")
    print(f"    dt             : {solver.dt}")
