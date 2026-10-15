"""
Takes dataset sampled over t_start and t_final (308.0101 - 315.00) and creates interval windows.
Computes mean and <w'w'> data from full set to slice.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from setup.constants import T_ZERO


def interval_mean(
    cumulative: NDArray, counts: NDArray, window_index: int, width: int = 1
) -> NDArray:
    """Mean profile over window (k-width -> k) from cumulative-mean snapshots.

    Args:
        cumulative: (n_snapshots, n_points) running means since T_START.
        counts: (n_snapshots,) number of samples in each snapshot.
        window_index: k. Ending window index.
        width: Number of interval steps the window spans (1 for 1-second, 2 for 2-second).
    """
    start_index = window_index - width
    if start_index < 0:
        return cumulative[window_index].copy()

    n_end = counts[window_index]
    n_start = counts[start_index]
    if n_end <= n_start:
        raise ValueError(
            f"Sample count must increase: {n_start} -> {n_end} at window {window_index}"
        )

    weighted_diff = n_end * cumulative[window_index] - n_start * cumulative[start_index]
    return weighted_diff / (n_end - n_start)


def interval_variance(
    cumulative_mean: NDArray,
    cumulative_variance: NDArray,
    counts: NDArray,
    window_index: int,
    width: int = 1,
) -> NDArray:
    """Variance (e.g. <w'w'>) over window (k-width -> k)."""
    raw_second_moment = cumulative_variance + cumulative_mean**2
    interval_second_moment = interval_mean(
        raw_second_moment, counts, window_index, width=width
    )
    window_mean = interval_mean(cumulative_mean, counts, window_index, width=width)
    return interval_second_moment - window_mean**2


def save_interval_profile(
    profile: NDArray,
    name: str,
    times: NDArray,
    window_index: int,
    save_dir: Path,
    t_start: float,
    width: int = 1,
) -> Path:
    """Save a window profile as <name>_<start>_<end>.npy supporting custom widths."""
    start_index = window_index - width
    window_start = t_start if start_index < 0 else times[start_index]
    window_end = times[window_index]

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{name}_{window_start}_{window_end}.npy"
    np.save(save_path, profile)
    return save_path


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    STAT_DIR = PROJECT_ROOT / "dns_data" / "curated" / "stat"

    save_dir = PROJECT_ROOT / "dns_data" / "intervals"

    w_mean_path = STAT_DIR / "mean_w.npy"
    ww_path = STAT_DIR / "mean_reynolds_w.npy"
    times_path = STAT_DIR / "times.npy"

    w_cum = np.load(w_mean_path)
    ww_cum = np.load(ww_path)
    times = np.load(times_path)

    counts = np.array([9995, 19996, 10001, 20002, 30003, 40004, 49904])

    windows = np.array([window for window in range(len(times))])
    mask = windows != 2
    windows_1s = windows[mask]

    T_START = T_ZERO

    # 1. Compute and save 1-second interval profiles (width = 1)
    for k in windows_1s:
        w_mean = interval_mean(w_cum, counts, k, width=1)
        w_var = interval_variance(w_cum, ww_cum, counts, k, width=1)

        save_interval_profile(w_mean, "w_mean", times, k, save_dir, T_START, width=1)
        save_interval_profile(w_var, "ww", times, k, save_dir, T_START, width=1)

    # 2. Compute and save 2-second interval profiles (width = 2)
    # A 2-second window ending at index k spans from k-2 to k.
    # It is invalid only if it uses the bad 1-second sub-interval ending at index 2 (i.e., step from 1 to 2).
    for k in range(2, len(times)):
        # Check if either constituent 1-second step is the bad one (ending at index 2)
        if k == 2 or k - 1 == 2:
            continue

        try:
            w_mean = interval_mean(w_cum, counts, k, width=2)
            w_var = interval_variance(w_cum, ww_cum, counts, k, width=2)

            save_interval_profile(
                w_mean, "w_mean", times, k, save_dir, T_START, width=2
            )
            save_interval_profile(w_var, "ww", times, k, save_dir, T_START, width=2)
        except ValueError:
            pass
