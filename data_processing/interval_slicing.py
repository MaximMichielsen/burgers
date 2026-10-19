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

    # Check for resets / sample drops strictly inside the window span
    for i in range(max(1, start_index + 1), window_index + 1):
        if counts[i] < counts[i - 1]:
            if width > 1:
                raise ValueError(
                    f"Multi-step window [{start_index} -> {window_index}] spans across reset at index {i}. Skipping."
                )

    n_end = counts[window_index]
    n_start = counts[start_index]

    if n_end <= n_start:
        return cumulative[window_index]

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
    """Save a window profile using standard Python float string conversion to match ReferenceTrajectory."""
    start_index = window_index - width

    window_start = t_start if start_index < 0 else float(times[start_index])
    window_end = float(times[window_index])

    # Use standard str() conversion so it matches ReferenceTrajectory's naming (e.g., 315.0 instead of 315.00)
    start_str = str(t_start) if start_index < 0 else str(window_start)
    end_str = str(window_end)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{name}_{start_str}_{end_str}.npy"
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
    T_START = T_ZERO

    # 1. Compute and save 1-second interval profiles (width = 1)
    for k in range(len(times)):
        try:
            w_mean = interval_mean(w_cum, counts, k, width=1)
            w_var = interval_variance(w_cum, ww_cum, counts, k, width=1)

            save_interval_profile(
                w_mean, "w_mean", times, k, save_dir, T_START, width=1
            )
            save_interval_profile(w_var, "ww", times, k, save_dir, T_START, width=1)
        except ValueError as e:
            print(f"Skipped 1s window {k}: {e}")

    # 2. Compute and save 2-second interval profiles (width = 2)
    # Start at k = 1 so it captures the 308-310 window (start_index = -1 -> T_ZERO)
    for k in range(1, len(times)):
        try:
            w_mean = interval_mean(w_cum, counts, k, width=2)
            w_var = interval_variance(w_cum, ww_cum, counts, k, width=2)

            save_interval_profile(
                w_mean, "w_mean", times, k, save_dir, T_START, width=2
            )
            save_interval_profile(w_var, "ww", times, k, save_dir, T_START, width=2)
        except ValueError as e:
            print(f"Skipped 2s window {k}: {e}")
