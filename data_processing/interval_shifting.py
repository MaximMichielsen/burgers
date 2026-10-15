from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def interval_mean(cumulative: NDArray, counts: NDArray, window_index: int) -> NDArray:
    """Mean profile over window (k-1 -> k) from cumulative-mean snapshots.

    Args:
        cumulative: (n_snapshots, n_points) running means since T_START.
        counts: (n_snapshots,) number of samples in each snapshot.
        window_index: k. Window 0 is T_START -> times[0].
    """
    if window_index == 0:
        return cumulative[0].copy()

    n_end = counts[window_index]
    n_start = counts[window_index - 1]
    if n_end <= n_start:
        raise ValueError(
            f"Sample count must increase: {n_start} -> {n_end} at window {window_index}"
        )

    weighted_diff = (
        n_end * cumulative[window_index] - n_start * cumulative[window_index - 1]
    )
    return weighted_diff / (n_end - n_start)


def interval_variance(
    cumulative_mean: NDArray,
    cumulative_variance: NDArray,
    counts: NDArray,
    window_index: int,
) -> NDArray:
    """Variance (e.g. <w'w'>) over window (k-1 -> k).

    Converts to raw second moments, reuses interval_mean on them, then
    subtracts the interval mean squared.
    """
    raw_second_moment = cumulative_variance + cumulative_mean**2
    interval_second_moment = interval_mean(raw_second_moment, counts, window_index)
    window_mean = interval_mean(cumulative_mean, counts, window_index)
    return interval_second_moment - window_mean**2


def save_interval_profile(
    profile: NDArray,
    name: str,
    times: NDArray,
    window_index: int,
    save_dir: Path,
    t_start: float,
) -> Path:
    """Save a window profile as <name>_<start>_<end>.npy."""
    window_start = t_start if window_index == 0 else times[window_index - 1]
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
    windows = windows[mask]

    T_START = 308.0101

    for k in windows:
        w_mean = interval_mean(w_cum, counts, k)
        w_var = interval_variance(w_cum, ww_cum, counts, k)

        save_interval_profile(w_mean, "w_mean", times, k, save_dir, T_START)
        save_interval_profile(w_var, "ww", times, k, save_dir, T_START)
