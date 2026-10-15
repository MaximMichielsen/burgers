import hashlib
import json
import os
from pathlib import Path

import numpy as np


def generate_temporal_cache_key(
    dns_file_path: Path, dt_les: float, n_modes: int, dt_dns: float = 1e-4
):
    """Generate a unique hash for the preprocessing configuration."""
    # Get file modification time & size for fast verification
    file_stat = os.stat(dns_file_path)

    config = {
        "file_size": file_stat.st_size,
        "mtime": file_stat.st_mtime,
        "dt_dns": dt_dns,
        "dt_les": dt_les,
        "n_modes": n_modes,
    }

    config_string = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_string.encode("utf-8")).hexdigest()


def get_coarsened_forcing(
    dns_forcing_file_path: Path,
    save_dir: Path,
    dt_dns: float = 1e-4,
    dt_les: float = 1e-2,
    n_modes: int = 20,
):
    cache_key = generate_temporal_cache_key(
        dns_forcing_file_path, dt_les, n_modes, dt_dns
    )
    cache_file = save_dir / f"cache_forcing_{cache_key}.npy"
    times_cache_file = save_dir / f"cache_times_{cache_key}.npy"

    if cache_file.exists() and times_cache_file.exists():
        print(f"[Cache HIT] Loading downsampled data and times from {save_dir}")
        forcing_les = np.load(cache_file)
        times = np.load(times_cache_file)
        return forcing_les, times

    print("[Cache MISS] Computing POD and coarsening time series...")

    # 1. Load raw DNS data
    forcing_dns = np.load(dns_forcing_file_path)  # Shape: (T_dns, Spatial_Nodes)
    T_dns = forcing_dns.shape[0]

    # 2. Perform Temporal POD / SVD
    # X = U @ S @ Vt
    temporal_coeffs, S, spatial_modes = np.linalg.svd(forcing_dns, full_matrices=False)

    # 3. Truncate to N modes (capturing dominant energy)
    U_trunc = temporal_coeffs[:, :n_modes]
    S_trunc = S[:n_modes]
    Vt_trunc = spatial_modes[:n_modes, :]

    # 4. Resample temporal modes U_trunc from dt_dns to dt_les
    stride = int(dt_les / dt_dns)  # e.g., 1e-2 / 1e-4 = 100

    # Low-pass boxcar filter before subsampling to avoid aliasing
    kernel = np.ones(stride) / stride
    U_filtered = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=U_trunc
    )
    U_subsampled = U_filtered[::stride, :]

    # Generate downsampled time indexing array aligned with DNS time axis
    times = np.arange(0, T_dns, stride) * dt_dns

    # 5. Reconstruct coarsened forcing time-series at dt_les
    forcing_les = U_subsampled @ np.diag(S_trunc) @ Vt_trunc

    # 6. Save to cache
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, forcing_les)
    np.save(times_cache_file, times)
    print(f"[Cache SAVED] Saved processed forcing and times to {save_dir}")

    return forcing_les, times


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.resolve()

    forcing_dns_path = ROOT_DIR / "dns_data" / "curated" / "dat" / "f.npy"
    save_dir = ROOT_DIR / "dns_data" / "temporal" / "unprojected"

    dt_les = 1e-3
    n_modes = 20

    generate_temporal_cache_key(
        dns_file_path=forcing_dns_path, dt_les=dt_les, n_modes=n_modes
    )
    forcing_les, times = get_coarsened_forcing(
        dns_forcing_file_path=forcing_dns_path,
        save_dir=save_dir,
        dt_les=dt_les,
        n_modes=n_modes,
    )
    print(f"Forcing shape: {forcing_les.shape}, Times shape: {times.shape}")
