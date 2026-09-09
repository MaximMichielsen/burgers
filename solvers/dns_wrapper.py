import re
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator


class DNSDataForcing:
    def __init__(self, crd_filepath: str, dat_filepath: str):
        crd_path, dat_path = Path(crd_filepath), Path(dat_filepath)

        self.z_arr = np.loadtxt(crd_path, dtype=np.float32)
        n_points = len(self.z_arr)

        times, forcing_snapshots, current_f = [], [], []
        time_pattern = re.compile(r"time=([0-9.eE+-]+)")

        with open(dat_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    match = time_pattern.search(line)
                    if match:
                        if len(current_f) == n_points:
                            forcing_snapshots.append(current_f)
                            current_f = []
                        times.append(float(match.group(1)))
                    continue

                parts = line.strip().split()
                if len(parts) == 2:
                    current_f.append(float(parts[1]))

            if len(current_f) == n_points:
                forcing_snapshots.append(current_f)

        self.t_start = times[0]
        # Shift time to start at 0.0 relative to simulation start
        self.t_arr = np.array(times, dtype=np.float32) - self.t_start
        self.F_data = np.array(forcing_snapshots, dtype=np.float32)

        self.interpolator = RegularGridInterpolator(
            (self.t_arr, self.z_arr), self.F_data, bounds_error=False, fill_value=None
        )

    @property
    def __name__(self):
        return self.__class__.__name__

    def __call__(self, x_mesh: np.ndarray, t: float) -> np.ndarray:
        t_clamped = np.clip(t, self.t_arr[0], self.t_arr[-1])
        t_vec = np.full_like(x_mesh, t_clamped, dtype=np.float32)
        points = np.column_stack([t_vec, x_mesh])
        return self.interpolator(points)
