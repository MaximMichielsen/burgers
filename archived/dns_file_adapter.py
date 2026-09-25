import re
from pathlib import Path
import numpy as np


class DNSDataReader:
    """Parses raw DNS .dat and .crd files and provides CSV export utilities for plotting."""

    def __init__(self, crd_path: str | Path, dat_path: str | Path):
        self.crd_path = Path(crd_path)
        self.dat_path = Path(dat_path)

        if not self.crd_path.exists():
            raise FileNotFoundError(
                f"Missing coordinate file: {self.crd_path.resolve()}"
            )
        if not self.dat_path.exists():
            raise FileNotFoundError(f"Missing data file: {self.dat_path.resolve()}")

        # 1. Load spatial grid x
        self.x_mesh = np.loadtxt(self.crd_path)
        self.n_nodes = len(self.x_mesh)

        # 2. Extract snapshots
        self.times, self.w_solutions, self.f_forcing = self._parse_dat_file()

    def _parse_dat_file(self):
        times, w_snaps, f_snaps = [], [], []
        curr_w, curr_f = [], []
        time_regex = re.compile(r"time=([0-9.eE+-]+)")

        with open(self.dat_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    match = time_regex.search(line)
                    if match:
                        if len(curr_w) == self.n_nodes:
                            w_snaps.append(curr_w)
                            f_snaps.append(curr_f)
                            curr_w, curr_f = [], []
                        times.append(float(match.group(1)))
                    continue

                parts = line.strip().split()
                if len(parts) == 2:
                    curr_w.append(float(parts[0]))  # Velocity w
                    curr_f.append(float(parts[1]))  # Forcing f

            if len(curr_w) == self.n_nodes:
                w_snaps.append(curr_w)
                f_snaps.append(curr_f)

        return (
            np.array(times, dtype=np.float32),
            np.array(w_snaps, dtype=np.float32),
            np.array(f_snaps, dtype=np.float32),
        )

    def export_to_csv_directory(
        self, target_dir: str | Path, stride: int = 100
    ) -> Path:
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        t_start = self.times[0]

        for idx in range(0, len(self.times), stride):
            t_normalized = self.times[idx] - t_start
            w_val = self.w_solutions[idx]

            csv_file = target_path / f"sol_t{t_normalized:.6f}.csv"
            if not csv_file.exists():
                indices = np.arange(self.n_nodes)
                data = np.column_stack([indices, self.x_mesh, w_val])
                header = "node_idx,x,u"
                np.savetxt(csv_file, data, delimiter=",", header=header, comments="")

        return target_path

    def get_final_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (x_mesh, final_velocity_u) directly for in-memory plotting."""
        return self.x_mesh, self.w_solutions[-1]
