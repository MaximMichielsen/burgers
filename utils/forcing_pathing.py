from pathlib import Path
import numpy as np


def get_forcing_path(project_root: Path, n_nodes_les: int, dt: float) -> Path:
    """Returns the correct forcing file path based on dt.

    - dt == 1e-4 uses the base projected directory.
    - dt == 1e-3 (or larger) uses the temporally filtered/projected directory.
    """
    dns_data_dir = project_root / "dns_data"
    if np.isclose(dt, 1e-4):
        forcing_dir = dns_data_dir / "projected"
    else:
        # Uses dns_data/temporal/projected for temporally filtered forcing
        forcing_dir = dns_data_dir / "temporal" / "projected"

    forcing_path = forcing_dir / f"forcing_l2_{n_nodes_les}.npy"

    if not forcing_path.exists():
        raise FileNotFoundError(f"Forcing file not found at: {forcing_path}")

    print(f"Forcing retrieved from: {forcing_path}")
    return forcing_path
