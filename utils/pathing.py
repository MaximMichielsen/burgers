from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def get_forcing_path(n_nodes_les: int, is_filtered: bool) -> Path:
    """Returns the correct forcing file path."""
    sub_dir = "filtered" if is_filtered else "unfiltered"
    path = (
        PROJECT_ROOT
        / "dns_data"
        / "projected_forcing"
        / sub_dir
        / f"forcing_l2_{n_nodes_les}.npy"
    )

    if not path.exists():
        raise FileNotFoundError(f"Did not find forcing at {path}")

    return path


def get_velocity_profiles_path(n_nodes_les: int) -> Path:
    """Returns the correct velocity file path."""
    path = (
        PROJECT_ROOT / "dns_data" / "projected_velocity" / f"w_linear_{n_nodes_les}.npy"
    )

    if not path.exists():
        raise FileNotFoundError(f"Velocity file not found at: {path}")

    return path
