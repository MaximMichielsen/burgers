import numpy as np
from numpy.typing import NDArray


def compute_error_metrics(
    field: NDArray, target: NDArray
) -> tuple[float, float, float]:
    """Compute L2 error and relative error between field and target vectors."""
    if field.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: field {field.shape} vs target {target.shape}"
        )
    l2_error = float(np.linalg.norm(field - target))
    rms_error = float(np.sqrt(np.mean((field - target) ** 2)))
    target_norm = float(np.linalg.norm(target))
    relative_error = (l2_error / (target_norm + 1e-12)) * 100.0
    return l2_error, rms_error, relative_error
