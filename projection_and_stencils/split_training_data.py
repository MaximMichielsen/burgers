"""Split training stencils to three blocks."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def split_data(
    x_input: NDArray, y_target: NDArray, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> tuple[
    tuple[NDArray, NDArray],
    tuple[NDArray, NDArray],
    tuple[NDArray, NDArray],
]:
    """Split training data contiguously into training, validation, and testing blocks."""
    total_samples = x_input.shape[0]

    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    X_train, y_train = x_input[:train_end], y_target[:train_end]
    X_val, y_val = x_input[:val_end], y_target[:val_end]
    X_test, y_test = x_input[val_end:], y_target[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def split_data_shuffled(
    x_input: NDArray, y_target: NDArray, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> tuple[
    tuple[NDArray, NDArray],
    tuple[NDArray, NDArray],
    tuple[NDArray, NDArray],
]:
    """Split training data shuffled into training, validation, and testing blocks."""
    total_samples = x_input.shape[0]

    # Create a random permutation of indices
    indices = np.random.permutation(total_samples)
    x_shuffled = x_input[indices]
    y_shuffled = y_target[indices]

    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    X_train, y_train = x_shuffled[:train_end], y_shuffled[:train_end]
    X_val, y_val = x_shuffled[train_end:val_end], y_shuffled[train_end:val_end]
    X_test, y_test = x_shuffled[val_end:], y_shuffled[val_end:]

    print(f"X Max: {X_train.max()}, X Min: {X_train.min()}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_splits(output_dir: Path | str, splits: tuple) -> None:
    """Saves the split blocks to disk for the ANN training pipeline."""
    path = Path(output_dir)
    names = ["train", "val", "test"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, (X_block, y_block) in zip(names, splits):
        np.save(path / f"X_{name}.npy", X_block)
        np.save(path / f"y_{name}.npy", y_block)

    print(f"Data blocks saved to {path}")


def verify_splits(split_path: Path) -> None:
    """Check the statistics of the split data."""
    print(f"--- Verification for {split_path.name} ---")
    for name in ["train", "val", "test"]:
        X = np.load(split_path / f"X_{name}.npy")
        y = np.load(split_path / f"y_{name}.npy")

        print(f"Block [{name.upper()}]:")
        print(f"  Samples: {X.shape[0]}")
        print(f"  X mean/std: {X.mean():.4f} / {X.std():.4f}")
        print(f"  y mean/std: {y.mean():.4f} / {y.std():.4f}")

        # Check for NaN or Inf (common in division by zero during projection_and_stencils)
        if not np.all(np.isfinite(X)):
            print(f"  ❌ CRITICAL: NaNs or Infs found in X_{name}!")
