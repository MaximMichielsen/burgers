"""SGS predictor model for VMS-ANN Burgers LES.

Architecture (Research Proposal §2.3.1, following Robijns 2019 / Pusuluri 2021):
    - 3 fully connected hidden layers, 64 units each, ReLU activation
    - Input:  20 features  (lagged extended stencil, FS2)
    - Output:  5 scalars   (cross, Reynolds, u't_L, u't_R, viscous SGS)
    - Optimizer: RMSprop
    - Loss (training):   MSE
    - Metric (monitoring): MAE on validation set
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from constants import (
    HIDDEN_UNITS,
    INPUT_UNITS,
    OUTPUT_UNITS,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split_data(
    path: Path,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load train/val tensors from *path*.

    Returns
    -------
    X_train, y_train, X_val, y_val : float32 Tensors
    """
    x_train = torch.tensor(np.load(path / "X_train.npy"), dtype=torch.float32)
    y_train = torch.tensor(np.load(path / "y_train.npy"), dtype=torch.float32)
    x_val = torch.tensor(np.load(path / "X_val.npy"), dtype=torch.float32)
    y_val = torch.tensor(np.load(path / "y_val.npy"), dtype=torch.float32)
    return x_train, y_train, x_val, y_val


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SGSPredictor(nn.Module):
    """Three-hidden-layer MLP for SGS closure prediction.

    Matches Research Proposal §2.3.1: 3 × 64 ReLU hidden layers.
    """

    def __init__(
        self,
        input_dim: int = INPUT_UNITS,
        hidden_dim: int = HIDDEN_UNITS,
        output_dim: int = OUTPUT_UNITS,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),  # linear output — no activation
        )

    def forward(self, x_input: Tensor) -> Tensor:
        """Forward pass."""
        return self.net(x_input)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Monitor validation MAE and stop when it stops improving.

    Saves the best model weights for restoration on stop.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_loss: float | None = None
        self.early_stop: bool = False
        self.best_state: dict | None = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """Update state given the latest validation loss."""
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_predictor(
    data_path: Path,
    output_dir: Path,
) -> tuple["SGSPredictor", dict]:
    """Train the SGS predictor on the assembled training data.

    Follows RP §2.3.1:
        - MSE loss for gradient updates
        - MAE monitored on validation set for early stopping
        - RMSprop optimiser
        - LR reduced on plateau

    Parameters
    ----------
    data_path:
        Directory with X_train.npy, y_train.npy, X_val.npy, y_val.npy.
    output_dir:
        Where to save the trained model and diagnostics.

    Returns
    -------
    model : SGSPredictor
        Best model (weights restored from best validation MAE checkpoint).
    training_stats : dict
        Keys: train_mse, train_mae, val_mae — one value per completed epoch.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val = load_split_data(data_path)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = SGSPredictor()
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    early_stopper = EarlyStopping(patience=15)
    mse_loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()

    training_stats: dict[str, list[float]] = {
        "train_mse": [],
        "train_mae": [],
        "val_mae": [],
    }

    print(f"\nTraining SGS predictor — {x_train.shape[0]} training samples")
    print(f"Architecture: {INPUT_UNITS} → 3×{HIDDEN_UNITS} → {OUTPUT_UNITS}")
    print("-" * 56)

    for epoch_idx in range(EPOCHS):
        # --- Training pass ---
        model.train()
        epoch_mse_total = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss_value = mse_loss_fn(prediction, batch_y)
            loss_value.backward()
            optimizer.step()
            epoch_mse_total += loss_value.item()

        epoch_train_mse = epoch_mse_total / len(train_loader)

        # --- Evaluation pass ---
        model.eval()
        with torch.no_grad():
            train_preds = model(x_train)
            val_preds = model(x_val)
            train_mae_value = mae_loss_fn(train_preds, y_train).item()
            val_mae_value = mae_loss_fn(val_preds, y_val).item()

        scheduler.step(val_mae_value)
        early_stopper(val_mae_value, model)

        training_stats["train_mse"].append(epoch_train_mse)
        training_stats["train_mae"].append(train_mae_value)
        training_stats["val_mae"].append(val_mae_value)

        if epoch_idx % 10 == 0 or epoch_idx == EPOCHS - 1:
            gap = val_mae_value - train_mae_value
            status = "STABLE" if abs(gap) < 0.05 else "DIVERGING"
            print(
                f"Epoch {epoch_idx:04d} | "
                f"Train MSE: {epoch_train_mse:.5f} | "
                f"Train MAE: {train_mae_value:.5f} | "
                f"Val MAE: {val_mae_value:.5f} | "
                f"Gap: {gap:+.5f} ({status})"
            )

        if early_stopper.early_stop:
            print(
                f"\nEarly stopping at epoch {epoch_idx}. "
                f"Best val MAE: {early_stopper.best_loss:.6f}"
            )
            break

    # Restore best weights
    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)

    # Save model
    model_save_path = output_dir / "sgs_predictor.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_UNITS,
            "hidden_dim": HIDDEN_UNITS,
            "output_dim": OUTPUT_UNITS,
        },
        model_save_path,
    )
    print(f"\nModel saved to '{model_save_path}'.")

    return model, training_stats


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_predictor(model_path: Path) -> "SGSPredictor":
    """Load a saved SGSPredictor from a .pt checkpoint."""
    checkpoint = torch.load(model_path, map_location="cpu")
    model = SGSPredictor(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        output_dim=checkpoint["output_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------


def evaluate_on_test_set(
    model: "SGSPredictor",
    data_path: Path,
    output_dir: Path,
) -> NDArray:
    """Evaluate the trained model on held-out test data and log results.

    Expects X_test.npy and y_test.npy in *data_path*.

    Returns
    -------
    predictions : NDArray of shape (n_test, OUTPUT_UNITS)
        Raw (normalised-space) predictions.
    """
    x_test = torch.tensor(np.load(data_path / "X_test.npy"), dtype=torch.float32)
    y_test = torch.tensor(np.load(data_path / "y_test.npy"), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions_tensor = model(x_test)
        mse_value = nn.functional.mse_loss(predictions_tensor, y_test).item()
        mae_value = nn.functional.l1_loss(predictions_tensor, y_test).item()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_lines = [
        "=" * 40,
        "  TEST SET EVALUATION",
        f"  MSE : {mse_value:.6f}",
        f"  MAE : {mae_value:.6f}",
        "=" * 40,
    ]
    print("\n" + "\n".join(log_lines))

    log_path = output_dir / "test_evaluation.log"
    with open(log_path, "w") as log_file:
        log_file.write("\n".join(log_lines) + "\n")
    print(f"Test evaluation log saved to '{log_path}'.")

    return predictions_tensor.numpy()


# ---------------------------------------------------------------------------
# Training diagnostics plot
# ---------------------------------------------------------------------------


def plot_training_diagnostics(
    training_stats: dict,
    output_dir: Path | str,
    show_fig: bool = False,
) -> None:
    """Plot MSE convergence and MAE generalisation gap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_range = range(len(training_stats["train_mae"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        epoch_range, training_stats["train_mse"], color="tab:orange", label="Train MSE"
    )
    axes[0].set_title("Training convergence (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epoch_range, training_stats["train_mae"], color="royalblue", label="Train MAE"
    )
    axes[1].plot(
        epoch_range,
        training_stats["val_mae"],
        color="tab:green",
        linestyle="--",
        label="Val MAE",
    )
    axes[1].set_title("Generalisation diagnostic (MAE)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "training_diagnostics.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved training diagnostics to '{save_path}'.")

    if show_fig:
        plt.show()
    plt.close(fig)
