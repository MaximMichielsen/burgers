"""SGS predictor: MLP training, loading, and diagnostics.

Architecture (Pusuluri 2021 §3.3.2):
    5 × 256 ReLU hidden layers | Input: 20 | Output: 5
    Optimizer: NAdam | Loss: MSE | Early-stop metric: val MAE
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
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_UNITS,
    INPUT_UNITS,
    LEARNING_RATE,
    NUM_HIDDEN_LAYERS,
    OUTPUT_UNITS,
)
from ml.a_priori_verification import run_apriori_verification


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split_data(data_path: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load train/val float32 tensors from CSV files in data_path."""
    data_path = Path(data_path)

    def _read(filename: str) -> Tensor:
        array = np.loadtxt(data_path / filename, delimiter=",", skiprows=1)
        return torch.tensor(array, dtype=torch.float32)

    return (
        _read("X_train.csv"),
        _read("y_train.csv"),
        _read("X_val.csv"),
        _read("y_val.csv"),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SGSPredictor(nn.Module):
    """Uniform MLP for SGS closure prediction (num_hidden_layers × hidden_dim, ReLU).

    Default: 5 × 256 per Pusuluri (2021) §3.3.2.
    """

    def __init__(
        self,
        input_dim: int = INPUT_UNITS,
        hidden_dim: int = HIDDEN_UNITS,
        num_hidden_layers: int = NUM_HIDDEN_LAYERS,
        output_dim: int = OUTPUT_UNITS,
    ) -> None:
        super().__init__()
        layer_list: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layer_list += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layer_list.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layer_list)

    def forward(self, x_input: Tensor) -> Tensor:
        """Forward pass."""
        return self.net(x_input)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Stop training when validation MAE stops improving; restores best weights."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_loss: float = float("inf")
        self.early_stop: bool = False
        self.best_state: dict | None = None

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """Update state; save best weights when validation loss improves."""
        if val_loss < self.best_loss - self.min_delta:
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
) -> tuple[SGSPredictor, dict]:
    """Train the SGS predictor; return (best_model, training_stats)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val = load_split_data(data_path)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = SGSPredictor()
    optimizer = optim.NAdam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    early_stopper = EarlyStopping(patience=30)
    mse_loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()

    training_stats: dict[str, list[float]] = {
        "train_mse": [],
        "train_mae": [],
        "val_mae": [],
    }

    print(f"\nTraining SGS predictor — {x_train.shape[0]} training samples")
    print(
        f"Architecture: {INPUT_UNITS} → {NUM_HIDDEN_LAYERS}×{HIDDEN_UNITS} → {OUTPUT_UNITS}"
    )
    print("-" * 56)

    for epoch_idx in range(EPOCHS):
        model.train()
        epoch_mse_total = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss_value = mse_loss_fn(model(batch_x), batch_y)
            loss_value.backward()
            optimizer.step()
            epoch_mse_total += loss_value.item()

        epoch_train_mse = epoch_mse_total / len(train_loader)

        model.eval()
        with torch.no_grad():
            train_mae_value = mae_loss_fn(model(x_train), y_train).item()
            val_mae_value = mae_loss_fn(model(x_val), y_val).item()

        scheduler.step(val_mae_value)
        early_stopper(val_mae_value, model)

        training_stats["train_mse"].append(epoch_train_mse)
        training_stats["train_mae"].append(train_mae_value)
        training_stats["val_mae"].append(val_mae_value)

        if epoch_idx % 10 == 0 or epoch_idx == EPOCHS - 1:
            gap = val_mae_value - train_mae_value
            print(
                f"Epoch {epoch_idx:04d} | "
                f"Train MSE: {epoch_train_mse:.5f} | "
                f"Train MAE: {train_mae_value:.5f} | "
                f"Val MAE: {val_mae_value:.5f} | "
                f"Gap: {gap:+.5f} ({'STABLE' if abs(gap) < 0.05 else 'DIVERGING'})"
            )

        if early_stopper.early_stop:
            print(
                f"\nEarly stopping at epoch {epoch_idx}. Best val MAE: {early_stopper.best_loss:.6f}"
            )
            break

    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)

    model_save_path = output_dir / "sgs_predictor.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_UNITS,
            "hidden_dim": HIDDEN_UNITS,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "output_dim": OUTPUT_UNITS,
        },
        model_save_path,
    )
    print(f"\nModel saved to '{model_save_path}'.")
    return model, training_stats


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_predictor(model_path: Path) -> SGSPredictor:
    """Load a saved SGSPredictor from a .pt checkpoint."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = SGSPredictor(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_hidden_layers=checkpoint["num_hidden_layers"],
        output_dim=checkpoint["output_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Validation set evaluation
# ---------------------------------------------------------------------------


def evaluate_on_val_set(
    model: SGSPredictor,
    data_path: Path,
    output_dir: Path,
) -> NDArray:
    """Evaluate on validation data; log MSE/MAE and return predictions.

    Returns raw (normalised-space) model output of shape (n_val, OUTPUT_UNITS).
    """
    data_path = Path(data_path)

    def _read(filename: str) -> Tensor:
        array = np.loadtxt(data_path / filename, delimiter=",", skiprows=1)
        return torch.tensor(array, dtype=torch.float32)

    x_val = _read("X_val.csv")
    y_val = _read("y_val.csv")

    model.eval()
    with torch.no_grad():
        predictions_tensor = model(x_val)
        mse_value = nn.functional.mse_loss(predictions_tensor, y_val).item()
        mae_value = nn.functional.l1_loss(predictions_tensor, y_val).item()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_lines = [
        "=" * 40,
        "  VALIDATION SET EVALUATION",
        f"  MSE : {mse_value:.6f}",
        f"  MAE : {mae_value:.6f}",
        "=" * 40,
    ]
    print("\n" + "\n".join(log_lines))

    log_path = output_dir / "val_evaluation.log"
    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"Validation evaluation log saved to '{log_path}'.")

    return predictions_tensor.numpy()


# ---------------------------------------------------------------------------
# Training diagnostics plot
# ---------------------------------------------------------------------------


def plot_training_diagnostics(
    training_stats: dict,
    output_dir: Path | str,
    show_fig: bool = False,
) -> None:
    """Plot MSE convergence and MAE generalisation gap; save to output_dir."""
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
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _data_path = (
        Path(__file__).parent.parent.parent / "test_suite" / "training_data" / "sgsp"
    )
    _output_dir = Path(__file__).parent.parent.parent / "test_suite" / "models" / "sgsp"

    _model, _training_stats = train_predictor(
        data_path=_data_path,
        output_dir=_output_dir,
    )
    plot_training_diagnostics(
        training_stats=_training_stats,
        output_dir=_output_dir,
        show_fig=False,
    )
    evaluate_on_val_set(
        model=_model,
        data_path=_data_path,
        output_dir=_output_dir,
    )
    run_apriori_verification(
        model=_model,
        data_dir=_data_path,
        output_dir=_output_dir,
        domain_length=1.0,
        n_elements=8,
    )
