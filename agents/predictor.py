import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from fem.constants import HIDDEN_UNITS, INPUT_UNITS, OUTPUT_UNITS, EPOCHS, BATCH_SIZE, LEARNING_RATE


def load_split_data(path: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load the split training data and transform to tensor."""
    X_train = torch.tensor(np.load(path / "X_train.npy"), dtype=torch.float32)
    y_train = torch.tensor(np.load(path / "y_train.npy"), dtype=torch.float32)
    X_val = torch.tensor(np.load(path / "X_val.npy"), dtype=torch.float32)
    y_val = torch.tensor(np.load(path / "y_val.npy"), dtype=torch.float32)
    return X_train, y_train, X_val, y_val


class SGSPredictor(nn.Module):
    def __init__(self, input_dim=INPUT_UNITS, output_dim=OUTPUT_UNITS):
        super(SGSPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
            self.counter = 0


def train_and_diagnose(data_path: Path):
    # Load all three blocks
    X_train, y_train, X_val, y_val = load_split_data(data_path)
    X_test = torch.tensor(np.load(data_path / "X_test.npy"), dtype=torch.float32)
    y_test = torch.tensor(np.load(data_path / "y_test.npy"), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    model = SGSPredictor()
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    early_stopper = EarlyStopping(patience=15)
    mse_loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()

    stats = {"train_mae": [], "val_mae": [], "train_mse": []}

    print(f"Starting Offline Training: {X_train.shape[0]} samples")
    print("-" * 50)

    for epoch in range(EPOCHS):
        model.train()
        epoch_mse = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = mse_loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_mse += loss.item()

        # Phase 1: Overfitting Diagnostic
        model.eval()
        with torch.no_grad():
            # Calculate metrics on full blocks for interpretation
            train_preds = model(X_train)
            val_preds = model(X_val)

            t_mae = mae_loss_fn(train_preds, y_train).item()
            v_mae = mae_loss_fn(val_preds, y_val).item()
            scheduler.step(v_mae)

            early_stopper(v_mae, model)

            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch}. Restoring best weights.")
                model.load_state_dict(early_stopper.best_state)
                break

            stats["train_mae"].append(t_mae)
            stats["val_mae"].append(v_mae)
            stats["train_mse"].append(epoch_mse / len(train_loader))

            if epoch % 5 == 0 or epoch == EPOCHS - 1:
                gap = v_mae - t_mae
                status = "STABLE" if gap < 0.05 else "DIVERGING"
                print(
                    f"Epoch {epoch:03d} | Train MSE: {stats['train_mse'][-1]:.5f} | "
                    f"Val MAE: {v_mae:.5f} | Gap: {gap:.5f} ({status})"
                )

    return model, stats, (X_test, y_test)


def evaluate_test_performance(model, test_data):
    X_test, y_test = test_data
    model.eval()
    with torch.no_grad():
        preds = model(X_test)

        # Point-wise Error Analysis
        mse = nn.functional.mse_loss(preds, y_test).item()
        mae = nn.functional.l1_loss(preds, y_test).item()

        print("\n" + "=" * 30)
        print("STAGE 1: TEST DATA ASSESSMENT")
        print(f"Direct MSE: {mse:.6f}")
        print(f"Direct MAE: {mae:.6f}")
        print("=" * 30)

        return preds.numpy()


def plot_training_diagnostics(stats):
    epochs = range(len(stats["train_mae"]))

    plt.figure(figsize=(12, 5))

    # Subplot 1: MSE Loss (Convergence)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, stats["train_mse"], label="Train MSE", color="orange")
    plt.title("Training Convergence (MSE)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")  # Log scale helps see convergence at small values
    plt.legend()

    # Subplot 2: MAE (Generalization Diagnostic)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, stats["train_mae"], label="Train MAE", color="blue")
    plt.plot(epochs, stats["val_mae"], label="Val MAE", color="green", linestyle="--")
    plt.title("Generalization Diagnostic (MAE)")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()

    plt.tight_layout()
    plt.show()
