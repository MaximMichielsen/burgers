import numpy as np
from pathlib import Path

# Quick inspection of what's saved
path = Path(
    r"C:\Users\poopy\PycharmProjects\burgers\runs\run_raj_two_0521_163124\training_data\pre_split"
)


projection_path = Path(path)

solutions = np.load(projection_path / "solutions_projection.npy")
print(f"Shape: {solutions.shape}")  # (T, N_nodes)
print(f"Spatial nodes: {solutions.shape[1]}")
print(f"Number of snapshots: {solutions.shape[0]}")

# Infer dt from the times saved in the CSV filenames if available
# Otherwise check your constants
try:
    from constants import DNS_TO_LES_RATIO

    print(f"DNS_TO_LES_RATIO: {DNS_TO_LES_RATIO}")
except ImportError:
    print("Could not import constants")

# Cross-check: what dt would give sensible Courant numbers?
# For Burgers u~O(1), h_les = 1/(N_nodes-1):
n_nodes = solutions.shape[1]
h_les = 1.0 / (n_nodes - 1)
print(f"\nh_LES = {h_les:.4f}")

# What dt values give Co = 0.01, 0.1, 1.0?
for co_target in [0.01, 0.1, 1.0]:
    dt_implied = co_target * h_les  # u ~ 1
    n_steps_implied = int(5.0 / dt_implied)
    print(
        f"  Co={co_target:.2f} → dt={dt_implied:.4f}, "
        f"steps over t=[0,5]: {n_steps_implied}"
    )

print(f"\nYour snapshot count: {solutions.shape[0]}")
print(f"Implied dt if t_end=5: {5.0 / solutions.shape[0]:.6f}")
print(f"Implied Courant (u~1): {(5.0 / solutions.shape[0]) / h_les:.4f}")
