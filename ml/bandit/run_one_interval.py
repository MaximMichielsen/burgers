from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from ml.bandit.ann import BanditConfig
from ml.reference_trajectory import BanditReferenceTrajectory
from ml.bandit.training import BanditTrainer
from proofing.action_based.config import ProofMode
from setup.config_discretization import DiscretizationConfig
from setup.constants import Z_LENGTH_DNS
from setup.problems import Problem
from solvers.solver_base import TauModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CURRENT_DIR = Path(__file__).resolve().parent

FILTERED_DIR = PROJECT_ROOT / "dns_data" / "projected"
TEMPORAL_DIR = PROJECT_ROOT / "dns_data" / "temporal" / "projected"
TRAINING_DIR = PROJECT_ROOT / "dns_data" / "training"
TEMPORAL_PROJECTED_DIR = PROJECT_ROOT / "dns_data" / "temporal" / "projected"


def generate_training_and_testing_intervals(
    base_intervals: List[Tuple[float, float]],
    max_duration: float = 2.0,
    test_interval: Optional[Tuple[float, float]] = None,
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """Generates valid training interval permutations (allowing overlaps with test interval)
    and returns both the training pool and the designated testing interval.
    """
    if not base_intervals:
        return [], (0.0, 0.0)

    if test_interval is None:
        test_interval = (312.01, 313.01)

    # Step 1: Group base intervals into continuous blocks
    blocks = []
    current_block = [base_intervals[0]]

    for i in range(1, len(base_intervals)):
        prev_end = current_block[-1][1]
        curr_start = base_intervals[i][0]

        if abs(curr_start - prev_end) < 1e-4:
            current_block.append(base_intervals[i])
        else:
            blocks.append(current_block)
            current_block = [base_intervals[i]]
    blocks.append(current_block)

    # Step 2: Find all sub-sequences within each block adhering to max_duration
    training_permutations = []
    for block in blocks:
        n = len(block)
        for i in range(n):
            for j in range(i, n):
                start = block[i][0]
                end = block[j][1]
                duration = end - start

                if duration <= max_duration + 1e-5:
                    training_permutations.append((start, end))

    # Overlap check removed — all valid duration sub-intervals are kept

    return training_permutations, test_interval


# Define base intervals from directory structure
base_intervals = [
    (308.0101, 309.01),
    # (309.01, 310.01),
    # (311.01, 312.01),
    # (313.01, 314.01),
]

test_interval = (308.0101, 309.01)

train_intervals, test_interval = generate_training_and_testing_intervals(
    base_intervals, max_duration=1.0, test_interval=test_interval
)

t_start, t_end = train_intervals[0]  # fallback

n_nodes_les = 17
n_nodes_dns = 513
z_length = Z_LENGTH_DNS
dt = 1e-3
tau_model = TauModel.TWO_PARAMS

n_episodes = 20
proof_mode = ProofMode.a
reynolds_signals = (False, True)

for use_reynolds_signal in reynolds_signals:
    sub_dir = (
        "no_reynolds_signal" if not use_reynolds_signal else "with_reynolds_signal"
    )

    run_dir = (
        PROJECT_ROOT
        / "solver_data"
        / "true_bandit_regularized"
        / "one_interval"
        / f"e{n_episodes}_n{n_nodes_les}_dt{dt}_p{tau_model.value}"
        / sub_dir
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    ann_path = run_dir / "ann_model.pt"
    target_profile_dir = PROJECT_ROOT / "dns_data" / "intervals"
    w_field_path = (
        PROJECT_ROOT / "dns_data" / "projected" / f"w_linear_{n_nodes_les}.npy"
    )
    times_path = PROJECT_ROOT / "dns_data" / "curated" / "dat" / "times.npy"
    forcing_projected = np.load(
        TEMPORAL_PROJECTED_DIR / f"forcing_l2_{n_nodes_les}.npy"
    )
    times_forcing = np.load(TEMPORAL_PROJECTED_DIR / "times.npy")

    problem = Problem(
        name="tcf_1d",
        domain_length=z_length,
        reynolds=100.0,
        initial_condition=None,
        forcing=forcing_projected,
        forcing_times=times_forcing,
        forcing_is_steady=False,
        boundary_condition_type="fixed",
        boundary_condition_value=0.0,
        t_start=t_start,
        t_end=t_end,
    )
    disc_config = DiscretizationConfig(
        problem=problem,
        n_nodes_les=n_nodes_les,
        n_nodes_dns=n_nodes_dns,
        dt_les=dt,
    )

    ann_config = BanditConfig(
        tau_model=tau_model,
        n_training_episodes=n_episodes,
        training_mode=True,
        include_reynolds_signal=use_reynolds_signal,
        ann_path=ann_path,
    )

    reference_trajectory = BanditReferenceTrajectory(
        mean_target_profile_dir=target_profile_dir,
        projected_w_path=w_field_path,
        times_path=times_path,
        n_nodes_les=n_nodes_les,
    )

    trainer = BanditTrainer(
        problem,
        disc_config,
        ann_config,
        master_path=run_dir,
        reference_trajectory=reference_trajectory,
        test_interval=test_interval,
        train_intervals=train_intervals,  # Passes the stochastic interval pool
    )

    # Execute stochastic training and final testing evaluation
    trainer.run_training()

    print(
        "\n[SUCCESS] Stochastic training and testing pipeline completed successfully."
    )
