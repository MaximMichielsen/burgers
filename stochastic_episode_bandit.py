from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from ml.ann_config import Scope
from ml.reference_scheduler import ReferenceTrajectory
from ml.td3 import TD3Trainer
from proofing.action_based.config import ProofMode
from proofing.contextual_bandit_reformulation.config import ANNBanditConfig
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import TauModel

PROJECT_ROOT = Path(__file__).resolve().parent
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
    """Generates valid training interval permutations (excluding overlaps with test interval)
    and returns both the training pool and the designated testing interval.
    """
    if not base_intervals:
        return [], (0.0, 0.0)

    if test_interval is None:
        test_interval = (312.01, 313.01)

    # Step 1: Group base intervals into continuous blocks (respecting the gap at 310-311)
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
    raw_permutations = []
    for block in blocks:
        n = len(block)
        for i in range(n):
            for j in range(i, n):
                start = block[i][0]
                end = block[j][1]
                duration = end - start

                if duration <= max_duration + 1e-5:
                    raw_permutations.append((start, end))

    # Step 3: Exclude any training interval that overlaps with the testing interval
    test_start, test_end = test_interval
    training_permutations = []

    for start, end in raw_permutations:
        is_overlapping = (start < test_end - 1e-4) and (end > test_start + 1e-4)
        if not is_overlapping:
            training_permutations.append((start, end))

    return training_permutations, test_interval


# Define base intervals from directory structure
base_intervals = [
    (308.0101, 309.01),
    (309.01, 310.01),
    (311.01, 312.01),
    (312.01, 313.01),
    (313.01, 314.01),
    (314.01, 315.0),
]

train_intervals, (t_start_test, t_end_test) = generate_training_and_testing_intervals(
    base_intervals, max_duration=1.0
)

# Initial fallback training bounds for setup (overwritten stochastically per episode)
t_start, t_end = train_intervals[0]

n_nodes_les = 17
n_nodes_dns = 513
z_length = 2.0
dt = 1e-3

n_episodes = 4
proof_mode = ProofMode.a

link = False
use_reynolds_signal = False

sub_dir = "no_reynolds_signal" if not use_reynolds_signal else "with_reynolds_signal"

run_dir = (
    CURRENT_DIR
    / "solver_data"
    / "proof_experiments"
    / "stochastic_intervals"
    / sub_dir
    / f"test_{t_start_test}_{t_end_test}"
    / f"proof_of_concept_{n_episodes}_{n_nodes_les}_{link}_{dt}"
)

run_dir.mkdir(parents=True, exist_ok=True)

tau_model = TauModel.TWO_PARAMS
timespan = t_end - t_start

ann_path = run_dir / "ann_model.pt"
target_profile_dir = PROJECT_ROOT / "dns_data" / "intervals"
w_field_path = PROJECT_ROOT / "dns_data" / "projected" / f"w_linear_{n_nodes_les}.npy"
times_path = PROJECT_ROOT / "dns_data" / "curated" / "dat" / "times.npy"
forcing_projected = np.load(TEMPORAL_PROJECTED_DIR / f"forcing_l2_{n_nodes_les}.npy")
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

test_problem = Problem(
    name="tcf_1d",
    domain_length=z_length,
    reynolds=100.0,
    initial_condition=None,
    forcing=forcing_projected,
    forcing_times=times_forcing,
    forcing_is_steady=False,
    boundary_condition_type="fixed",
    boundary_condition_value=0.0,
    t_start=t_start_test,
    t_end=t_end_test,
)

disc_config = DiscretizationConfig(
    n_nodes_les=n_nodes_les,
    n_nodes_dns=n_nodes_dns,
    temporal_refinement=1,
    set_dt_les=dt,
    domain_length=z_length,
    t_start=t_start,
    t_end=t_end,
)

ann_config = ANNBanditConfig(
    tau_model,
    disc_config,
    ann_path=ann_path,
    output_scope=Scope.GLOBAL,
    input_scope=Scope.GLOBAL,
    n_training_episodes=n_episodes,
    training_mode=True,
    run_final_evaluation=False,
    n_skip_steps=1,
    link_coefficients=link,
    include_reynolds_stress_reward_signal=use_reynolds_signal,
)

reference_trajectory = ReferenceTrajectory(
    target_profile_dir=target_profile_dir,
    projected_w_path=w_field_path,
    times_path=times_path,
    n_les_nodes=n_nodes_les,
    t_start=t_start,
    t_end=t_end,
    t_start_test=t_start_test,
    t_end_test=t_end_test,
)

trainer = TD3Trainer(
    problem,
    disc_config,
    ann_config,
    master_path=run_dir,
    reference_trajectory=reference_trajectory,
    test_problem=test_problem,
    train_intervals=train_intervals,  # Passes the stochastic interval pool
)

# Execute stochastic training and final testing evaluation
trainer.run_training()
trainer.run_testing()

print("\n[SUCCESS] Stochastic training and testing pipeline completed successfully.")
