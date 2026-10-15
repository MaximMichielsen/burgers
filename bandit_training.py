from pathlib import Path

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

n_nodes_les = 17
n_nodes_dns = 513
z_length = 2.0

dt = 1e-3
t_start = 312.01
t_end = 313.01
t_start_test = 314.01
t_end_test = 315.0

n_episodes = 10
proof_mode = ProofMode.a

link = True
use_reynolds_signal = False

sub_dir = "no_reynolds_signal" if not use_reynolds_signal else "with_reynolds_signal"

run_dir = (
    CURRENT_DIR
    / "solver_data"
    / "proof_experiments"
    / "testing_testing"
    / sub_dir
    / f"train_{t_start}_{t_end}_test_{t_start_test}_{t_end_test}"
    / f"proof_of_concept_{n_episodes}_{n_nodes_les}_{link}_{dt}"
)

run_dir.mkdir(parents=True, exist_ok=True)

tau_model = TauModel.TWO_PARAMS

timespan = t_end - t_start

ann_path = run_dir / "ann_model.pt"
target_profile_dir = PROJECT_ROOT / "dns_data" / "intervals"
w_field_path = PROJECT_ROOT / "dns_data" / "projected" / f"w_linear_{n_nodes_les}.npy"
# Load times and forcing
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
    domain_timespan=timespan,
)

ann_config = ANNBanditConfig(
    tau_model,
    disc_config,
    ann_path=ann_path,
    output_scope=Scope.GLOBAL,
    input_scope=Scope.GLOBAL,
    n_training_episodes=n_episodes,
    training_mode=True,
    run_final_evaluation=True,
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
)

# Execute training and plot diagnostics
trainer.run_training()
trainer.run_testing()

print(f"\n[SUCCESS] Mode '{proof_mode.value}' proof experiment completed.")
