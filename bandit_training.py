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

n_nodes_les = 17
n_nodes_dns = 513
z_length = 2.0

dt = 1e-3
t_start = 308.0101
t_end = 309.0100

n_episodes = 9
proof_mode = ProofMode.a

run_dir = (
    CURRENT_DIR / "solver_data" / "proof_experiments" / f"proof_of_concept_{n_episodes}"
)

run_dir.mkdir(parents=True, exist_ok=True)

tau_model = TauModel.TWO_PARAMS

ic_les = np.load(FILTERED_DIR / f"ic_linear_{n_nodes_les}.npy")
forcing_projected = np.load(TEMPORAL_DIR / f"forcing_l2_{n_nodes_les}.npy")

timespan = t_end - t_start

ann_path = run_dir / "ann_model.pt"
target_profile_path = TRAINING_DIR / f"n{n_nodes_les}" / "training" / "stat_mean_w.npy"
w_field_path = (
    TRAINING_DIR / f"n{n_nodes_les}" / "training" / f"w_field_{n_nodes_les}.npy"
)

problem = Problem(
    name="tcf_1d",
    domain_length=z_length,
    domain_timespan=timespan,
    reynolds=100.0,
    initial_condition=ic_les,
    forcing=forcing_projected,
    forcing_is_steady=False,
    boundary_condition_type="fixed",
    boundary_condition_value=0.0,
    t_start=t_start,
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
    run_final_evaluation=False,
    n_skip_steps=1,
)

reference_trajectory = ReferenceTrajectory(
    target_profile_path=target_profile_path,
    projected_w_path=w_field_path,
    n_les_nodes=n_nodes_les,
    target_time_index=0,
)

trainer = TD3Trainer(
    problem,
    disc_config,
    ann_config,
    master_path=run_dir,
    reference_trajectory=reference_trajectory,
)

# Execute training and plot diagnostics
trainer.run_training()
trainer.plot_reward_evolution()
trainer.run_diagnostic_plotting()

print(f"\n[SUCCESS] Mode '{proof_mode.value}' proof experiment completed.")
