from pathlib import Path
import numpy as np

from final.ann_training import TD3Hyperparameters, TD3Trainer
from final.ann_coupling.reference_scheduler import ReferenceTrajectory
from ml.tau_ann import Scope, TauANNConfig
from setup.config_discretization import DiscretizationConfig
from setup.problems import Problem
from solvers.solver_base import TauModel


def main():
    # Resolve project paths
    script_dir = Path(__file__).resolve().parent
    project_root = (
        script_dir.parent.parent
        if script_dir.name == "ann_coupling"
        else script_dir.parent
    )

    filtered_dir = project_root / "dns_data" / "projected"
    training_data_dir = project_root / "dns_data" / "training" / "training"

    output_dir = project_root / "final" / "solver_data" / "td3_training_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Problem & Discretization Definitions
    # -------------------------------------------------------------------------
    n_nodes_les = 65
    z_length = 2.0
    dt = 1e-4
    t_start = 308.0101
    t_end = 309.0100

    ic_les = np.load(filtered_dir / f"ic_linear_{n_nodes_les}.npy")
    forcing_projected = np.load(filtered_dir / f"forcing_l2_{n_nodes_les}.npy")

    n_timesteps, _ = np.shape(forcing_projected)
    timespan = n_timesteps * dt
    if t_end is not None:
        timespan = t_end - t_start

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
    )

    disc_config = DiscretizationConfig(
        n_nodes_les=n_nodes_les,
        temporal_refinement=1,
        courant_les=1,
        domain_length=z_length,
        domain_timespan=1,
    )
    disc_config.dt_les = dt
    disc_config.n_nodes_dns = 513

    print(f"Project Root:          {project_root}")
    print(f"Training Data Dir:     {training_data_dir}")
    print(f"Output Artifacts Dir:  {output_dir}")

    # -------------------------------------------------------------------------
    # Instantiate ReferenceTrajectory with direct file paths
    # -------------------------------------------------------------------------
    reference_schedule = ReferenceTrajectory(
        target_profile_path=training_data_dir / "stat_mean_w.npy",
        projected_w_path=training_data_dir / f"w_field_{n_nodes_les}.npy",
        target_time_index=1,
        n_les_nodes=n_nodes_les,
    )

    # Policy Configuration
    ann_config = TauANNConfig(
        output_scope=Scope.GLOBAL,
        tau_model=TauModel.TWO_PARAMS,
        n_wavenumber_bins=(n_nodes_les + 1) // 2,
        n_skip_steps=1,
        n_nodes_les=n_nodes_les,
        n_local_action_groups=1,
        ann_path=output_dir / "best_tau_ann.pt",
    )

    # TD3 Hyperparameters
    hyperparams = TD3Hyperparameters(
        total_episodes=100,
        stochastic_timesteps=1200,
        reduced_parameter_timesteps=800,
        batch_size=64,
        lr=1e-4,
        expl_noise=0.1,
    )

    # Trainer Setup
    trainer = TD3Trainer(
        problem=problem,
        disc_config=disc_config,
        ann_config=ann_config,
        master_path=output_dir,
        reference_schedule=reference_schedule,
        hp=hyperparams,
    )

    # Execute
    print("\n=======================================")
    print("      Starting TD3 Training Loop       ")
    print("=======================================\n")

    trained_actor, best_actions = trainer.run_td3_tau_ann_training()

    print("\n=======================================")
    print("    Training Completed Successfully    ")
    print("=======================================\n")

    print("Generating Reward Evolution Plot...")
    trainer.plot_reward_evolution(show_plot=True)


if __name__ == "__main__":
    main()
