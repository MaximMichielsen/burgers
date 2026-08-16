"""Main execution file, runs all important blocks."""

from dataclasses import replace
from pathlib import Path

import matplotlib

from ml.corrector_training.SAC import SACConfig, SACAgent
from ml.corrector_training.online_training import OnlineAVCTrainer, AVCTrainingConfig
from ml.corrector_training.projection_schedule import ProjectionReferenceSchedule
from ml.corrector_training.rl_environment import AVCEnvironment
from ml.ml_agents.corrector import AVCConfig, AVController, save_corrector
from problems_and_configurations.disc_config import DiscretizationConfig
from problems_and_configurations.problems import Problems, Problem
from solvers.explicit.avc_augment_rk2 import AVCSolverRK2

from solvers.explicit.base_solver_rk2 import BaseRK2
from utils.plotting.dissipation_evolution import plot_dissipation_comparison
from utils.plotting.energy_evolution import plot_energy_comparison
from utils.pipeline_utils import (
    run_dns,
    resolve_pathing,
)
from utils.plotting.velocity_profiles import (
    plot_solution_comparison,
    create_velocity_plot_configs,
)

# todo: move n_wavenumberbins to disc confg?

CURRENT_DIR = Path(__file__).parent.resolve()
matplotlib.use("Agg")  # needed when running on M12

# -------------------- Problem and pipeline configuration ------------------------------ #
problem: Problem = Problems.raj_one
problem = replace(problem, domain_timespan=1.0)

# general simulation parameters
n_nodes_les: int = 9
temporal_refinement: int = 1
courant_les: float = 0.1

# discretization config
disc_cfg = DiscretizationConfig(
    n_nodes_les,
    temporal_refinement,
    courant_les,
    problem.domain_length,
)

# pathing
paths = resolve_pathing(problem.name, CURRENT_DIR)

# AVC (hyper-)parameters
avc_output_scale = 2 * problem.viscosity

AVC_EPOCHS: int = 10
AVC_N_SKIP: int = 5
AVC_WARMUP_STEPS: int = 500
AVC_BATCH_SIZE = 264
TAU_REWARD_WARMUP: float = 0.2

# DEBUG FLAGS
AVC_ZERO_RUN: bool = False
SET_OFF_SGSP: bool = False

if __name__ == "__main__":
    # ----------------------------------------- DNS data --------------------------------------------- #
    DNS_CACHE_ROOT = CURRENT_DIR / "dns_cache"
    run_dns(DNS_CACHE_ROOT, problem, disc_cfg, paths)

    # ----------------------------------------- LES solvers ------------------------------------------ #
    solver_shakib_one = BaseRK2(
        problem,
        disc_cfg,
        simulation_mode="tau_model",
        tau_model="shakib_one",
        master_path=paths.les_shakib_one_data,
    )
    solver_shakib_one.run_simulation()
    solver_shakib_one.post_processing()

    solver_shakib_two = BaseRK2(
        problem,
        disc_cfg,
        simulation_mode="tau_model",
        tau_model="shakib_two",
        master_path=paths.les_shakib_one_data,
    )
    solver_shakib_two.run_simulation()
    solver_shakib_two.post_processing()

    solver_shakib_three = BaseRK2(
        problem,
        disc_cfg,
        simulation_mode="tau_model",
        tau_model="shakib_three",
        master_path=paths.les_shakib_one_data,
    )
    solver_shakib_three.run_simulation()
    solver_shakib_three.post_processing()

    solver_no_model = BaseRK2(problem, disc_cfg, "no_model", paths.les_nm_data)
    solver_no_model.run_simulation()
    solver_no_model.post_processing()

    # --------------------------------------- GAVC training --------------------------------------- #
    if not paths.avc_gg_model.exists() and paths.projection is not None:
        n_wavenumber_bins = (n_nodes_les + 1) // 2

        proj_reference_schedule = ProjectionReferenceSchedule.from_projection_directory(
            projection_dir=paths.projection,
            domain_length=problem.domain_length,
            n_wavenumber_bins=n_wavenumber_bins,
        )

        avc_config = AVCConfig(
            avc_model_path=paths.avc_gg_model,
            input_scope="global",
            output_scope="global",
            n_skip_steps=AVC_N_SKIP,
            n_wavenumber_bins=n_wavenumber_bins,
        )

        av_corrector = AVController(avc_config)
        paths.agents.mkdir(parents=True, exist_ok=True)
        save_corrector(av_corrector, paths.avc_gg_model)

        training_config = AVCTrainingConfig()
        sac_config = SACConfig()

        environment = AVCEnvironment(
            problem=problem,
            disc_config=disc_cfg,
            avc_config=avc_config,
            avc_training_config=AVCTrainingConfig,
            simulation_mode="no_model",
            master_path=paths.avc_data / "training_global",
            proj_ref_schedule=proj_reference_schedule,
        )
        sac_agent = SACAgent(
            av_corrector=av_corrector,
            state_dim=av_corrector.output_dimensions,
            sac_config=sac_config,
            training_config=training_config,
        )
        trainer = OnlineAVCTrainer(
            environment=environment,
            agent=sac_agent,
            agent_config=sac_config,
            training_config=training_config,
            output_dir=paths.agents / "avcg_checkpoints",
        )
        trainer.train(n_episodes=AVC_EPOCHS)
        save_corrector(sac_agent.policy, paths.avc_gg_model)

    # --------------------------------------- GAVC run --------------------------------------- #
    solver_avc_global = AVCSolverRK2(
        problem=problem,
        disc_config=disc_cfg,
        avc_config=avc_config,
        simulation_mode="no_model",
        master_path=paths.avc_data / "global",
    )
    solver_avc_global.run_simulation()
    solver_avc_global.post_processing()

    # -------------------------------------- Plotting --------------------------------------- #
    plot_solution_comparison(
        configs=create_velocity_plot_configs(paths, disc_cfg),
        output_path=paths.master,
    )

    plot_energy_comparison(
        paths=paths,
        output_path=paths.master,
        domain_length=problem.domain_length,
    )

    plot_dissipation_comparison(
        paths, paths.master, problem.viscosity, problem.domain_length
    )
