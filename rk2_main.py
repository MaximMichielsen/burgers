"""Main execution file, runs all important blocks."""

from dataclasses import replace
from pathlib import Path

import matplotlib

from ml.corrector_training.online_trainer import SACConfig, BurgersAVCEnvironment, SACAgent, OnlineAVTrainer
from ml.corrector_training.projection_schedule import ProjectionReferenceSchedule
from ml.ml_agents.corrector import AVController, save_corrector
from ml.ml_agents.solver_configs import AVCConfig
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
        problem, disc_cfg, simulation_mode="tau_model", tau_model="shakib_one", master_path=paths.les_shakib_one_data
    )
    solver_shakib_one.run_simulation()
    solver_shakib_one.post_processing()

    solver_shakib_two = BaseRK2(
        problem, disc_cfg, simulation_mode="tau_model", tau_model="shakib_two", master_path=paths.les_shakib_one_data
    )
    solver_shakib_two.run_simulation()
    solver_shakib_two.post_processing()

    solver_shakib_three = BaseRK2(
        problem, disc_cfg, simulation_mode="tau_model", tau_model="shakib_three", master_path=paths.les_shakib_one_data
    )
    solver_shakib_three.run_simulation()
    solver_shakib_three.post_processing()

    solver_no_model = BaseRK2(problem, disc_cfg, "no_model", paths.les_nm_data)
    solver_no_model.run_simulation()
    solver_no_model.post_processing()

    # --------------------------------------- GAVC training --------------------------------------- #
    if not paths.avc_gg_model.exists() and paths.projection is not None:
        proj_reference_schedule = ProjectionReferenceSchedule.from_projection_directory(
            projection_dir=paths.projection,
            domain_length=problem.domain_length,
            n_wavenumber_bins=(n_nodes_les + 1) // 2,
        )

        avc_trainer_cfg_global = AVCConfig(
            avc_model_path=paths.avc_gg_model,
            simulation_mode="avc",
            correction_mode="global",
            n_skip_steps=AVC_N_SKIP,
            n_wavenumber_bins=(n_nodes_les + 1) // 2,
            externally_driven=True,
        )

        av_corrector_global = AVController(
            n_wavenumber_bins=(n_nodes_les + 1) // 2,
            output_scale=avc_output_scale,
            correction_mode="global",
            output_dim=1,
        )
        paths.agents.mkdir(parents=True, exist_ok=True)
        save_corrector(av_corrector_global, paths.avc_gg_model)

        sac_cfg = SACConfig(
            n_skip_steps=AVC_N_SKIP,
            warmup_steps=AVC_WARMUP_STEPS,
            batch_size=AVC_BATCH_SIZE,
            tau_transient_warmup=TAU_REWARD_WARMUP,
        )
        environment_global = BurgersAVCEnvironment(
            problem=problem,
            disc_cfg=disc_cfg,
            avc_cfg=avc_trainer_cfg_global,
            master_path=paths.avc_data / "training_global",
            sac_config=sac_cfg,
            proj_ref_schedule=proj_reference_schedule,
        )
        sac_agent_global = SACAgent(
            av_corrector=av_corrector_global,
            state_dim=environment_global.state_dim,
            sac_config=sac_cfg,
        )
        trainer_global = OnlineAVTrainer(
            environment=environment_global,
            sac_agent=sac_agent_global,
            sac_config=sac_cfg,
            output_dir=paths.agents / "avcg_checkpoints",
        )
        trainer_global.train(n_episodes=AVC_EPOCHS)
        save_corrector(sac_agent_global.policy, paths.avc_gg_model)

    # --------------------------------------- GAVC run --------------------------------------- #
    avc_run_cfg = AVCConfig(
        avc_model_path=paths.avc_gg_model,
        n_wavenumber_bins=(n_nodes_les + 1) // 2,
        correction_mode="global",
    )
    solver_avc_global = AVCSolverRK2(
        problem,
        disc_cfg,
        "avc",
        paths.avc_data / "global",
        avc_run_cfg,
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
