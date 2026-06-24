"""Main execution file, runs all important blocks."""

from dataclasses import replace
from pathlib import Path

import matplotlib

from ml.corrector_training.DNS_snapshot_converter import DNSReferenceSchedule
from ml.corrector_training.online_trainer import (
    SACConfig,
    BurgersAVCEnvironment,
    SACAgent,
    OnlineAVTrainer,
)
from ml.ml_agents.corrector import AVController, save_corrector
from problems_and_configurations.disc_config import DiscretizationConfig
from problems_and_configurations.problems import Problems, Problem
from solvers.burgers_avc import BurgersAVC
from solvers.burgers_base import BurgersBase
from ml.ml_agents.solver_configs import SGSPConfig, AVCRunConfig, AVCTrainerConfig
from utils.plotting.energy_evolution import plot_energy_comparison
from utils.io_utils import (
    load_first_projected_solution,
)
from utils.pipeline_utils import (
    run_dns,
    run_sgsp_training,
    run_sgsp_coupled_solver,
    resolve_pathing,
    load_manual_models,
)
from utils.plotting.velocity_profiles import (
    plot_solution_comparison,
    create_velocity_plot_configs,
)


CURRENT_DIR = Path(__file__).parent.resolve()
matplotlib.use("Agg")  # needed when running on M12

# -------------------- Problem and pipeline configuration ------------------------------ #
problem: Problem = Problems.pipeline_test
problem = replace(problem, domain_timespan=0.5)

clip_pusuluri: bool = True
clip_rajampeta: bool = False

run_analytical_les: bool = False
run_no_model_les: bool = False

# general simulation parameters
n_nodes_les: int = 17
temporal_refinement: int = 1
courant_les: float = 0.1

# AVC (hyper-)parameters
AVC_ALPHA_MAX: float = problem.viscosity / 2  # currently the parameter that
# sets the lower and upper limits for random batch filling

AVC_EPOCHS: int = 10
AVC_N_SKIP: int = 5

# discretization dict
disc_cfg = DiscretizationConfig(
    n_nodes_les,
    temporal_refinement,
    courant_les,
    problem.domain_length,
)

# pathing
paths = resolve_pathing(problem.name, CURRENT_DIR)

manual_load_sgsp_model: str = r""
training_path: str = r""

manual_load_avcg_model: str = r""

load_manual_models(paths, manual_load_sgsp_model, training_path, manual_load_avcg_model)

if __name__ == "__main__":
    # --------------------------------------- DNS & SGSP data --------------------------------------- #
    DNS_CACHE_ROOT = CURRENT_DIR / "dns_cache"
    run_dns(DNS_CACHE_ROOT, problem, disc_cfg, paths)

    # --------------------------------------- SGSP training --------------------------------------- #
    if not paths.sgsp_model.exists() and paths.training is not None:
        run_sgsp_training(
            data_path=paths.training,
            output_dir=paths.agents,
            domain_length=problem.domain_length,
            n_elements=n_nodes_les - 1,
        )

    # --------------------------------------- SGSP coupled solver --------------------------------------- #
    sgsp_cfg = SGSPConfig(
        sgsp_model_path=paths.sgsp_model,
        normalization_path=paths.training,  # contains normalisation_stats.csv
        blown_up_path=paths.les_sgsp_data / "blown_up",
        clip_pusuluri=clip_pusuluri,
        clip_rajampeta=clip_rajampeta,
        turn_off_predictor=False,
    )
    run_sgsp_coupled_solver(problem, disc_cfg, paths.les_sgsp_data, sgsp_cfg)

    # --------------------------------------- Bare LES solvers --------------------------------------- #
    if run_analytical_les:
        les_run = BurgersBase(problem, disc_cfg, "les", paths.les_a_data)
        les_run.run_simulation()

    if run_no_model_les:
        no_model_run = BurgersBase(problem, disc_cfg, "no_model", paths.les_nm_data)
        no_model_run.run_simulation()

    # --------------------------------------- GAVC training --------------------------------------- #
    if not paths.avcg_model.exists():
        dns_solution_on_les = load_first_projected_solution(paths.projection)

        dns_solver_ref = BurgersBase(
            problem,
            disc_cfg,
            simulation_mode="no_model",
            master_path=paths.dns_data,
        )
        _, dns_positive_spectrum = dns_solver_ref.get_positive_spectrum(
            *dns_solver_ref.compute_energy_spectrum(dns_solution_on_les)
        )
        dns_dissipation_ref = float(
            dns_solver_ref.compute_dissipation(dns_solution_on_les)
        )
        n_wavenumber_bins = len(dns_positive_spectrum)
        dns_reference_schedule = DNSReferenceSchedule.from_projection_directory(
            projection_dir=paths.projection,
            domain_length=problem.domain_length,
            viscosity=problem.viscosity,
            n_wavenumber_bins=n_wavenumber_bins,
        )

        avc_trainer_cfg = AVCTrainerConfig(
            paths.avcg_model,
            dns_energy_spectrum=dns_positive_spectrum,
            dns_dissipation=dns_dissipation_ref,
            correction_mode="global",
            n_skip_steps=AVC_N_SKIP,
            exclude_diss_from_reward=True,
            simulation_mode="avc",
        )

        av_corrector_global = AVController(
            alpha_max=AVC_ALPHA_MAX * problem.viscosity,
            output_scale=1,
            n_wavenumber_bins=n_wavenumber_bins,
            correction_mode=avc_trainer_cfg.correction_mode,
            n_output_nodes=1,
        )
        paths.agents.mkdir(parents=True, exist_ok=True)
        save_corrector(av_corrector_global, paths.avcg_model)

        sac_config = SACConfig(
            n_skip_steps=AVC_N_SKIP,
            warmup_steps=500,
            batch_size=64 * 2,
        )

        environment_global = BurgersAVCEnvironment(
            problem=problem,
            disc_cfg=disc_cfg,
            sgsp_cfg=sgsp_cfg,
            avc_cfg=avc_trainer_cfg,
            master_path=paths.les_avc_data / "training_global",
            sac_config=sac_config,
            dns_reference_schedule=dns_reference_schedule,
        )
        sac_agent_global = SACAgent(
            av_corrector=av_corrector_global,
            state_dim=environment_global.state_dim,
            sac_config=sac_config,
        )
        trainer_global = OnlineAVTrainer(
            environment=environment_global,
            sac_agent=sac_agent_global,
            sac_config=sac_config,
            output_dir=paths.agents / "avcg_checkpoints",
        )
        trainer_global.train(n_episodes=AVC_EPOCHS)

    # --------------------------------------- GAVC run --------------------------------------- #
    avc_run_config = AVCRunConfig(avc_model_path=paths.avcg_model)
    solver_avc_global = BurgersAVC(
        problem,
        disc_cfg,
        "avc",
        paths.les_avc_data / "global",
        sgsp_cfg,
        avc_run_config,
    )
    solver_avc_global.run_simulation()
    solver_avc_global.post_processing()

    # -------------------------------------- Plotting --------------------------------------- #
    plot_solution_comparison(
        configs=create_velocity_plot_configs(paths, disc_cfg),
        output_path=paths.master,
        filename="comparison_dns_sgsp.png",
    )

    plot_energy_comparison(
        paths=paths,
        output_path=paths.master,
        viscosity=problem.viscosity,
        domain_length=problem.domain_length,
    )
