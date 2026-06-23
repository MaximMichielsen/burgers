"""Main execution file, runs all important blocks."""

from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from constants import RUNS_FOLDER
from ml.corrector_training.DNS_snapshot_converter import DNSReferenceSchedule
from ml.corrector_training.online_trainer import (
    SACConfig,
    BurgersAVCEnvironment,
    SACAgent,
    OnlineAVTrainer,
)
from ml.ml_agents.corrector import AVController, save_corrector
from pipeline_settings import PipelineConfig, RunPaths
from problems_and_configurations.disc_config import DiscretisationConfig
from problems_and_configurations.problems import Problems, Problem
from solvers.burgers_avc import BurgersAVC
from solvers.burgers_base import BurgersBase
from ml.ml_agents.solver_configs import SGSPConfig, AVCRunConfig, AVCTrainerConfig
from utils.enegy_evolution_utils import plot_energy_comparison
from utils.io_utils import (
    read_data,
    load_first_projected_solution,
)
from utils.pipeline_utils import (
    resolve_dns_caching,
    run_sgsp_training,
    run_sgsp_coupled_solver,
)
from utils.plot_utils import (
    plot_solution_comparison,
    SolutionConfig,
    is_viable_solution_path,
)


CURRENT_DIR = Path(__file__).parent.resolve()
matplotlib.use("Agg")  # needed when running on M12

# -------------------- Problem and pipeline configuration ------------------------------ #

problem: Problem = Problems.raj_one
problem = replace(problem, domain_timespan=2.0)

pipeline = PipelineConfig.all(manual_path=r"")
pipeline.clip_pusuluri = False
pipeline.clip_rajampeta = False

n_nodes_les: int = 9
temporal_refinement: int = 1
courant_les: float = 0.1

PROJECTION_MODE: str = "nodal"
ALPHA_MAX: float = 100 * problem.viscosity
OUTPUT_SCALE: float = 1
AVC_EPOCHS: int = 20
N_SKIP: int = 5

disc_cfg = DiscretisationConfig(
    n_nodes_les=n_nodes_les,
    temporal_refinement=temporal_refinement,
    courant_les=courant_les,
    domain_length=problem.domain_length,
)

master_path = CURRENT_DIR / RUNS_FOLDER / pipeline.get_run_id(problem_name=problem.name)
paths = RunPaths.from_master(master_path)
paths.create_master()


manual_load_model: str = r""
paths.model_output = (
    Path(manual_load_model) if manual_load_model != "" else paths.model_output
)

paths.avcg_model = master_path / "agents" / "av_global_corrector.pt"
paths.avcg_model.parent.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # --------------------------------------- DNS & SGSP data --------------------------------------- #
    DNS_CACHE_ROOT = CURRENT_DIR / "dns_cache"
    resolve_dns_caching(DNS_CACHE_ROOT, problem, disc_cfg, paths)

    # --------------------------------------- SGSP training --------------------------------------- #
    if not (paths.model_output / "sgs_predictor.pt").exists():
        run_sgsp_training(
            data_path=paths.training,
            output_dir=paths.model_output,
            domain_length=problem.domain_length,
            n_elements=n_nodes_les - 1,
        )

    # --------------------------------------- SGSP coupled solver --------------------------------------- #
    sgsp_cfg = SGSPConfig(
        sgsp_model_path=paths.model_output / "sgs_predictor.pt",
        normalization_path=paths.training,  # contains normalisation_stats.csv
        blown_up_path=paths.les_sgsp_data / "blown_up",
        clip_pusuluri=pipeline.clip_pusuluri,
        clip_rajampeta=pipeline.clip_rajampeta,
        set_off_predictor=False,
    )
    run_sgsp_coupled_solver(problem, disc_cfg, paths.les_sgsp_data, sgsp_cfg)

    les_run = BurgersBase(problem, disc_cfg, "les", paths.les_a_data)
    les_run.run_simulation()
    les_run.post_processing()

    dns_solution, _ = read_data(directory=paths.dns_data, final_only=True)
    projected_solution = np.interp(disc_cfg.mesh_les, disc_cfg.mesh_dns, dns_solution)

    # --------------------------------------- GAVC training --------------------------------------- #
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
    dns_dissipation_ref = float(dns_solver_ref.compute_dissipation(dns_solution_on_les))
    n_wavenumber_bins = len(dns_positive_spectrum)
    dns_reference_schedule = DNSReferenceSchedule.from_projection_directory(
        projection_dir=paths.projection,
        domain_length=problem.domain_length,
        viscosity=problem.viscosity,
        n_wavenumber_bins=n_wavenumber_bins,
    )

    sac_config = SACConfig(
        n_skip_steps=N_SKIP,
        warmup_steps=500,  # ~5 full episodes of random exploration
        batch_size=64 * 2,  # fill faster — fine for this problem size
    )
    avc_trainer_cfg = AVCTrainerConfig(
        paths.avcg_model,
        dns_energy_spectrum=dns_positive_spectrum,
        dns_dissipation=dns_dissipation_ref,
        correction_mode="global",
        n_skip_steps=sac_config.n_skip_steps,
        exclude_diss_from_reward=True,
        simulation_mode="avc",
    )

    av_corrector_global = AVController(
        alpha_max=ALPHA_MAX * problem.viscosity,
        output_scale=problem.viscosity * OUTPUT_SCALE,
        n_wavenumber_bins=n_wavenumber_bins,
        correction_mode=avc_trainer_cfg.correction_mode,
        n_output_nodes=1,
    )
    paths.model_output.mkdir(parents=True, exist_ok=True)
    save_corrector(av_corrector_global, paths.avcg_model)
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
        output_dir=paths.model_output / "avcg_checkpoints",
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

    global_avc_path = paths.les_avc_data / "global"

    plot_solution_comparison(
        configs=[
            SolutionConfig(
                data_path=paths.dns_data,
                label="DNS",
                color="gray",
                linestyle="-",
                marker="",
                alpha=0.7,
                mesh=disc_cfg.mesh_dns,
                solution=dns_solution,
            ),
            SolutionConfig(
                data_path=paths.dns_data,
                label="LES - projection",
                color="lightgreen",
                marker="x",
                mesh=disc_cfg.mesh_les,
                solution=projected_solution,
            ),
            SolutionConfig(
                data_path=paths.les_a_data,
                label="LES - A",
                color="tab:orange",
                marker="^",
                mesh=disc_cfg.mesh_les,
            ),
            SolutionConfig(
                data_path=paths.les_sgsp_data,
                label="LES - SGSP",
                color="crimson",
                marker="d",
                mesh=disc_cfg.mesh_les,
            ),
            SolutionConfig(
                data_path=global_avc_path,
                label="LES - AVC (global)",
                color="royalblue",
                linestyle="--",
                marker="s",
                mesh=disc_cfg.mesh_les,
            ),
        ],
        output_path=master_path,
        filename="comparison_dns_sgsp.png",
    )

    plot_energy_comparison(
        dns_dir=paths.dns_data,
        les_a_dir=paths.les_a_data,
        les_nm_dir=paths.les_nm_data
        if is_viable_solution_path(paths.les_nm_data)
        else None,
        les_sgsp_dir=paths.les_sgsp_data
        if is_viable_solution_path(paths.les_sgsp_data)
        else None,
        les_avcg_dir=global_avc_path
        if is_viable_solution_path(global_avc_path)
        else None,
        output_path=paths.master,
        viscosity=problem.viscosity,
        domain_length=problem.domain_length,
        projection_dir=None,
    )
