"""Entry point: runs the full DNS → LES → projection → training → coupled-solver pipeline."""

from pathlib import Path

from replacing.constants import RUNS_FOLDER
from replacing.input_settings.disc_config import DiscretisationConfig
from replacing.input_settings.problems import Problems, Problem
from replacing.pipeline_settings import PipelineConfig, RunPaths
from replacing.solvers.burgers_base import BurgersBase

CURRENT_DIR = Path(__file__).parent.resolve()

problem: Problem = Problems.pipeline_test
pipeline = PipelineConfig.all_stages(manual_path="")

pipeline.clip_pusuluri = True
pipeline.clip_rajampeta = False
pipeline.debug_sgsp = True

disc_cfg = DiscretisationConfig(
    n_nodes_les=8,
    temporal_refinement=1,
    courant_les=0.01,
    domain_length=problem.domain_length,
    initial_condition_fn=problem.initial_condition,
)

ALPHA_MAX: float = 100
OUTPUT_SCALE: float = 10

master_path = CURRENT_DIR / RUNS_FOLDER / pipeline.get_run_id(problem_name=problem.name)
paths = RunPaths.from_master(master_path)
paths.create_master()

manual_load_dns = ""
paths.dns_data = Path(manual_load_dns) if manual_load_dns != "" else paths.dns_data

if pipeline.run_dns:
    solver_dns = BurgersBase(
        problem,
        disc_cfg,
        simulation_mode="dns",
        master_path=paths.dns_data,
        snapshot_factor=1,
    )
    solver_dns.run_simulation()
    solver_dns.post_processing()
