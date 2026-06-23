"""Run DNS for three problems."""

from dataclasses import replace
from pathlib import Path

import matplotlib

from constants import RUNS_FOLDER
from pipeline_settings import PipelineConfig, RunPaths
from problems_and_configurations.disc_config import DiscretisationConfig
from problems_and_configurations.problems import Problems, Problem
from utils.pipeline_utils import (
    resolve_dns_caching,
)


CURRENT_DIR = Path(__file__).parent.resolve()
matplotlib.use("Agg")  # needed when running on M12

# -------------------- Problem and pipeline configuration ------------------------------ #

problem_one = replace(Problems.raj_one, domain_timespan=4.0, name="Raj_one")
problem_two = replace(Problems.raj_two, domain_timespan=12.0, name="Raj_two")
problem_three = replace(Problems.raj_three, domain_timespan=4.0, name="Raj_three")

pipeline = PipelineConfig.all(manual_path=r"")

n_nodes_les: int = 9
temporal_refinement: int = 1
courant_les: float = 0.1

disc_cfg = DiscretisationConfig(
    n_nodes_les=n_nodes_les,
    temporal_refinement=temporal_refinement,
    courant_les=courant_les,
    domain_length=1.0,
)

disc_cfg_3 = DiscretisationConfig(
    n_nodes_les=17,
    temporal_refinement=temporal_refinement,
    courant_les=courant_les,
    domain_length=1.0,
)
master_path_one = CURRENT_DIR / RUNS_FOLDER / pipeline.get_run_id(problem_name=problem_one.name)
paths_one = RunPaths.from_master(master_path_one)
paths_one.create_master()

master_path_two = CURRENT_DIR / RUNS_FOLDER / pipeline.get_run_id(problem_name=problem_two.name)
paths_two = RunPaths.from_master(master_path_two)
paths_two.create_master()

master_path_three = CURRENT_DIR / RUNS_FOLDER / pipeline.get_run_id(problem_name=problem_three.name)
paths_three = RunPaths.from_master(master_path_three)
paths_three.create_master()

if __name__ == "__main__":
    # --------------------------------------- DNS & SGSP data --------------------------------------- #
    DNS_CACHE_ROOT = CURRENT_DIR / "dns_cache"
    resolve_dns_caching(DNS_CACHE_ROOT, problem_one, disc_cfg, paths_one)

    resolve_dns_caching(DNS_CACHE_ROOT, problem_two, disc_cfg, paths_two)

    resolve_dns_caching(DNS_CACHE_ROOT, problem_three, disc_cfg, paths_three)
