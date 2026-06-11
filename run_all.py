"""Run the full pipeline for all Rajampeta problems."""

from pathlib import Path

from problems_and_configurations.problems import Problems
from combined_runs.main import run_pipeline


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.resolve()

    run_pipeline(problem=Problems.raj_one, base_dir=PROJECT_ROOT)
    run_pipeline(problem=Problems.raj_two, base_dir=PROJECT_ROOT)
    run_pipeline(problem=Problems.raj_three, n_nodes_les=17, base_dir=PROJECT_ROOT)
