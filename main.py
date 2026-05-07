"""Main pipeline for thesis."""

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from numpy._typing import NDArray

from fem.agents.predictor import evaluate_test_performance, plot_training_diagnostics, train_and_diagnose
from fem.constants import N_NODES_DNS
from fem.data_generation.main_data import main_ as data_generation_main
from fem.projection_and_stencils.project import run_projection
from fem.projection_and_stencils.split_training_data import save_splits, split_data_shuffled, verify_splits

generate_data: bool = True
create_projection: bool = True
perform_split: bool = True
perform_training: bool = True

# This acts as a fallback if generate_data is False
run_id_directory = "run_dns_n512_0506_163212"

def create_problem_definition(forcing_type: str | Callable | None, reynolds: float, simulation_length: float, simulation_duration: float, boundary_conditions: str, initial_condition: NDArray) -> dict:
    """Create dictionary containing problem parameters."""
    viscosity = 1 * simulation_length / reynolds
    return {"time": simulation_duration,
            "length": simulation_length,
            "viscosity": viscosity,
            "forcing": forcing_type,
            "boundary_conditions": boundary_conditions,
            "solution_initial": initial_condition,
            }

if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()

    problem_robijns_one = create_problem_definition(forcing_type="uniform",
                                                    simulation_length=1,
                                                    simulation_duration=2,
                                                    reynolds=100,
                                                    boundary_conditions="fixed_one",
                                                    initial_condition=np.ones(N_NODES_DNS))

    # -----------------------   DNS   -----------------------#
    if generate_data:
        run_id_directory = data_generation_main(problem_definition=problem_robijns_one, run_dns=True, run_les=False, run_all_les=False)

    # ----------------------- stencils -----------------------#
    if create_projection:
        if current_dir.name == "fem":
            base_path = current_dir / "data" / "runs"
        else:
            base_path = current_dir / "fem" / "data" / "runs"
        target_path = base_path / run_id_directory

        if not target_path.exists():
            raise FileNotFoundError(f"Could not find DNS data at {target_path.absolute()}")

        X, y, stats = run_projection(str(target_path), save=True, output_dir="data/training_data/predictor")

    # -----------------------  split  -----------------------#
    if perform_split:
        projection_path = current_dir / "data" / "training_data" / "predictor" / run_id_directory / "pre_split"
        split_path = current_dir / "data" / "training_data" / "predictor" / run_id_directory / "post_split"

        if not create_projection:
            X, y = np.load(projection_path / "X.npy"), np.load(projection_path / "y.npy")

        splits = split_data_shuffled(x_input=X, y_target=y)
        save_splits(output_dir=split_path, splits=splits)
        verify_splits(split_path)

    # ------------------  predictor training  ------------------#
    if perform_training:
        # Define path to the post-split data
        split_path = current_dir / "data" / "training_data" / "predictor" / run_id_directory / "post_split"
        model_save_path = current_dir / "agents" / "predictor" / run_id_directory
        model_save_path.mkdir(parents=True, exist_ok=True)

        model, stats, test_data = train_and_diagnose(split_path)

        test_preds = evaluate_test_performance(model, test_data)

        torch.save(model.state_dict(), model_save_path / "sgs_mlp_model.pth")
        np.save(model_save_path / "training_history.npy", stats)
        print(f"Model saved to: {model_save_path / 'sgs_mlp_model.pth'}")

        plot_training_diagnostics(stats)
