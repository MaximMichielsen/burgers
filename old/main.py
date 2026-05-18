"""Main pipeline for thesis."""

import json
from pathlib import Path

import numpy as np
import torch

import logging

from matplotlib import pyplot as plt

from old.agents.predictor import (
    evaluate_test_performance,
    plot_training_diagnostics,
    train_and_diagnose,
)
from constants import (
    RUNS_FOLDER,
    SOLVER_DATA_FOLDER,
    PREDICTOR_FOLDER,
    PRE_SPLIT_FOLDER,
    POST_SPLIT_FOLDER,
    AGENTS_FOLDER,
    INPUT_STENCIL,
    OUTPUT_STENCIL,
    TRAINING_DATA_FOLDER,
    NORM_STATS,
    DNS_SAVE_PATH,
    LES_ANALYTICAL_SAVE_PATH,
    LES_NO_MODEL_SAVE_PATH,
    LES_ANN_SAVE_PATH,
)
from data_generation.configurations import (
    create_code_test_config,
    create_solver_configs,
)
from functions import run_config, read_data, SolutionConfig, plot_solution_comparison
from problem_domains.problems import problem_robijns_one
from projection_and_stencils.project_2 import run_projection
from projection_and_stencils.split_training_data import (
    save_splits,
    split_data_shuffled,
    verify_splits,
)
from burgers import Burgers
from old.agents.predictor import SGSPredictor

import datetime

logging.getLogger().setLevel(logging.DEBUG)  # see ANN SGS channel norms


# ── Pipeline flags ────────────────────────────────────────────────────────────
test_pipeline: bool = False
set_manual_run: bool = False
_manual_run_id = "run_None_0516_155128"
set_ann_manually: bool = False

generate_data_dns: bool = True
run_les_models: bool = True
run_analytical = True
run_no_model = True

create_projection: bool = True
perform_split: bool = True
perform_training: bool = True

show_ann_diagnostics: bool = False

run_predictor_model: bool = True

compute_energy_evolution: bool = True
compare_solvers: bool = True

# ── Paths ─────────────────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).parent.resolve()


if __name__ == "__main__":
    problem: dict = problem_robijns_one

    # ── Master Path Definition ───────────────────────────────────────────────
    if generate_data_dns or run_les_models:
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        master_run_id = f"run_{problem['name']}_{timestamp}"
        master_path = CURRENT_DIR / RUNS_FOLDER / master_run_id

    elif set_manual_run:
        # Update this string to the specific folder
        master_path = CURRENT_DIR / RUNS_FOLDER / _manual_run_id

    else:
        with open("../runs/latest_run_parameters.json") as f:
            latest_run_details = json.load(f)
        latest_run_id = latest_run_details["run_id"]
        master_path = Path(latest_run_details["master_path"])

    master_path.mkdir(parents=True, exist_ok=True)

    # ── Derived paths  ───────────────────────────────────────────────────
    training_data_path = master_path / TRAINING_DATA_FOLDER
    pre_split_path = training_data_path / PRE_SPLIT_FOLDER
    post_split_path = training_data_path / POST_SPLIT_FOLDER
    model_save_path = master_path / AGENTS_FOLDER / PREDICTOR_FOLDER
    predictor_data_path = master_path / SOLVER_DATA_FOLDER / LES_ANN_SAVE_PATH

    # ── DNS data generation ──────────────────────────────────────────────
    if generate_data_dns or run_les_models:
        if not test_pipeline:
            config_dns, config_les_analytical, config_les_no_model = (
                create_solver_configs(
                    problem_definition=problem, master_dir=master_path
                )
            )
        else:
            config_dns = create_code_test_config()

    if generate_data_dns:
        solver_data_path_dns = run_config(
            config_dns, save_path=f"{SOLVER_DATA_FOLDER}/{DNS_SAVE_PATH}"
        )
    else:
        solver_data_path_dns = master_path / SOLVER_DATA_FOLDER / DNS_SAVE_PATH
        solver_data_path_les_analytical = (
            master_path / SOLVER_DATA_FOLDER / LES_ANALYTICAL_SAVE_PATH
        )
        solver_data_path_les_no_model = (
            master_path / SOLVER_DATA_FOLDER / LES_NO_MODEL_SAVE_PATH
        )

    solver_data_path_dns = Path(solver_data_path_dns)

    # ── LES data generation ──────────────────────────────────────────────
    if run_les_models:
        if run_analytical:
            solver_data_path_les_analytical = run_config(
                config_les_analytical,
                save_path=f"{SOLVER_DATA_FOLDER}/{LES_ANALYTICAL_SAVE_PATH}",
            )
        if run_no_model:
            solver_data_path_les_no_model = run_config(
                config_les_no_model,
                save_path=f"{SOLVER_DATA_FOLDER}/{LES_NO_MODEL_SAVE_PATH}",
            )

    # ── Projection / stencils ─────────────────────────────────────────────────
    if create_projection:
        pre_split_path.mkdir(parents=True, exist_ok=True)
        if not solver_data_path_dns.exists():
            raise FileNotFoundError(f"DNS data not found at: {solver_data_path_dns}")

        X, y, stats, projected_solution = run_projection(
            directory=solver_data_path_dns,
            bc_mode=problem["boundary_condition_type"],
            bc_values=problem["boundary_condition_value"],
            save=True,
            output_dir=pre_split_path,
            verify=False,
        )

    # ── Train/test split ──────────────────────────────────────────────────────
    if perform_split:
        post_split_path.mkdir(parents=True, exist_ok=True)
        if not create_projection:
            X = np.load(pre_split_path / INPUT_STENCIL)
            y = np.load(pre_split_path / OUTPUT_STENCIL)

        splits = split_data_shuffled(x_input=X, y_target=y)
        save_splits(output_dir=post_split_path, splits=splits)
        verify_splits(post_split_path)

    # ── Predictor training ────────────────────────────────────────────────────
    if perform_training:
        model_save_path.mkdir(parents=True, exist_ok=True)

        model, train_stats, test_data = train_and_diagnose(post_split_path)
        test_preds = evaluate_test_performance(
            model, test_data, output_dir=model_save_path
        )

        torch.save(model.state_dict(), model_save_path / "sgs_mlp_model.pth")

        import shutil

        stats_src = pre_split_path / (Path(NORM_STATS).stem + ".npz")
        shutil.copy(stats_src, model_save_path / stats_src.name)

        np.save(model_save_path / "training_history.npy", train_stats)

        plot_training_diagnostics(
            model_save_path, train_stats, show_fig=show_ann_diagnostics
        )

    # ── Predictor Run ──────────────────────────────────────────────────────────
    if run_predictor_model:
        predictor_data_path.mkdir(parents=True, exist_ok=True)
        config_predictor = create_solver_configs(
            problem_definition=problem,
            master_dir=master_path,
            create_predictor_config=True,
        )

        config_predictor["save_path"] = f"{SOLVER_DATA_FOLDER}/{LES_ANN_SAVE_PATH}"

        if set_ann_manually:
            run_path = ""
            model_save_path = (
                CURRENT_DIR / RUNS_FOLDER / run_path / AGENTS_FOLDER / PREDICTOR_FOLDER
            )
        elif not perform_training:
            raise ValueError(
                "set_ann_manually is False and perform_training is False — "
                "no model was trained this run and no manual path was given. "
                "Set set_ann_manually=True and provide a run path."
            )

        solver_predictor = Burgers(
            config_predictor,
            ann_model_path=model_save_path,
            ann_model_class=SGSPredictor,
        )

        # ── Run the simulation directly on the coupled solver ──
        solver_predictor.print_configuration()
        solver_predictor.run_simulation()
        solver_predictor.post_logging()

        predictor_data_path = solver_predictor.save_path_dir

    # ── Compare models ────────────────────────────────────────────────────────
    if compare_solvers:
        dns_solution, mesh_dns = read_data(
            directory=solver_data_path_dns, final_only=True
        )
        _, mesh_les = read_data(solver_data_path_les_analytical, final_only=True)

        configs = [
            SolutionConfig(
                data_path=solver_data_path_dns,
                label="DNS",
                color="gray",
                linestyle="-",
                marker="",  # no marker for the reference curve
                alpha=0.7,
                mesh=mesh_dns,
                solution=dns_solution,
            ),
            SolutionConfig(
                data_path=solver_data_path_les_analytical,
                label="LES - A",
                color="royalblue",
                marker="x",
                mesh=mesh_les,
            ),
            SolutionConfig(
                data_path=predictor_data_path,
                label="LES - ANN",
                color="salmon",
                marker="d",
                mesh=mesh_les,
            ),
        ]

        if create_projection:
            projection_config = SolutionConfig(
                data_path=solver_data_path_dns,  # unused when mesh/solution are provided
                label="LES - projection",
                color="lightgreen",
                marker="^",
                mesh=mesh_les,
                solution=projected_solution,
            )
            configs.append(projection_config)

        if run_no_model:
            no_model_config = SolutionConfig(
                data_path=solver_data_path_les_no_model,
                label="LES - no model",
                color="tab:orange",
                marker=".",
                mesh=mesh_les,
            )
            configs.append(no_model_config)

        fig, ax = plot_solution_comparison(configs, output_path=master_path)

    # ── Energy evolution ────────────────────────────────────────────────────────
    if compute_energy_evolution:
        # mesh_dns, times_dns, solutions_dns, _ = read_data(directory=solver_data_path_dns, final_only=False)
        mesh_les, times_les, solutions_les, _ = read_data(
            directory=solver_data_path_les_analytical, final_only=False
        )
        mesh_ann, times_ann, solutions_ann, _ = read_data(
            directory=predictor_data_path, final_only=False
        )

        x_coords_les = mesh_les
        # x_coords_dns = mesh_dns
        x_coords_ann = mesh_ann

        dns_energy_evolution = [
            np.trapezoid(0.5 * u**2, x_coords_les) for u in solutions_les
        ]
        ann_energy_evolution = [
            np.trapezoid(0.5 * u**2, x_coords_ann) for u in solutions_ann
        ]

        plt.plot(times_les, dns_energy_evolution, label="LES")
        plt.plot(times_ann, ann_energy_evolution, label="ANN")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.legend()
        plt.show()

    # ── Latest run writing ──────────────────────────────────────────────────────
    # Write to a json for simple run_id path housekeeping.
    if not generate_data_dns or not set_manual_run:
        print("Writing run paths to the latest_run_parameters.json file.")
        latest_run_config = {
            "master_path": str(master_path),
            "run_id": str(latest_run_id),  # was master_run_id
        }
        with open("../runs/latest_run_parameters.json", "w") as f:
            json.dump(latest_run_config, f, indent=2)
