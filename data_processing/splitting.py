"""Algorithm for splitting DNS data into training and evaluation sets."""

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CURATED_DIR = PROJECT_ROOT / "dns_data" / "curated"
PROJECTED_DIR = PROJECT_ROOT / "dns_data" / "projected"
TRAINING_DIR = PROJECT_ROOT / "dns_data" / "training"

nodes_les = [9, 17, 33, 65]
time_cutoff = 313.01
for n_les in nodes_les:
    path_to_w = PROJECTED_DIR / f"w_linear_{n_les}.npy"
    path_to_mean_w = CURATED_DIR / "stat" / "mean_w.npy"
    path_to_times_dat = CURATED_DIR / "dat" / "times.npy"
    path_to_times_stat = CURATED_DIR / "stat" / "times.npy"

    path_n_dir = TRAINING_DIR / f"n{n_les}"

    path_to_training_dir = path_n_dir / "training"
    path_to_eval_dir = path_n_dir / "evaluation"

    path_to_training_dir.mkdir(exist_ok=True, parents=True)
    path_to_eval_dir.mkdir(exist_ok=True, parents=True)

    w_field_data = np.load(path_to_w)
    mean_w_data = np.load(path_to_mean_w)

    times_dat = np.load(path_to_times_dat)
    times_stat = np.load(path_to_times_stat)

    index_dat = np.argmax(times_dat == time_cutoff)
    index_stat = np.argmax(times_stat == time_cutoff)

    times_training = times_dat[:index_dat]
    times_validation = times_dat[index_dat:]

    training_fraction = len(times_training) / len(times_dat)
    validation_fraction = len(times_validation) / len(times_dat)

    w_training = w_field_data[:index_dat]
    w_validation = w_field_data[index_dat:]

    mean_w_training = mean_w_data[:index_stat]
    mean_w_evaluation = mean_w_data[index_stat:]

    np.save(path_to_training_dir / f"w_field_{n_les}.npy", w_training)
    np.save(path_to_training_dir / "stat_mean_w.npy", mean_w_training)
    np.save(path_to_training_dir / "training_times.npy", times_training)

    np.save(path_to_eval_dir / f"w_field_{n_les}.npy", w_validation)
    np.save(path_to_eval_dir / "stat_mean_w.npy", mean_w_evaluation)
    np.save(path_to_eval_dir / "evaluation_times.npy", times_validation)

# TODO: Reshifting of mean w evaluation
