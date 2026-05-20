from pathlib import Path

from functions import plot_solutions_from_directory_animated

if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent.resolve() / "runs"

    master_dir = Path("run_robijns_one_0520_141710")

    dir = "solver_data"
    dir_2 = "LES_ANN"

    directory = CURRENT_DIR / master_dir / dir / dir_2

    plot_solutions_from_directory_animated(directory=directory)
