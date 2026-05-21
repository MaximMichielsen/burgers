from pathlib import Path

from functions import plot_solutions_from_directory_animated

if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent.resolve() / "runs"

    master_dir = Path("run_robijns_one_0521_120437")

    solver_dir = "solver_data"
    specific_dir = "LES_ANN"

    directory = CURRENT_DIR / master_dir / solver_dir / specific_dir

    plot_solutions_from_directory_animated(directory=directory)
