from functions import plot_solutions_from_directory_animated
from pathlib import Path

# __file__ is always the script's own path
_HERE = Path(__file__).parent  # → fem/examples/
_DATA = _HERE.parent / "data"  # → fem/data/

if __name__ == "__main__":
    data_folder = "run_DNS_0423_174914"
    plot_solutions_from_directory_animated(_DATA / data_folder)
