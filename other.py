from pathlib import Path

from functions import plot_multiple_solutions_animated

_HERE = Path(__file__).parent  # → fem/examples/
path_ = _HERE / "runs" / "run_psfslvlt_0512_152815" / "solver_data"
path_predict = path_ / "LES_ANN"
path_an = path_ / "LES_A"
path_nm = path_ / "LES_NM"
dirs = [path_predict, path_an, path_nm]
plot_multiple_solutions_animated(directories=dirs)
