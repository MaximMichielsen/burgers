from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).parent.resolve()

PARSED_STAT_DIR = PROJECT_ROOT / "dns_data" / "curated" / "parsed_stat"

mean_w_profiles = np.load(PARSED_STAT_DIR / "stat_mean_w.npy")
mean_profile_1 = mean_w_profiles[0]

mesh_dns = np.linspace(0, 2, 513)

mean_profile_les_path = r'C:\Users\poopy\PycharmProjects\burgers\final\solver_data\les_33\mean_profiles\mean_w.csv'

plt.plot(mesh_dns, mean_profile_1)
plt.show()
