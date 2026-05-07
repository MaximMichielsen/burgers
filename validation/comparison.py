"""Plot the runs of different configurations (DNS -> coarse LES)."""

import pandas as pd
from matplotlib import pyplot as plt

coarse_data = pd.read_csv("../data/LES_COARSE_DATA_RE_180")
solid_data = pd.read_csv("../data/LES_SOLID_DATA_RE_180")
fine_data = pd.read_csv("../data/LES_FINE_DATA_RE_180")
dns_data = pd.read_csv("../data/DNS_DATA_RE_180")

fig, ax = plt.subplots()
ax.plot(coarse_data["x_coordinate"], coarse_data["velocity"], marker="x", label="coarse")
ax.plot(solid_data["x_coordinate"], solid_data["velocity"], marker="x", label="solid")
ax.plot(fine_data["x_coordinate"], fine_data["velocity"], marker="x", label="fine")
ax.plot(dns_data["x_coordinate"], dns_data["velocity"], marker="x", label="dns")

ax.legend()
ax.set_title("comparison solver runs")
ax.set_ylabel("velocity")
ax.set_xlabel("x_coordinate")
ax.grid(True)

plt.show()
