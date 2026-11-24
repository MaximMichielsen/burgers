"""
Extrapolate alpha, beta and gamma from the supplied Reynolds-number data
to Re = 180.

Model choices:
  alpha, beta: power-law fits, y = a * Re**b (least-squares fit in log space)
  gamma:       constant fit (mean of the supplied values)

The target Re=180 is outside the data range (600--1200), so this is an
extrapolation and should be treated as model-dependent rather than validated.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------
Re = np.array([600, 700, 800, 900, 1000, 1100, 1200], dtype=float)

alpha = np.array([0.099, 0.091, 0.085, 0.082, 0.080, 0.078, 0.077])
beta = np.array([9.39, 9.43, 9.59, 9.83, 10.14, 10.48, 10.86])
gamma = np.array([2.16, 2.16, 2.16, 2.16, 2.16, 2.16, 2.17])

Re_target = 180.0


# ---------------------------------------------------------------------
# Fitting functions
# ---------------------------------------------------------------------
def fit_power_law(x, y):
    """Fit y = a*x**b by linear regression in log-log space."""
    b, log_a = np.polyfit(np.log(x), np.log(y), 1)
    a = np.exp(log_a)
    return a, b


def evaluate_power_law(x, a, b):
    return a * x**b


# Fit alpha and beta with power laws.
alpha_a, alpha_b = fit_power_law(Re, alpha)
beta_a, beta_b = fit_power_law(Re, beta)

# Gamma varies by only 0.01 across the supplied data. Treat it as constant.
gamma_fit = np.mean(gamma)

alpha_target = evaluate_power_law(Re_target, alpha_a, alpha_b)
beta_target = evaluate_power_law(Re_target, beta_a, beta_b)
gamma_target = gamma_fit


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
print("Power-law extrapolation to Re =", int(Re_target))
print("-" * 52)
print(f"alpha(Re) = {alpha_a:.8g} * Re^({alpha_b:.8g})")
print(f"beta(Re)  = {beta_a:.8g} * Re^({beta_b:.8g})")
print(f"gamma(Re) = {gamma_fit:.8f}  [constant mean]")
print()
print(f"alpha({Re_target:.0f}) = {alpha_target:.8f}")
print(f"beta({Re_target:.0f})  = {beta_target:.8f}")
print(f"gamma({Re_target:.0f}) = {gamma_target:.8f}")


# ---------------------------------------------------------------------
# Plot data and fitted curves
# ---------------------------------------------------------------------
Re_plot = np.linspace(Re_target, Re.max(), 400)

alpha_plot = evaluate_power_law(Re_plot, alpha_a, alpha_b)
beta_plot = evaluate_power_law(Re_plot, beta_a, beta_b)
gamma_plot = np.full_like(Re_plot, gamma_fit)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

plot_data = [
    (axes[0], alpha, alpha_plot, alpha_target, r"$\alpha$", "tab:blue"),
    (axes[1], beta, beta_plot, beta_target, r"$\beta$", "tab:orange"),
    (axes[2], gamma, gamma_plot, gamma_target, r"$\gamma$", "tab:green"),
]

for ax, values, fitted, target_value, label, color in plot_data:
    ax.scatter(Re, values, color=color, label="Input data", zorder=3)
    ax.plot(Re_plot, fitted, color=color, label="Fitted model")
    ax.scatter(
        [Re_target],
        [target_value],
        marker="x",
        s=90,
        linewidths=2,
        color="black",
        label=f"Extrapolated Re={int(Re_target)}",
        zorder=4,
    )
    ax.axvline(Re_target, color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$Re$")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("Extrapolation of model coefficients to Re = 180")
fig.tight_layout()
plt.show()
