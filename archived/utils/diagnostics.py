import numpy as np
from numpy.typing import NDArray


def compute_energy(solution: NDArray, domain_length: float) -> float:
    """Compute exact energy ½∫u² dx for piecewise linear elements on a uniform grid."""
    n_elements = len(solution) - 1
    dx = domain_length / n_elements
    u_left = solution[:-1]
    u_right = solution[1:]

    # Exact element integral of (u_left*N1 + u_right*N2)^2 is dx/3 * (u_l^2 + u_l*u_r + u_r^2)
    element_energies = (dx / 6.0) * (u_left**2 + u_left * u_right + u_right**2)
    return float(np.sum(element_energies))


def compute_dissipation(
    solution: NDArray, domain_length: float, viscosity: float
) -> float:
    """Compute exact viscous dissipation ν∫(∂u/∂x)² dx for piecewise linear elements."""
    n_elements = len(solution) - 1
    dx = domain_length / n_elements
    du_dx = (solution[1:] - solution[:-1]) / dx

    return float(viscosity * dx * np.sum(du_dx**2))


def compute_energy_spectrum(
    solution: NDArray, domain_length: float, positive_only: bool = True
) -> tuple[NDArray, NDArray]:
    """Return positive wavenumbers and spectral energy distribution E(k)."""
    n_points = len(solution)
    u_hat = np.fft.fft(solution)
    wavenumbers = np.fft.fftfreq(n_points, d=domain_length / n_points) * 2 * np.pi
    spectrum = 0.5 * np.abs(u_hat) ** 2 / n_points

    if positive_only:
        mask = wavenumbers > 0
        return wavenumbers[mask], spectrum[mask]

    return wavenumbers, spectrum
