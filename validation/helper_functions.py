import numpy as np
from numpy.typing import NDArray


def compute_u_prime_field(u_dns_on_les: NDArray, u_bar: NDArray) -> NDArray:
    """Unresolved field: u' = u_dns_projected - ū at every LES node."""
    return u_dns_on_les - u_bar


def compute_du_prime_dx(u_prime: NDArray, element_size: float) -> NDArray:
    """Approximate ∂u'/∂x via central differences (forward/backward at boundaries)."""
    du_dx = np.empty_like(u_prime)
    du_dx[1:-1] = (u_prime[2:] - u_prime[:-2]) / (2.0 * element_size)
    du_dx[0] = (u_prime[1] - u_prime[0]) / element_size
    du_dx[-1] = (u_prime[-1] - u_prime[-2]) / element_size
    return du_dx


def compute_du_prime_dt(
    u_prime_now: NDArray,
    u_prime_prev: NDArray | None,
    dt: float,
) -> NDArray:
    """Backward-Euler time derivative of u'; zero at first snapshot."""
    if u_prime_prev is None:
        return np.zeros_like(u_prime_now)
    return (u_prime_now - u_prime_prev) / dt
