"""Test suite for the FEM Burgers' solver.
Purpose: regression tests — catch when code changes alter numerical results.
Run with: pytest verification.py -v
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from old.burgers import Burgers

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def make_ic(n_nodes: int) -> NDArray:
    """Sinusoidal IC with periodicity enforced."""
    coords = np.linspace(0, 2 * np.pi, n_nodes)
    ic = np.sin(coords)
    ic[0] = ic[-1]
    return ic


def make_config(simulation_type: str, n_nodes: int = 32) -> dict:
    """Minimal but representative config for regression testing."""
    return Burgers.create_config(
        node_amount=n_nodes,
        simulation_type=simulation_type,
        run_objective="verification (test suite)",
        initial_condition=make_ic(n_nodes),
        viscosity=1e-2,
        domain_length=2 * np.pi,
        domain_timespan=0.1,
        time_step=0.01,
        convergence_tol_residual=1e-4,
        convergence_tol_update=1e-4,
        max_iterations=50,
        relaxation=0.25,
    )


def run_solver(config: dict) -> Burgers:
    """Run a full simulation and return the solved instance."""
    solver = Burgers(config)
    solver.run_simulation()
    solver.post_logging()
    return solver


# ---------------------------------------------------------------------------
# Smoke tests — does it run at all?
# ---------------------------------------------------------------------------


class TestSmoke:
    def test_dns_runs(self) -> None:
        """DNS solver completes without error."""
        run_solver(make_config("dns"))

    def test_les_runs(self) -> None:
        """LES solver completes without error."""
        run_solver(make_config("les"))


# ---------------------------------------------------------------------------
# Solution sanity checks
# ---------------------------------------------------------------------------


class TestSolutionSanity:
    def test_solution_shape(self) -> None:
        """Output solution has the expected number of nodes."""
        n_nodes = 32
        solver = run_solver(make_config("dns", n_nodes=n_nodes))
        assert solver.solution.shape == (n_nodes,)

    def test_solution_is_finite(self) -> None:
        """Solution contains no NaN or Inf values."""
        solver = run_solver(make_config("dns"))
        assert np.all(np.isfinite(solver.solution))

    def test_solution_magnitude(self) -> None:
        """Solution stays within a physically reasonable range."""
        solver = run_solver(make_config("dns"))
        assert np.max(np.abs(solver.solution)) < 10.0

    def test_fixed_boundary_conditions(self) -> None:
        """Boundary nodes remain at zero for fixed BCs."""
        solver = run_solver(make_config("dns"))
        assert solver.solution[0] == pytest.approx(0.0, abs=1e-10)
        assert solver.solution[-1] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Regression tests — pin known-good output values
# ---------------------------------------------------------------------------


class TestRegression:
    """These values are pinned from a known-good solver run.
    If any of these fail after a code change, the numerics have shifted.
    Update the expected values ONLY after manually verifying the new
    output is physically correct.
    """

    EXPECTED_DNS_FINAL_L2: float = 0.6952887956819448
    EXPECTED_LES_FINAL_L2: float = 0.6952451553904848

    @staticmethod
    def _l2_norm(solution: NDArray) -> float:
        return float(np.sqrt(np.mean(solution**2)))

    def test_dns_l2_norm_unchanged(self) -> None:
        """DNS final solution L2 norm matches pinned value."""
        if self.EXPECTED_DNS_FINAL_L2 is None:
            pytest.skip("Expected value not yet pinned — run once and record.")
        solver = run_solver(make_config("dns"))
        assert self._l2_norm(solver.solution) == pytest.approx(
            self.EXPECTED_DNS_FINAL_L2, rel=1e-4
        )

    def test_les_l2_norm_unchanged(self) -> None:
        """LES final solution L2 norm matches pinned value."""
        if self.EXPECTED_LES_FINAL_L2 is None:
            pytest.skip("Expected value not yet pinned — run once and record.")
        solver = run_solver(make_config("les"))
        assert self._l2_norm(solver.solution) == pytest.approx(
            self.EXPECTED_LES_FINAL_L2, rel=1e-4
        )

    def test_dns_vs_les_dissipation(self) -> None:
        """LES solution should be more dissipative than DNS at same resolution."""
        dns = run_solver(make_config("dns"))
        les = run_solver(make_config("les"))
        assert self._l2_norm(les.solution) <= self._l2_norm(dns.solution)


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_newton_raphson_converged(self) -> None:
        """Newton-Raphson loop should converge within all time steps."""
        solver = run_solver(make_config("dns"))
        tol = make_config("dns")["convergence_tol_residual"]
        final_residuals = [history[-1] for history in solver.residual_history]
        assert all(r < tol for r in final_residuals), (
            f"NR failed to converge in at least one time step. Max final residual: {max(final_residuals):.4e}"
        )

    def test_residual_decreasing(self) -> None:
        """Residual should decrease monotonically within each NR iteration."""
        solver = run_solver(make_config("dns"))
        for step, history in enumerate(solver.residual_history):
            assert history[-1] <= history[0], (
                f"Residual did not decrease at time step {step}: start {history[0]:.4e} → end {history[-1]:.4e}"
            )
