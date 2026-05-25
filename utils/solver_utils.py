"""Utility functions for solver related aspects."""

from solvers.burgers_base import BurgersBase


def run_config(config: dict) -> None:
    """Instantiate and run a BurgersPure solver from a config dict."""
    solver = BurgersBase(config)
    solver.print_configuration()
    solver.run_simulation()
    solver.post_processing()
