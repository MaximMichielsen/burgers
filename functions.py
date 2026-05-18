from burgers_pure import BurgersPure


def run_config(config: dict) -> None:
    """Run a given configuration file."""
    solver = BurgersPure(config)
    solver.run_simulation()
    solver.post_processing()
