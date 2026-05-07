"""Run to generate training data."""

from pathlib import Path
from typing import Callable

from fem.burgers import Burgers
from fem.data_generation.dns_data_generation import config as config_dns
from fem.data_generation.les_data_generation import configs as configs_les


def run_config(configuration: dict) -> None:
    """Run the set config."""
    solver = Burgers(configuration=configuration)
    solver.print_configuration()
    solver.run_simulation()
    solver.post_logging()





def main_(problem_definition: dict, run_dns: bool = True, run_les: bool = False, run_all_les: bool = False,
          ) -> str | Path | None:
    """Run the main script."""
    # -------------------------- DNS -------------------------- #
    dns_id = None
    if run_dns:
        config_dns_ = config_dns
        for k, v in problem_definition.items():
            config_dns_[k] = v
        solver_dns = Burgers(configuration=config_dns_)
        solver_dns.print_configuration()
        solver_dns.run_simulation()
        solver_dns.post_logging()
        dns_id = solver_dns.run_dir.name

    # -------------------------- LES -------------------------- #
    if run_les:
        if run_all_les:
            for config in configs_les:
                run_config(config)

        else:
            config = configs_les[0]
            run_config(config)

    return dns_id


if __name__ == "__main__":
    main_()
