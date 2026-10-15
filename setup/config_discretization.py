"""Discretization dataclass for DNS/LES pipeline setup."""

from dataclasses import dataclass
from math import ceil

import numpy as np

from setup.constants import DT_DNS, N_NODES_DNS
from setup.problems import Problem


@dataclass
class DiscretizationConfig:
    """Spatial and temporal discretization parameters for both DNS and LES grids."""

    problem: Problem
    n_nodes_les: int
    dt_les: float

    n_nodes_dns: int = N_NODES_DNS
    suppress_file_logging: bool = False

    def __post_init__(self) -> None:
        self.n_elements_les: int = self.n_nodes_les - 1
        self.n_elements_dns: int = self.n_nodes_dns - 1

        self.h_les: float = self.problem.domain_length / self.n_elements_les
        self.h_dns: float = self.problem.domain_length / self.n_elements_dns

        self.courant_les = self.dt_les / self.h_les
        self.dt_dns: float = DT_DNS

        self.mesh_les = np.linspace(0, self.problem.domain_length, self.n_nodes_les)
        self.mesh_dns = np.linspace(0, self.problem.domain_length, self.n_nodes_dns)

        self.n_wavenumber_bins: int = (self.n_nodes_les - 1) // 2
        self.n_timesteps = ceil(self.problem.domain_timespan / self.dt_les)
