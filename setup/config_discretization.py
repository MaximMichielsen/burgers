"""Discretization dataclass for DNS/LES pipeline setup."""

from dataclasses import dataclass

import numpy as np

DNS_TO_LES_RATIO = 6


@dataclass
class DiscretizationConfig:
    """Spatial and temporal discretization parameters for both DNS and LES grids.

    Derives all mesh and time-step quantities from the LES element count,
    Courant number, and DNS-to-LES refinement ratios.
    Domain length only to be used internally for calculating element size,
    Setting this parameter for the simulation should follow from Problem!
    """

    n_nodes_les: int
    n_nodes_dns: int
    temporal_refinement: int
    domain_length: float
    domain_timespan: float

    courant_les: float | None = None
    set_dt_les: float | None = None
    suppress_file_logging: bool = False

    def __post_init__(self) -> None:
        self.n_elements_les: int = self.n_nodes_les - 1
        self.n_elements_dns: int = self.n_nodes_dns - 1

        self.h_les: float = self.domain_length / self.n_elements_les
        self.h_dns: float = self.domain_length / self.n_elements_dns

        self.dt_les: float = (
            self.set_dt_les
            if self.set_dt_les is not None
            else self.courant_les * self.h_les
        )
        self.dt_dns: float = self.dt_les / self.temporal_refinement

        self.mesh_les = np.linspace(0, self.domain_length, self.n_nodes_les)
        self.mesh_dns = np.linspace(0, self.domain_length, self.n_nodes_dns)

        self.n_wavenumber_bins: int = (self.n_nodes_les - 1) // 2

        self.n_timesteps = int(self.domain_timespan / self.dt_les)
