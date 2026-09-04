"""Discretization dataclass for DNS/LES pipeline setup."""

from dataclasses import dataclass

import numpy as np

from old.constants import DNS_TO_LES_RATIO


# TODO: find n_wavenumber_bins usage and update to use disc_config parameters.


@dataclass
class DiscretizationConfig:
    """Spatial and temporal discretization parameters for both DNS and LES grids.

    Derives all mesh and time-step quantities from the LES element count,
    Courant number, and DNS-to-LES refinement ratios.
    Domain length only to be used internally for calculating element size,
    Setting this parameter for the simulation should follow from Problem!
    """

    n_nodes_les: int
    temporal_refinement: int
    courant_les: float
    domain_length: float
    suppress_file_logging: bool = False

    def __post_init__(self) -> None:
        self.n_nodes_dns: int = self.n_nodes_les * DNS_TO_LES_RATIO

        self.n_elements_les: int = self.n_nodes_les - 1
        self.n_elements_dns: int = self.n_nodes_dns - 1

        self.h_les: float = self.domain_length / self.n_elements_les
        self.h_dns: float = self.domain_length / self.n_elements_dns

        self.dt_les: float = self.courant_les * self.h_les
        self.dt_dns: float = self.dt_les / self.temporal_refinement

        self.mesh_les = np.linspace(0, self.domain_length, self.n_nodes_les)
        self.mesh_dns = np.linspace(0, self.domain_length, self.n_nodes_dns)

        self.n_wavenumber_bins: int = (self.n_nodes_les - 1) // 2
