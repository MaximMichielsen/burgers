"""Mesh and discretisation dataclasses for DNS/LES pipeline setup."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from constants import DNS_TO_LES_RATIO


@dataclass
class DiscretisationConfig:
    """Spatial and temporal discretization parameters derived from LES element count."""

    n_elements_les: int
    temporal_refinement: int
    courant_les: float
    domain_length: float

    def __post_init__(self) -> None:
        self.n_nodes_les: int = self.n_elements_les + 1
        self.n_elements_dns: int = self.n_elements_les * DNS_TO_LES_RATIO
        self.n_nodes_dns: int = self.n_elements_dns + 1
        self.h_les: float = self.domain_length / self.n_elements_les
        self.dt_les: float = self.courant_les * self.h_les
        self.dt_dns: float = self.dt_les / self.temporal_refinement


@dataclass
class MeshConfig:
    """Resolved grid, time-step, and initial condition for one resolution level."""

    n_nodes: int
    mesh: NDArray
    element_size: float
    time_step: float
    initial_solution: NDArray


def build_mesh_config(
    n_nodes: int,
    domain_length: float,
    time_step: float,
    initial_condition_fn: Callable,
) -> MeshConfig:
    """Build a MeshConfig from node count, domain length, time step, and IC function."""
    mesh, element_size = np.linspace(0, domain_length, n_nodes, retstep=True)
    return MeshConfig(
        n_nodes=n_nodes,
        mesh=mesh,
        element_size=float(element_size),
        time_step=time_step,
        initial_solution=initial_condition_fn(mesh),
    )
