"""Scheduler for mean profile and instantaneous snapshots for reward signaling."""

from numpy.typing import NDArray


# TODO: adjust name to something better


class ReferenceSchedule:
    target_profile = str("bar")
    projected_solution: NDArray
    pass
