"""Environment registry: canonical name -> class."""

from __future__ import annotations

from .base import Environment
from .dyadic import OpenDyadicEnvironment
from .forum import ForumEnvironment
from .network import NetworkDyadicEnvironment
from .no_interaction import NoInteractionEnvironment

ENVIRONMENTS: dict[str, type[Environment]] = {
    "forum": ForumEnvironment,
    "open_dyadic": OpenDyadicEnvironment,
    "erdos_renyi": NetworkDyadicEnvironment,
    "no_interaction": NoInteractionEnvironment,
}


def register_environment(name: str, cls: type[Environment]) -> None:
    ENVIRONMENTS[name] = cls


def get_environment_class(name: str) -> type[Environment]:
    try:
        return ENVIRONMENTS[name]
    except KeyError:
        raise ValueError(f"unknown environment {name!r}; known: {sorted(ENVIRONMENTS)}") from None
