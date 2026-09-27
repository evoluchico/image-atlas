"""Recipient selection rules for dyadic communication.

A selector picks one recipient for a sender from a candidate list supplied by
the environment (everyone else for open dyadic, graph neighbours for the
network condition). New rules (homophily, reply probability, persistent
partners, agent-selected recipients, ...) are added by registering a function;
the environments do not change.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class RecipientSelector(Protocol):
    def __call__(
        self, sender_id: str, candidates: list[str], rng: np.random.Generator, context: dict
    ) -> str: ...


RECIPIENT_SELECTORS: dict[str, RecipientSelector] = {}


def register_selector(name: str) -> Callable[[RecipientSelector], RecipientSelector]:
    def deco(fn: RecipientSelector) -> RecipientSelector:
        RECIPIENT_SELECTORS[name] = fn
        return fn

    return deco


def get_selector(name: str) -> RecipientSelector:
    try:
        return RECIPIENT_SELECTORS[name]
    except KeyError:
        raise ValueError(f"unknown recipient selection rule {name!r}; known: {sorted(RECIPIENT_SELECTORS)}") from None


@register_selector("uniform_random")
@register_selector("uniform_random_neighbor")
def uniform_random(sender_id: str, candidates: list[str], rng: np.random.Generator, context: dict) -> str:
    if not candidates:
        raise ValueError(f"{sender_id} has no eligible recipients")
    return candidates[int(rng.integers(len(candidates)))]
