"""Communication environment interface.

An environment decides *who can see what, when*. It never generates text and
knows nothing about prompts or LLMs. Each round the runner asks it:

1. `select_recipients(round)` - addressee for each agent (None = public/none);
2. `plan_round(round)` - batches of agents that generate from the same
   snapshot (one batch = fully synchronous; one agent per batch = sequential);
3. `route_messages(messages, round)` - after a batch, the order in which the
   batch's messages enter the record and which agents receive each message.

All randomness comes from generators created from the run's derived seeds,
and generator state is part of `state_dict()` for checkpointing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..config.schema import ExperimentConfig
from ..seeds import restore_rng, rng_state


@dataclass
class OutgoingMessage:
    message_id: str
    round: int
    sender_id: str
    recipient_id: str | None
    text: str


@dataclass
class RoutingResult:
    # Message IDs in the order they enter the public record / delivery queue.
    ordered_ids: list[str]
    # (recipient agent id, message id) pairs; a message may go to many agents.
    deliveries: list[tuple[str, str]] = field(default_factory=list)


class Environment(ABC):
    name: str = "base"
    framing: str = "none"  # which context template agents see: forum | dyadic | none
    channel: str = "none"  # stored in the messages table

    def __init__(self, cfg: ExperimentConfig, routing_rng: np.random.Generator, ordering_rng: np.random.Generator):
        self.cfg = cfg
        self.routing_rng = routing_rng
        self.ordering_rng = ordering_rng
        self.agent_ids: list[str] = []

    def initialise(self, agent_ids: list[str], network_rng: np.random.Generator | None = None) -> None:
        self.agent_ids = list(agent_ids)

    def select_recipients(self, round_id: int) -> dict[str, str | None]:
        return {a: None for a in self.agent_ids}

    def plan_round(self, round_id: int) -> list[list[str]]:
        return [list(self.agent_ids)]

    @abstractmethod
    def route_messages(self, messages: list[OutgoingMessage], round_id: int) -> RoutingResult: ...

    def network_edges(self) -> list[tuple[str, str]] | None:
        return None

    def describe(self) -> dict:
        return {"name": self.name, "framing": self.framing, "channel": self.channel}

    # ---- checkpointing ----------------------------------------------
    def state_dict(self) -> dict:
        return {
            "agent_ids": self.agent_ids,
            "routing_rng": rng_state(self.routing_rng),
            "ordering_rng": rng_state(self.ordering_rng),
        }

    def load_state_dict(self, state: dict) -> None:
        self.agent_ids = list(state["agent_ids"])
        self.routing_rng = restore_rng(state["routing_rng"])
        self.ordering_rng = restore_rng(state["ordering_rng"])
