"""Explicit agent memory and context-selection policies.

`AgentMemory` records everything an agent *has access to* (by message id).
A `MemoryPolicy` decides which of those messages are actually placed in the
prompt. The same policy is used for every environment, and the runner logs
the ids it selected, so truncation is always recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..config.schema import MemoryConfig


@dataclass
class AgentMemory:
    agent_id: str
    initial_profile: dict
    own_messages: list[str] = field(default_factory=list)
    received_messages: list[str] = field(default_factory=list)
    public_messages_seen: list[str] = field(default_factory=list)

    def add_own(self, message_id: str) -> None:
        self.own_messages.append(message_id)

    def receive(self, message_id: str, public: bool) -> None:
        (self.public_messages_seen if public else self.received_messages).append(message_id)

    def available(self) -> list[str]:
        return [*self.own_messages, *self.received_messages, *self.public_messages_seen]

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "initial_profile": self.initial_profile,
            "own_messages": self.own_messages,
            "received_messages": self.received_messages,
            "public_messages_seen": self.public_messages_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentMemory":
        return cls(**d)


class MemoryPolicy:
    """Select the messages to show an agent.

    `sequence_of` gives each message's global position in the run record;
    `cost_of` gives its token cost as rendered for this agent.

    * full: everything available.
    * recent_messages: the most recent `recent_messages` messages.
    * token_budget: the agent's `reserve_own_messages` most recent own
      messages first (if they fit), then the most recent remaining messages
      until `budget` tokens are used.
    Output is always in chronological order.
    """

    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg

    def select(
        self,
        memory: AgentMemory,
        sequence_of: Callable[[str], int],
        cost_of: Callable[[str], int],
        budget: int,
    ) -> list[str]:
        available = sorted(set(memory.available()), key=sequence_of)
        strategy = self.cfg.strategy
        if strategy == "full":
            return available
        if strategy == "recent_messages":
            return available[-self.cfg.recent_messages :] if self.cfg.recent_messages > 0 else []
        chosen: set[str] = set()
        used = 0
        own_recent = sorted(memory.own_messages, key=sequence_of)[-self.cfg.reserve_own_messages :] if self.cfg.reserve_own_messages > 0 else []
        for mid in reversed(own_recent):
            c = cost_of(mid)
            if used + c <= budget:
                chosen.add(mid)
                used += c
        for mid in reversed(available):
            if mid in chosen:
                continue
            c = cost_of(mid)
            if used + c > budget:
                break
            chosen.add(mid)
            used += c
        return [m for m in available if m in chosen]
