"""A2. Unrestricted dyadic communication."""

from __future__ import annotations

from ..simulation.scheduler import get_selector
from .base import Environment, OutgoingMessage, RoutingResult


class OpenDyadicEnvironment(Environment):
    """Each agent sends one private message per round to a recipient chosen
    from the whole population (excluding itself). Messages are delivered at
    the end of the round and visible to the recipient from the next round."""

    name = "open_dyadic"
    framing = "dyadic"
    channel = "dyadic"

    def selection_rule(self) -> str:
        return self.cfg.dyadic.recipient_selection

    def candidates(self, agent_id: str) -> list[str]:
        return [a for a in self.agent_ids if a != agent_id]

    def select_recipients(self, round_id: int) -> dict[str, str | None]:
        select = get_selector(self.selection_rule())
        ctx = {"round": round_id}
        # Fixed agent order -> deterministic given the routing RNG state.
        return {a: select(a, self.candidates(a), self.routing_rng, ctx) for a in self.agent_ids}

    def route_messages(self, messages: list[OutgoingMessage], round_id: int) -> RoutingResult:
        deliveries = []
        for m in messages:
            if m.recipient_id is None or m.recipient_id == m.sender_id:
                raise ValueError(f"invalid dyadic recipient for {m.message_id}")
            deliveries.append((m.recipient_id, m.message_id))
        return RoutingResult([m.message_id for m in messages], deliveries)

    def describe(self) -> dict:
        d = super().describe()
        d["recipient_selection"] = self.selection_rule()
        return d
