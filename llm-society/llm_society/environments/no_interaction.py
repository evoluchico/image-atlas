"""Control: no interaction. Agents answer the question repeatedly and see only
their own previous contributions, never anyone else's."""

from __future__ import annotations

from .base import Environment, OutgoingMessage, RoutingResult


class NoInteractionEnvironment(Environment):
    name = "no_interaction"
    framing = "none"
    channel = "none"

    def route_messages(self, messages: list[OutgoingMessage], round_id: int) -> RoutingResult:
        return RoutingResult([m.message_id for m in messages], [])
