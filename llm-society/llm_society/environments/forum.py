"""A1. Public forum."""

from __future__ import annotations

from .base import Environment, OutgoingMessage, RoutingResult


class ForumEnvironment(Environment):
    """Every post is potentially visible to every other agent.

    Synchronous scheduler (default): all agents write from the forum state at
    the end of the previous round; posts are appended in a seeded random
    order and become visible next round.

    Asynchronous scheduler: agents post one at a time in a seeded random
    order, and later posters see earlier posts from the same round.

    With `exposure_matching.enabled`, each post reaches each other agent
    independently with probability posts_per_round / (N - 1).
    """

    name = "forum"
    framing = "forum"
    channel = "forum"

    def plan_round(self, round_id: int) -> list[list[str]]:
        if self.cfg.forum.scheduler == "asynchronous":
            order = self.ordering_rng.permutation(len(self.agent_ids))
            return [[self.agent_ids[i]] for i in order]
        return [list(self.agent_ids)]

    def route_messages(self, messages: list[OutgoingMessage], round_id: int) -> RoutingResult:
        if self.cfg.forum.scheduler == "synchronous" and len(messages) > 1:
            perm = self.ordering_rng.permutation(len(messages))
            messages = [messages[i] for i in perm]
        em = self.cfg.exposure_matching
        n_others = max(1, len(self.agent_ids) - 1)
        p = min(1.0, em.posts_per_round / n_others) if em.enabled else 1.0
        deliveries = []
        for m in messages:
            for a in self.agent_ids:
                if a == m.sender_id:
                    continue
                if p >= 1.0 or self.routing_rng.random() < p:
                    deliveries.append((a, m.message_id))
        return RoutingResult([m.message_id for m in messages], deliveries)

    def describe(self) -> dict:
        d = super().describe()
        d.update(scheduler=self.cfg.forum.scheduler, exposure_matching=self.cfg.exposure_matching.model_dump())
        return d
