"""A3. Network-constrained dyadic communication on an Erdős–Rényi graph."""

from __future__ import annotations

import networkx as nx
import numpy as np

from ..config.schema import NetworkConfig
from .base import OutgoingMessage, RoutingResult
from .dyadic import OpenDyadicEnvironment


def edge_probability(n: int, target_mean_degree: float) -> float:
    """p = k / (N - 1), so expected mean degree is k for every N."""
    return min(1.0, target_mean_degree / (n - 1))


def generate_er_graph(n: int, cfg: NetworkConfig, rng: np.random.Generator) -> tuple[nx.Graph, int]:
    """Sample G(n, p); if required, resample until connected. Returns (graph, attempts)."""
    p = edge_probability(n, cfg.target_mean_degree)
    for attempt in range(1, cfg.max_resample_attempts + 1):
        g = nx.gnp_random_graph(n, p, seed=int(rng.integers(2**32 - 1)))
        if not cfg.require_connected or nx.is_connected(g):
            return g, attempt
    raise RuntimeError(
        f"no connected G({n}, {p:.4f}) after {cfg.max_resample_attempts} attempts; increase target_mean_degree"
    )


class NetworkDyadicEnvironment(OpenDyadicEnvironment):
    """As open dyadic, but agents can only message graph neighbours.
    The graph is fixed for the whole run."""

    name = "erdos_renyi"
    framing = "dyadic"
    channel = "network"

    def initialise(self, agent_ids: list[str], network_rng: np.random.Generator | None = None) -> None:
        super().initialise(agent_ids)
        if network_rng is None:
            raise ValueError("network environment needs a network RNG")
        g, attempts = generate_er_graph(len(agent_ids), self.cfg.network, network_rng)
        self.resample_attempts = attempts
        self._set_edges([(agent_ids[u], agent_ids[v]) for u, v in sorted(g.edges())])

    def _set_edges(self, edges: list[tuple[str, str]]) -> None:
        self.edges = [tuple(e) for e in edges]
        self.neighbours: dict[str, list[str]] = {a: [] for a in self.agent_ids}
        for u, v in self.edges:
            self.neighbours[u].append(v)
            self.neighbours[v].append(u)
        for a in self.neighbours:
            self.neighbours[a].sort()

    def selection_rule(self) -> str:
        return self.cfg.network.recipient_selection

    def candidates(self, agent_id: str) -> list[str]:
        return self.neighbours[agent_id]

    def route_messages(self, messages: list[OutgoingMessage], round_id: int) -> RoutingResult:
        for m in messages:
            if m.recipient_id not in self.neighbours[m.sender_id]:
                raise ValueError(f"{m.sender_id} -> {m.recipient_id} is not an edge")
        return super().route_messages(messages, round_id)

    def network_edges(self) -> list[tuple[str, str]]:
        return list(self.edges)

    def describe(self) -> dict:
        d = super().describe()
        n = len(self.agent_ids)
        d.update(
            target_mean_degree=self.cfg.network.target_mean_degree,
            edge_probability=edge_probability(n, self.cfg.network.target_mean_degree),
            realised_mean_degree=2 * len(self.edges) / n,
            resample_attempts=getattr(self, "resample_attempts", None),
        )
        return d

    def state_dict(self) -> dict:
        s = super().state_dict()
        s["edges"] = [list(e) for e in self.edges]
        s["resample_attempts"] = getattr(self, "resample_attempts", None)
        return s

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.resample_attempts = state.get("resample_attempts")
        self._set_edges([tuple(e) for e in state["edges"]])
