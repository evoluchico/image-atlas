"""Network-science measures: local vs global consensus, emergent interaction networks."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _pair_distance(mat: pd.DataFrame, u: str, v: str) -> float:
    return float(np.linalg.norm(mat.loc[u].to_numpy() - mat.loc[v].to_numpy()) / np.sqrt(mat.shape[1]))


def morans_i(values: np.ndarray, W: np.ndarray) -> float:
    z = values - values.mean()
    denom = (z**2).sum()
    if denom == 0 or W.sum() == 0:
        return float("nan")
    return float(len(values) / W.sum() * (z @ W @ z) / denom)


def stance_network_metrics(
    mat: pd.DataFrame, graph: nx.Graph, communities: list[set[str]] | None = None, prefix: str = ""
) -> dict[str, float]:
    """Local vs global similarity of stances over a graph.

    neighbour_distance < global_distance means neighbours are more alike than
    random pairs (local consensus); ratio = neighbour / global.
    """
    nodes = [n for n in mat.index if n in graph]
    g = graph.subgraph(nodes)
    out: dict[str, float] = {}
    if len(nodes) < 3 or g.number_of_edges() == 0:
        return out
    m = mat.loc[nodes]
    X = m.to_numpy()
    from scipy.spatial.distance import pdist, squareform

    Dm = squareform(pdist(X)) / np.sqrt(X.shape[1])
    idx = {n: i for i, n in enumerate(nodes)}
    edge_d = [Dm[idx[u], idx[v]] for u, v in g.edges()]
    glob = Dm[np.triu_indices(len(nodes), 1)].mean()
    out[prefix + "neighbour_distance"] = float(np.mean(edge_d))
    out[prefix + "global_distance"] = float(glob)
    out[prefix + "local_global_ratio"] = float(np.mean(edge_d) / glob) if glob > 0 else float("nan")
    local = []
    for n in nodes:
        nb = list(g.neighbors(n))
        if nb:
            local.append(np.mean([Dm[idx[n], idx[v]] for v in nb]))
    out[prefix + "local_disagreement"] = float(np.mean(local))
    A = nx.to_numpy_array(g, nodelist=nodes)
    out[prefix + "morans_i"] = float(np.nanmean([morans_i(X[:, j], A) for j in range(X.shape[1])]))
    mean_stance = X.mean(axis=1)
    a = [mean_stance[idx[u]] for u, v in g.edges()] + [mean_stance[idx[v]] for u, v in g.edges()]
    b = [mean_stance[idx[v]] for u, v in g.edges()] + [mean_stance[idx[u]] for u, v in g.edges()]
    if np.std(a) > 0:
        out[prefix + "stance_assortativity"] = float(pearsonr(a, b)[0])
    if nx.is_connected(g):
        sp = dict(nx.all_pairs_shortest_path_length(g))
        iu = np.triu_indices(len(nodes), 1)
        path = np.array([sp[nodes[i]][nodes[j]] for i, j in zip(*iu)])
        if np.std(path) > 0:
            out[prefix + "path_distance_corr"] = float(spearmanr(path, Dm[iu])[0])
    if communities:
        within, var_within = [], []
        total_var = X.var(axis=0).sum()
        for c in communities:
            members = [idx[n] for n in c if n in idx]
            if len(members) >= 2:
                sub = Dm[np.ix_(members, members)]
                within.append(sub[np.triu_indices(len(members), 1)].mean())
                var_within.append(X[members].var(axis=0).sum() * len(members))
        if within:
            out[prefix + "community_distance"] = float(np.mean(within))
            if total_var > 0:
                out[prefix + "community_variance_ratio"] = float(sum(var_within) / (total_var * len(nodes)))
    return out


def detect_communities(graph: nx.Graph) -> list[set[str]]:
    return [set(c) for c in nx.algorithms.community.greedy_modularity_communities(graph)]


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def interaction_network_metrics(messages: pd.DataFrame, agent_ids: list[str]) -> dict[str, float]:
    """Emergent directed interaction network from dyadic messages (i -> j counts)."""
    dm = messages.dropna(subset=["recipient_id"])
    if dm.empty:
        return {}
    pairs = dm.groupby(["sender_id", "recipient_id"]).size()
    edges = set(pairs.index)
    reciprocal = sum((v, u) in edges for u, v in edges)
    indeg = dm["recipient_id"].value_counts().reindex(agent_ids, fill_value=0).to_numpy()
    return {
        "interaction_distinct_pairs": float(len(edges)),
        "interaction_reciprocity": float(reciprocal / len(edges)),
        "interaction_indegree_gini": gini(indeg),
        "interaction_repeat_share": float((pairs - 1).clip(lower=0).sum() / len(dm)),
    }


def interaction_graph(messages: pd.DataFrame, agent_ids: list[str]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(agent_ids)
    dm = messages.dropna(subset=["recipient_id"])
    g.add_edges_from(zip(dm["sender_id"], dm["recipient_id"]))
    return g
