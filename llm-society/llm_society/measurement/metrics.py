"""Metric plugin registry and the built-in metrics.

A metric turns one run (plus shared experiment-wide products such as tokens,
embeddings and topic distributions) into long-format rows:

    {"round": int, "metric_name": str, "metric_value": float, "metric_metadata": dict}

Adding a measurement = writing a class with `name`, `flag` and `compute`, and
calling `register_metric`. Neither the simulation engine nor the pipeline
needs to change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from .lexical import lexical_round_metrics
from .network import detect_communities, interaction_graph, interaction_network_metrics, stance_network_metrics
from .semantic import mean_pairwise_cosine_distance
from .stance import cluster_summary, dispersion_rows, mean_pairwise_distance, stance_matrices
from .topics import entropy_bits
from .transmission import transmission_analysis

if TYPE_CHECKING:
    from ..analysis.data import AnalysisContext, RunData


class Metric(Protocol):
    name: str
    flag: str  # AnalysisConfig attribute that enables it

    def compute(self, run: "RunData", ctx: "AnalysisContext") -> list[dict]: ...


METRICS: dict[str, Metric] = {}


def register_metric(metric: Metric) -> Metric:
    METRICS[metric.name] = metric
    return metric


def _rows(round_id: int, values: dict[str, float], meta: dict | None = None) -> list[dict]:
    return [
        {"round": int(round_id), "metric_name": k, "metric_value": float(v), "metric_metadata": meta or {}}
        for k, v in values.items()
        if v is not None
    ]


def _windows(run: "RunData", window: int) -> dict[int, dict[str, list[str]]]:
    """round -> agent -> message ids in rounds (t - window, t]."""
    m = run.messages
    out: dict[int, dict[str, list[str]]] = {}
    for t in sorted(m["round"].unique()):
        sub = m[(m["round"] > t - window) & (m["round"] <= t)]
        out[int(t)] = sub.groupby("sender_id")["message_id"].apply(list).to_dict()
    return out


class StanceMetric:
    name = "stance"
    flag = "stance_convergence"

    def compute(self, run, ctx):
        mats = stance_matrices(run.probes, ctx.cfg.stance_dimensions)
        if not mats:
            return []
        rows = []
        r0 = min(mats)
        base = mats[r0]
        d0 = mean_pairwise_distance(base.to_numpy())
        c0 = base.mean().to_numpy()
        base_mean = base.mean(axis=1)
        lo_q, hi_q = base_mean.quantile(0.2), base_mean.quantile(0.8)
        low_group = set(base_mean[base_mean <= lo_q].index)
        high_group = set(base_mean[base_mean >= hi_q].index)
        seed = ctx.seed
        for t, mat in sorted(mats.items()):
            X = mat.to_numpy()
            d = mean_pairwise_distance(X)
            vals = {
                "stance_distance": d,
                "convergence_ratio": d / d0 if d0 else float("nan"),
                "centroid_displacement": float(np.linalg.norm(X.mean(axis=0) - c0) / np.sqrt(X.shape[1])),
                "probe_valid_share": len(mat) / max(1, len(run.agents)),
            }
            vals.update(dispersion_rows(mat))
            vals.update(cluster_summary(X, ctx.cfg.analysis.max_clusters, seed))
            # Persistence of initially extreme positions: share of the initial
            # bottom/top quintile still in the bottom/top quintile at round t.
            mean_t = mat.mean(axis=1)
            lo_t, hi_t = mean_t.quantile(0.2), mean_t.quantile(0.8)
            still = [a for a in low_group if a in mean_t and mean_t[a] <= lo_t]
            still += [a for a in high_group if a in mean_t and mean_t[a] >= hi_t]
            denom = len(low_group) + len(high_group)
            vals["extreme_persistence"] = len(still) / denom if denom else float("nan")
            rows += _rows(t, vals, {"source": "self_report_probe"})
        return rows


class LexicalMetric:
    name = "lexical"
    flag = "lexical_convergence"

    def compute(self, run, ctx):
        a = ctx.cfg.analysis
        rows = []
        for t, by_agent in _windows(run, a.lexical_window).items():
            toks = {ag: [w for mid in mids for w in ctx.tokens.get(mid, [])] for ag, mids in by_agent.items()}
            rows += _rows(t, lexical_round_metrics(toks, a.lexical_top_k), {"window": a.lexical_window})
        rows += self._tfidf(run, ctx)
        return rows

    def _tfidf(self, run, ctx):
        from sklearn.feature_extraction.text import TfidfVectorizer

        from .lexical import mean_pairwise_cosine_similarity

        if run.messages.empty:
            return []
        docs = {mid: " ".join(ctx.tokens.get(mid, [])) for mid in run.messages["message_id"]}
        vec = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True).fit(list(docs.values()) or [""])
        rows = []
        for t, by_agent in _windows(run, ctx.cfg.analysis.lexical_window).items():
            agent_docs = [" ".join(docs[m] for m in mids) for mids in by_agent.values()]
            X = vec.transform(agent_docs).toarray()
            rows += _rows(t, {"tfidf_cosine": mean_pairwise_cosine_similarity(X)})
        return rows


class SemanticMetric:
    name = "semantic"
    flag = "semantic_convergence"

    def compute(self, run, ctx):
        if not ctx.embeddings:
            return []
        rows = []
        for t, by_agent in _windows(run, ctx.cfg.analysis.lexical_window).items():
            cents = [
                np.mean([ctx.embeddings[m] for m in mids], axis=0) for mids in by_agent.values() if mids
            ]
            if len(cents) >= 2:
                rows += _rows(t, {"semantic_distance": mean_pairwise_cosine_distance(np.array(cents))}, ctx.info.get("embedding"))
        return rows


class TopicMetric:
    name = "topic"
    flag = "topic_convergence"

    def compute(self, run, ctx):
        from .lexical import mean_pairwise_jsd

        if not ctx.topic_dist:
            return []
        rows = []
        for t, by_agent in _windows(run, ctx.cfg.analysis.lexical_window).items():
            P = np.array([np.mean([ctx.topic_dist[m] for m in mids], axis=0) for mids in by_agent.values() if mids])
            if len(P) < 2:
                continue
            pop = P.mean(axis=0)
            vals = {"topic_jsd": mean_pairwise_jsd(P), "topic_entropy": entropy_bits(pop)}
            vals.update({f"topic_prevalence:{k:02d}": float(v) for k, v in enumerate(pop)})
            rows += _rows(t, vals, ctx.info.get("topics"))
        return rows


class NetworkMetric:
    name = "network"
    flag = "network_metrics"

    def compute(self, run, ctx):
        mats = stance_matrices(run.probes, ctx.cfg.stance_dimensions)
        rows = []
        g = run.graph()
        if g is not None:
            comms = detect_communities(g)
            for t, mat in sorted(mats.items()):
                rows += _rows(t, stance_network_metrics(mat, g, comms), {"graph": "imposed"})
            rows += _rows(0, {"graph_mean_degree": 2 * g.number_of_edges() / g.number_of_nodes(),
                              "graph_n_communities": float(len(comms))})
        dyadic = run.messages.dropna(subset=["recipient_id"]) if not run.messages.empty else run.messages
        if not dyadic.empty:
            for t, mat in sorted(mats.items()):
                if t == 0:
                    continue
                upto = dyadic[dyadic["round"] <= t]
                vals = interaction_network_metrics(upto, run.agent_ids)
                ig = interaction_graph(upto, run.agent_ids)
                vals.update(stance_network_metrics(mat, ig, None, prefix="interaction_"))
                rows += _rows(t, vals, {"graph": "empirical_cumulative"})
        return rows


class TransmissionMetric:
    name = "transmission"
    flag = "transmission"

    def compute(self, run, ctx):
        if run.messages.empty or run.exposures.empty:
            return []
        def units_of(mid, text):
            return {w for w in ctx.tokens.get(mid, []) if len(w) >= 3}

        metric_rows, events = transmission_analysis(run.messages, run.exposures, units_of)
        ctx.info.setdefault("transmission_events", {})[run.run_id] = events
        rows = []
        for r in metric_rows:
            t = r.pop("round")
            rows += _rows(t, {k: v for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))}, {"unit": "word"})
        return rows


for _m in (StanceMetric(), LexicalMetric(), SemanticMetric(), TopicMetric(), NetworkMetric(), TransmissionMetric()):
    register_metric(_m)
