"""AnalysisPipeline: post-hoc measurement of every completed run.

Runs strictly after simulation (nothing here can influence agent behaviour).
Experiment-wide products (tokens, embeddings, a single shared topic model)
are computed on the pooled corpus first so that measures are comparable
across conditions; then every registered, enabled metric is applied per run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..measurement.lexical import Preprocessor
from ..measurement.metrics import METRICS
from ..measurement.semantic import Embedder
from ..measurement.topics import SklearnTopicModel
from ..seeds import derive_seed, make_rng
from ..simulation.events import utcnow
from ..storage.parquet import atomic_write_json, write_frame
from .data import AnalysisContext, RunData, experiment_config, load_runs
from .statistics import condition_trajectories, final_values, mixed_models, paired_differences

KEY_METRICS = [
    "stance_distance", "convergence_ratio", "centroid_displacement", "stance_sd_mean", "cluster_count",
    "lexical_jsd", "tfidf_cosine", "semantic_distance", "topic_jsd", "topic_entropy",
    "local_global_ratio", "adoption_lift",
]


class AnalysisPipeline:
    def __init__(self, exp_dir: Path, log: Callable[[str], None] = print, metrics: list[str] | None = None):
        self.exp_dir = Path(exp_dir)
        self.cfg = experiment_config(self.exp_dir)
        self.out = self.exp_dir / "analysis"
        self.log = log
        self.metric_names = metrics

    def build_context(self, runs: list[RunData]) -> AnalysisContext:
        a = self.cfg.analysis
        seed = derive_seed(self.cfg.seed, "analysis")
        ctx = AnalysisContext(cfg=self.cfg, seed=seed % (2**31))
        msgs = pd.concat([r.messages for r in runs if not r.messages.empty], ignore_index=True) if runs else pd.DataFrame()
        if msgs.empty:
            return ctx
        # Message ids are only unique within a run.
        msgs["key"] = msgs["run_id"] + "/" + msgs["message_id"]
        pre = Preprocessor(a.remove_stopwords, a.lemmatise)
        ctx.info["preprocessing"] = pre.describe()
        processed = []
        for key, text in zip(msgs["key"], msgs["text"]):
            ctx.tokens[key] = pre.content_tokens(text)
            processed.append(" ".join(ctx.tokens[key]))
        write_frame(
            self.out / "processed_messages.parquet",
            msgs[["run_id", "message_id", "round", "sender_id", "text"]].assign(processed_tokens=processed),
        )
        texts = msgs["text"].fillna("").tolist()
        if a.semantic_convergence:
            emb = Embedder(a.embedding_backend, a.embedding_model, ctx.seed)
            E = emb.embed(texts)
            ctx.embeddings = dict(zip(msgs["key"], E))
            ctx.info["embedding"] = emb.describe()
            self.log(f"embeddings: {emb.describe()}")
        if a.topic_convergence and len(texts) >= 10:
            tm = SklearnTopicModel(a.topic_model, a.n_topics, ctx.seed)
            W, terms = tm.fit_transform(texts)
            ctx.topic_dist = dict(zip(msgs["key"], W))
            ctx.info["topics"] = tm.describe()
            pd.DataFrame(
                {"topic": range(len(terms)), "top_terms": [", ".join(t) for t in terms]}
            ).to_csv(self.out / "topic_terms.csv", index=False)
        return ctx

    def _run_view(self, run: RunData, ctx: AnalysisContext) -> tuple[RunData, AnalysisContext]:
        """Present experiment-wide products keyed by this run's message ids."""
        prefix = run.run_id + "/"
        sub = AnalysisContext(
            cfg=ctx.cfg,
            seed=ctx.seed,
            tokens={k[len(prefix):]: v for k, v in ctx.tokens.items() if k.startswith(prefix)},
            embeddings={k[len(prefix):]: v for k, v in ctx.embeddings.items() if k.startswith(prefix)},
            topic_dist={k[len(prefix):]: v for k, v in ctx.topic_dist.items() if k.startswith(prefix)},
            info=ctx.info,
        )
        return run, sub

    def run(self) -> pd.DataFrame:
        self.out.mkdir(parents=True, exist_ok=True)
        runs = load_runs(self.exp_dir)
        if not runs:
            raise RuntimeError(f"no completed runs in {self.exp_dir}")
        self.log(f"analysing {len(runs)} completed runs")
        ctx = self.build_context(runs)
        enabled = [
            m for name, m in METRICS.items()
            if (self.metric_names is None or name in self.metric_names) and getattr(self.cfg.analysis, m.flag, True)
        ]
        all_rows = []
        transmission_events = []
        for run in runs:
            r, sub = self._run_view(run, ctx)
            rows = []
            for metric in enabled:
                for row in metric.compute(r, sub):
                    row["run_id"] = run.run_id
                    rows.append(row)
            events = sub.info.get("transmission_events", {}).pop(run.run_id, [])
            transmission_events += [e | {"run_id": run.run_id} for e in events]
            df = pd.DataFrame(rows, columns=["run_id", "round", "metric_name", "metric_value", "metric_metadata"])
            df["metric_metadata"] = df["metric_metadata"].map(lambda m: json.dumps(m, default=str))
            write_frame(run.run_dir / "metrics.parquet", df)
            df = df.assign(
                environment_type=run.info["environment_type"],
                population_size=run.info["population_size"],
                replicate=run.info["replicate"],
            )
            all_rows.append(df)
            self.log(f"  {run.run_id}: {len(df)} metric values")
        metrics = pd.concat(all_rows, ignore_index=True)
        write_frame(self.out / "metrics.parquet", metrics, csv=True)
        if transmission_events:
            write_frame(self.out / "transmission_events.parquet", pd.DataFrame(transmission_events))
        self.summaries(metrics, runs)
        atomic_write_json(
            self.out / "analysis_info.json",
            {"created": utcnow(), "n_runs": len(runs), "metrics": [m.name for m in enabled],
             "context": {k: v for k, v in ctx.info.items() if k != "transmission_events"}},
        )
        return metrics

    def summaries(self, metrics: pd.DataFrame, runs: list[RunData]) -> None:
        a = self.cfg.analysis
        rng = make_rng(self.cfg.seed, "analysis", "bootstrap")
        traj = condition_trajectories(metrics, rng, a.bootstrap_samples)
        write_frame(self.out / "condition_trajectories.parquet", traj, csv=True)
        present = [m for m in KEY_METRICS if m in set(metrics["metric_name"])]
        paired = paired_differences(metrics, present, rng, a.bootstrap_samples)
        paired.to_csv(self.out / "paired_differences.csv", index=False)
        (self.out / "mixed_models.txt").write_text(mixed_models(metrics, [m for m in present if m != "cluster_count"]))
        self.run_summary(metrics).to_csv(self.out / "run_summary.csv", index=False)
        self.topic_lifecycle(metrics)

    def run_summary(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """One row per run: initial vs final values of the headline measures."""
        rows = []
        for run_id, grp in metrics.groupby("run_id"):
            first = grp.iloc[0]
            row = {"run_id": run_id, "environment_type": first["environment_type"],
                   "population_size": first["population_size"], "replicate": first["replicate"]}

            def at(name, which):
                s = grp[grp["metric_name"] == name].sort_values("round")
                if s.empty:
                    return float("nan")
                return float(s["metric_value"].iloc[0 if which == "first" else -1])

            row["initial_dispersion"] = at("stance_distance", "first")
            row["final_dispersion"] = at("stance_distance", "last")
            row["dispersion_pct_change"] = 100 * (row["final_dispersion"] / row["initial_dispersion"] - 1) if row["initial_dispersion"] else float("nan")
            row["final_convergence_ratio"] = at("convergence_ratio", "last")
            row["centroid_displacement"] = at("centroid_displacement", "last")
            for name in ("lexical_jsd", "semantic_distance", "topic_jsd"):
                row[f"{name}_first"] = at(name, "first")
                row[f"{name}_final"] = at(name, "last")
                row[f"{name}_change"] = row[f"{name}_final"] - row[f"{name}_first"]
            row["final_cluster_count"] = at("cluster_count", "last")
            row["final_local_global_ratio"] = at("local_global_ratio", "last")
            rows.append(row)
        return pd.DataFrame(rows)

    def topic_lifecycle(self, metrics: pd.DataFrame) -> None:
        prev = metrics[metrics["metric_name"].str.startswith("topic_prevalence:")]
        if prev.empty:
            return
        thr = self.cfg.analysis.topic_presence_threshold
        rows = []
        for (run_id, name), grp in prev.groupby(["run_id", "metric_name"]):
            g = grp.sort_values("round")
            present = g[g["metric_value"] >= thr]
            peak = g.loc[g["metric_value"].idxmax()]
            rows.append({
                "run_id": run_id, "environment_type": g["environment_type"].iloc[0],
                "population_size": g["population_size"].iloc[0], "topic": int(name.split(":")[1]),
                "first_appearance_round": int(present["round"].iloc[0]) if len(present) else None,
                "peak_prevalence": float(peak["metric_value"]), "peak_round": int(peak["round"]),
                "final_prevalence": float(g["metric_value"].iloc[-1]),
                "extinct_by_final": bool(len(present) and g["metric_value"].iloc[-1] < thr),
            })
        pd.DataFrame(rows).to_csv(self.out / "topic_lifecycle.csv", index=False)


def final_matrix(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    fin = final_values(metrics[metrics["metric_name"] == metric])
    return fin.pivot_table(index="environment_type", columns="population_size", values="metric_value", aggfunc=np.mean)
