"""Loading run datasets for analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import pandas as pd

from ..config.loader import load_config
from ..config.schema import ExperimentConfig


@dataclass
class RunData:
    info: dict
    agents: pd.DataFrame
    messages: pd.DataFrame
    exposures: pd.DataFrame
    probes: pd.DataFrame
    edges: pd.DataFrame
    run_dir: Path

    @property
    def run_id(self) -> str:
        return self.info["run_id"]

    @property
    def environment(self) -> str:
        return self.info["environment_type"]

    @property
    def agent_ids(self) -> list[str]:
        return sorted(self.agents["agent_id"])

    def graph(self) -> nx.Graph | None:
        if self.edges.empty:
            return None
        g = nx.Graph()
        g.add_nodes_from(self.agent_ids)
        g.add_edges_from(zip(self.edges["source"], self.edges["target"]))
        return g


@dataclass
class AnalysisContext:
    """Shared, experiment-wide products passed to every metric."""

    cfg: ExperimentConfig
    seed: int
    tokens: dict[str, list[str]] = field(default_factory=dict)  # message_id -> content tokens
    embeddings: dict[str, object] = field(default_factory=dict)  # message_id -> vector
    topic_dist: dict[str, object] = field(default_factory=dict)  # message_id -> topic distribution
    info: dict = field(default_factory=dict)  # backend descriptions


def resolve_experiment_dir(ref: str | Path, data_root: str | Path = "data/experiments") -> Path:
    p = Path(ref)
    if (p / "config.yaml").exists():
        return p
    q = Path(data_root) / str(ref)
    if (q / "config.yaml").exists():
        return q
    raise FileNotFoundError(f"no experiment {ref!r} (looked in {p} and {q})")


def experiment_config(exp_dir: Path) -> ExperimentConfig:
    return load_config(exp_dir / "config.yaml")


def load_run(run_dir: Path) -> RunData:
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    def t(name: str) -> pd.DataFrame:
        path = run_dir / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    return RunData(info, t("agents"), t("messages"), t("exposures"), t("probes"), t("network_edges"), run_dir)


def load_runs(exp_dir: Path, include_incomplete: bool = False) -> list[RunData]:
    runs = []
    for rd in sorted((exp_dir / "runs").glob("*/run.json")):
        info = json.loads(rd.read_text(encoding="utf-8"))
        if info.get("status") == "complete" or include_incomplete:
            runs.append(load_run(rd.parent))
    return runs
