"""Initial agent populations.

Positions are generated *externally* and assigned to agents explicitly. A
population is generated once per (population size, replicate), saved, and
then reused by every communication environment of that replicate, so the
environments are compared on identical starting populations (paired design).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config.schema import ExperimentConfig
from ..seeds import derive_seed


@dataclass
class AgentProfile:
    agent_id: str
    agent_index: int
    public_name: str
    latent_position: float | None
    stance: dict[str, float]
    model_key: str = "default"
    # Reserved for future persona manipulation; always None in the base design.
    persona: str | None = None


@dataclass
class Population:
    population_id: str
    population_size: int
    replicate: int
    seed: int
    method: str
    profiles: list[AgentProfile] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Population":
        d = dict(d)
        d["profiles"] = [AgentProfile(**p) for p in d["profiles"]]
        return cls(**d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Population":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def population_id(n: int, replicate: int) -> str:
    return f"pop_N{n:03d}_r{replicate:03d}"


def agent_id(index: int) -> str:
    return f"Agent_{index:03d}"


def _latent(cfg: ExperimentConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    ic = cfg.initialisation
    m = ic.method
    if m == "stratified":
        # One agent per equal-width stratum of [-1, 1], uniformly placed within it.
        edges = np.linspace(-1.0, 1.0, n + 1)
        z = edges[:-1] + rng.random(n) * (edges[1:] - edges[:-1])
    elif m == "uniform":
        z = rng.uniform(-1.0, 1.0, n)
    elif m == "normal":
        z = rng.normal(0.0, ic.normal_sd, n)
    elif m == "bimodal":
        side = np.where(np.arange(n) < n // 2, -1.0, 1.0)
        z = side * ic.bimodal_separation + rng.normal(0.0, ic.bimodal_sd, n)
    elif m == "clustered":
        k = max(1, min(ic.n_clusters, n))
        centres = np.linspace(-1.0, 1.0, k + 2)[1:-1] if k > 1 else np.array([0.0])
        assign = np.arange(n) % k
        z = centres[assign] + rng.normal(0.0, ic.cluster_sd, n)
    else:
        raise ValueError(m)
    return np.clip(z, -1.0, 1.0)


def _from_custom_file(cfg: ExperimentConfig, n: int, rng: np.random.Generator) -> tuple[np.ndarray | None, np.ndarray]:
    df = pd.read_csv(cfg.initialisation.custom_file)
    missing = [d for d in cfg.dimension_names if d not in df.columns]
    if missing:
        raise ValueError(f"custom initialisation file lacks columns {missing}")
    if len(df) < n:
        raise ValueError(f"custom initialisation file has {len(df)} rows; need at least {n}")
    rows = df.iloc[np.sort(rng.choice(len(df), size=n, replace=False))] if len(df) > n else df
    latent = rows["latent"].to_numpy(float) if "latent" in rows.columns else None
    return latent, rows[cfg.dimension_names].to_numpy(float)


def _assign_models(cfg: ExperimentConfig, n: int, rng: np.random.Generator) -> list[str]:
    pc = cfg.population
    if pc.model_assignment == "homogeneous":
        return ["default"] * n
    keys = list(pc.models)
    if pc.model_assignment == "random":
        w = np.array([pc.weights.get(k, 1.0) for k in keys], float)
        return [keys[i] for i in rng.choice(len(keys), size=n, p=w / w.sum())]
    if pc.model_assignment == "stratified":
        return [keys[i % len(keys)] for i in range(n)]
    df = pd.read_csv(pc.custom_file)
    mapping = dict(zip(df["agent_index"].astype(int), df["model_key"].astype(str)))
    out = [mapping.get(i) for i in range(n)]
    if any(k not in keys for k in out):
        raise ValueError("custom model assignment file must map every agent_index to a key in population.models")
    return out


def generate_population(cfg: ExperimentConfig, n: int, replicate: int) -> Population:
    seed = derive_seed(cfg.seed, "profile", n, replicate)
    rng = np.random.default_rng(seed)
    dims = cfg.stance_dimensions
    names = list(dims)

    if cfg.initialisation.method == "custom_file":
        latent, values = _from_custom_file(cfg, n, rng)
    else:
        latent = _latent(cfg, n, rng)
        loadings = np.array([dims[d].loading for d in names])
        # Normalised position on each dimension: 0.5 + 0.5 * loading * z + noise.
        # Idiosyncratic noise keeps dimensions from being perfectly correlated.
        u = 0.5 + 0.5 * latent[:, None] * loadings[None, :]
        u = u + rng.normal(0.0, cfg.initialisation.noise_sd, size=u.shape)
        u = np.clip(u, 0.0, 1.0)
        lo = np.array([dims[d].min for d in names])
        hi = np.array([dims[d].max for d in names])
        values = lo + u * (hi - lo)

    # Randomly assign positions to internal IDs, so e.g. Agent_000 is not
    # systematically the most extreme agent.
    order = rng.permutation(n)
    models = _assign_models(cfg, n, rng)
    profiles = []
    for idx in range(n):
        src = int(order[idx])
        profiles.append(
            AgentProfile(
                agent_id=agent_id(idx),
                agent_index=idx,
                public_name=cfg.prompts.participant_label.format(n=idx + 1),
                latent_position=None if latent is None else float(latent[src]),
                stance={d: float(round(values[src, j], 2)) for j, d in enumerate(names)},
                model_key=models[idx],
            )
        )
    return Population(
        population_id=population_id(n, replicate),
        population_size=n,
        replicate=replicate,
        seed=seed,
        method=cfg.initialisation.method,
        profiles=profiles,
    )
