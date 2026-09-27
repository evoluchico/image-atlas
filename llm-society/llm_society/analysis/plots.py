"""Core figures (static PNG, matplotlib).

Colour follows the entity: each environment has one fixed colour everywhere
(categorical slots in fixed order; the no-interaction control is a neutral
dashed grey). Magnitude uses a single-hue sequential ramp; stance positions
use a diverging red <-> blue ramp with a neutral grey midpoint.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ..measurement.stance import stance_matrices  # noqa: E402
from .data import RunData  # noqa: E402

ENV_COLORS = {
    "forum": "#2a78d6",
    "open_dyadic": "#eb6834",
    "erdos_renyi": "#1baf7a",
    "no_interaction": "#8a8985",
}
ENV_LABELS = {
    "forum": "Public forum",
    "open_dyadic": "Open dyadic",
    "erdos_renyi": "Network dyadic (ER)",
    "no_interaction": "No interaction (control)",
}
EXTRA_COLORS = ["#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list("stance", ["#c23b3a", "#e8908f", "#f0efec", "#86b6ef", "#1c5cab"])


def env_color(env: str, i: int = 0) -> str:
    return ENV_COLORS.get(env, EXTRA_COLORS[i % len(EXTRA_COLORS)])


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "legend.frameon": False,
            "font.size": 9,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _envs(df: pd.DataFrame) -> list[str]:
    order = list(ENV_COLORS)
    return sorted(df["environment_type"].unique(), key=lambda e: (order.index(e) if e in order else 99, e))


def trajectory_figure(traj: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path) -> Path | None:
    """Mean across runs (line) with run-bootstrap 95% CI (band); one panel per N."""
    df = traj[traj["metric_name"] == metric]
    if df.empty:
        return None
    _style()
    sizes = sorted(df["population_size"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.6 * len(sizes) + 1.8, 3.2), sharey=True, squeeze=False)
    envs = _envs(df)
    for ax, n in zip(axes[0], sizes):
        sub = df[df["population_size"] == n]
        for i, env in enumerate(envs):
            s = sub[sub["environment_type"] == env].sort_values("round")
            if s.empty:
                continue
            c = env_color(env, i)
            ls = "--" if env == "no_interaction" else "-"
            ax.fill_between(s["round"], s["ci_low"], s["ci_high"], color=c, alpha=0.15, linewidth=0)
            ax.plot(s["round"], s["mean"], color=c, lw=2, ls=ls, marker="o", ms=3.5, label=ENV_LABELS.get(env, env))
        ax.set_title(f"N = {n}")
        ax.set_xlabel("Round")
        ax.xaxis.get_major_locator().set_params(integer=True)
    axes[0][0].set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), labelcolor=TEXT)
    fig.suptitle(title, x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def final_heatmap(matrix: pd.DataFrame, label: str, title: str, path: Path) -> Path | None:
    if matrix.empty:
        return None
    _style()
    envs = _envs(matrix.reset_index())
    matrix = matrix.reindex(envs)
    fig, ax = plt.subplots(figsize=(1.4 * matrix.shape[1] + 3.2, 0.7 * matrix.shape[0] + 1.6))
    vals = matrix.to_numpy(float)
    im = ax.imshow(vals, cmap=SEQUENTIAL, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(matrix.shape[1]), [f"N = {c}" for c in matrix.columns])
    ax.set_yticks(range(matrix.shape[0]), [ENV_LABELS.get(e, e) for e in matrix.index])
    finite = vals[np.isfinite(vals)]
    mid = (finite.min() + finite.max()) / 2 if finite.size else 0
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if vals[i, j] > mid else TEXT)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(label)
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left")
    fig.savefig(path)
    plt.close(fig)
    return path


def centroid_figure(traj: pd.DataFrame, dims: list[str], path: Path) -> Path | None:
    """Population centroid per stance dimension over rounds (small multiples)."""
    df = traj[traj["metric_name"].isin([f"stance_mean:{d}" for d in dims])]
    if df.empty:
        return None
    _style()
    sizes = sorted(df["population_size"].unique())
    fig, axes = plt.subplots(len(sizes), len(dims), figsize=(2.6 * len(dims) + 1.8, 2.3 * len(sizes)), sharex=True, sharey=True, squeeze=False)
    envs = _envs(df)
    for r, n in enumerate(sizes):
        for c, d in enumerate(dims):
            ax = axes[r][c]
            s = df[(df["population_size"] == n) & (df["metric_name"] == f"stance_mean:{d}")]
            for i, env in enumerate(envs):
                e = s[s["environment_type"] == env].sort_values("round")
                col = env_color(env, i)
                ax.fill_between(e["round"], e["ci_low"], e["ci_high"], color=col, alpha=0.12, linewidth=0)
                ax.plot(e["round"], e["mean"], color=col, lw=2, ls="--" if env == "no_interaction" else "-",
                        label=ENV_LABELS.get(env, env))
            ax.axhline(0.5, color=TEXT_2, lw=0.8, ls=":")
            if r == 0:
                ax.set_title(d.replace("_", " "), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"N = {n}\ncentroid (0-1)")
            if r == len(sizes) - 1:
                ax.set_xlabel("Round")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), labelcolor=TEXT)
    fig.suptitle("Population movement: stance centroid by dimension", x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def individual_trajectories(runs: list[RunData], dims: dict, path: Path, n_agents: int = 12, seed: int = 0) -> Path | None:
    """Mean self-reported stance of a random subset of agents, one panel per run.
    Line colour = the agent's initial position (diverging ramp)."""
    if not runs:
        return None
    _style()
    fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 3.2), sharey=True, squeeze=False)
    rng = np.random.default_rng(seed)
    for ax, run in zip(axes[0], runs):
        mats = stance_matrices(run.probes, dims)
        if not mats:
            continue
        wide = pd.DataFrame({t: m.mean(axis=1) for t, m in mats.items()})
        chosen = rng.choice(wide.index.to_numpy(), size=min(n_agents, len(wide)), replace=False)
        for a in chosen:
            row = wide.loc[a].dropna()
            ax.plot(row.index, row.values, color=DIVERGING(float(wide.loc[a].iloc[0])), lw=1.6, alpha=0.9)
        ax.set_title(f"{ENV_LABELS.get(run.environment, run.environment)}\nN = {run.info['population_size']}, rep {run.info['replicate']}", fontsize=9)
        ax.set_xlabel("Round")
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("Mean stance (0-1)")
    fig.suptitle("Individual trajectories (colour = initial position)", x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def network_snapshots(run: RunData, dims: dict, path: Path, seed: int = 0) -> Path | None:
    """Imposed network at initial / midpoint / final probe, fixed layout,
    nodes coloured by mean self-reported stance."""
    g = run.graph()
    mats = stance_matrices(run.probes, dims)
    if g is None or not mats:
        return None
    _style()
    rounds = sorted(mats)
    picks = [rounds[0], rounds[len(rounds) // 2], rounds[-1]]
    pos = nx.spring_layout(g, seed=seed)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, t in zip(axes, picks):
        mean = mats[t].mean(axis=1)
        nodes = [n for n in g.nodes if n in mean.index]
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color=GRID, width=0.8)
        nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=[mean[n] for n in nodes], cmap=DIVERGING,
                               vmin=0, vmax=1, node_size=max(20, 1800 // max(1, len(nodes))), ax=ax,
                               edgecolors=SURFACE, linewidths=1.0)
        ax.set_title({picks[0]: "Initial", picks[1]: "Midpoint", picks[2]: "Final"}[t] + f" (round {t})")
        ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("Mean stance (0-1)")
    cb.outline.set_visible(False)
    fig.suptitle(f"Network condition: {run.run_id}", x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.savefig(path)
    plt.close(fig)
    return path
