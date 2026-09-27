"""Summary tables, figures and a Markdown report for an experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .data import experiment_config, load_runs
from .pipeline import AnalysisPipeline, final_matrix
from .plots import (
    ENV_COLORS,
    centroid_figure,
    final_heatmap,
    individual_trajectories,
    network_snapshots,
    trajectory_figure,
)

INTERPRETATION = (
    "Agents are independent conversational instances of the configured LLM, not simulations of "
    "individual humans. Results describe what happens when instances of this model are placed in "
    "these communication environments; they are not evidence about human behaviour."
)


def build_report(exp_dir: Path, log: Callable[[str], None] = print, reanalyse: bool = False) -> Path:
    exp_dir = Path(exp_dir)
    cfg = experiment_config(exp_dir)
    analysis = exp_dir / "analysis"
    if reanalyse or not (analysis / "metrics.parquet").exists():
        AnalysisPipeline(exp_dir, log=log).run()
    metrics = pd.read_parquet(analysis / "metrics.parquet")
    traj = pd.read_parquet(analysis / "condition_trajectories.parquet")
    fig_dir = exp_dir / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    figs: list[tuple[str, Path | None]] = [
        ("Figure 1 - Stance convergence", trajectory_figure(
            traj, "stance_distance", "Mean pairwise stance distance", "Stance convergence (self-reported probes)", fig_dir / "fig1_stance_convergence.png")),
        ("Figure 2 - Lexical convergence", trajectory_figure(
            traj, "lexical_jsd", "Mean pairwise lexical JSD", "Lexical convergence", fig_dir / "fig2_lexical_convergence.png")),
        ("Figure 3 - Topical convergence", trajectory_figure(
            traj, "topic_jsd", "Mean pairwise topic JSD", "Topical convergence (shared topic model)", fig_dir / "fig3_topic_convergence.png")),
        ("Figure 4 - Population movement", centroid_figure(traj, cfg.dimension_names, fig_dir / "fig4_centroids.png")),
        ("Figure 5 - Final-state matrix", final_heatmap(
            final_matrix(metrics, "convergence_ratio"), "Final D_T / D_0 (<1 = convergence)",
            "Final convergence ratio (mean over replicates)", fig_dir / "fig5_final_matrix.png")),
    ]
    runs = load_runs(exp_dir)
    example = [r for r in runs if r.info["replicate"] == 0]
    order = list(ENV_COLORS)
    example = sorted(example, key=lambda r: (r.info["population_size"], order.index(r.environment) if r.environment in order else 99))[:4]
    figs.append(("Figure 6 - Individual trajectories", individual_trajectories(example, cfg.stance_dimensions, fig_dir / "fig6_individual_trajectories.png")))
    net = [r for r in runs if not r.edges.empty and r.info["replicate"] == 0]
    if net:
        figs.append(("Figure 7 - Network condition", network_snapshots(net[0], cfg.stance_dimensions, fig_dir / "fig7_network_snapshots.png")))
    figs.append(("Semantic convergence", trajectory_figure(
        traj, "semantic_distance", "Mean pairwise cosine distance", "Semantic convergence (embedding centroids)", fig_dir / "fig_semantic_convergence.png")))
    if "local_global_ratio" in set(traj["metric_name"]):
        figs.append(("Local vs global consensus", trajectory_figure(
            traj, "local_global_ratio", "Neighbour / global distance", "Local vs global consensus (network runs)", fig_dir / "fig_local_global.png")))

    summary = pd.read_csv(analysis / "run_summary.csv")
    cond = summary.groupby(["environment_type", "population_size"]).agg(
        runs=("run_id", "count"),
        initial_dispersion=("initial_dispersion", "mean"),
        final_dispersion=("final_dispersion", "mean"),
        dispersion_pct_change=("dispersion_pct_change", "mean"),
        centroid_displacement=("centroid_displacement", "mean"),
        lexical_jsd_change=("lexical_jsd_change", "mean"),
        semantic_distance_change=("semantic_distance_change", "mean"),
        topic_jsd_change=("topic_jsd_change", "mean"),
        final_cluster_count=("final_cluster_count", "mean"),
    ).reset_index()
    cond.to_csv(exp_dir / "report" / "condition_summary.csv", index=False)
    paired = pd.read_csv(analysis / "paired_differences.csv") if (analysis / "paired_differences.csv").stat().st_size > 1 else pd.DataFrame()
    info = json.loads((analysis / "analysis_info.json").read_text())
    manifest = json.loads((exp_dir / "manifest.json").read_text()) if (exp_dir / "manifest.json").exists() else {}

    lines = [
        f"# {cfg.experiment_id}",
        "",
        cfg.experiment.description.strip(),
        "",
        f"> **Interpretation.** {INTERPRETATION}",
        "",
        "## Design",
        "",
        f"- Topic: **{cfg.topic.title}** - {cfg.topic.opening_question}",
        f"- Environments: {', '.join(cfg.all_environments())}",
        f"- Population sizes: {cfg.population_sizes}; replicates: {cfg.replicates}; rounds: {cfg.max_rounds}",
        f"- Model(s): {', '.join(f'{m.provider}/{m.model}' for m in cfg.model_configs().values())}",
        f"- Completed runs analysed: {info['n_runs']}; master seed {cfg.seed}; "
        f"git commit {manifest.get('git', {}).get('commit') or 'unknown'}",
        f"- Measurement backends: {json.dumps(info.get('context', {}), default=str)}",
        "",
        "The forum condition differs from the dyadic conditions in topology *and* in the amount of "
        "information each agent receives; see the exposure-matching robustness option. Compare every "
        "condition against the no-interaction control (if run) to separate socially induced "
        "convergence from drift toward the shared model prior.",
        "",
        "## Condition summary (means over replicates)",
        "",
        cond.round(3).to_markdown(index=False) if _has_tabulate() else cond.round(3).to_string(index=False),
        "",
    ]
    if not paired.empty:
        key = paired[paired["metric_name"].isin(["stance_distance", "lexical_jsd", "topic_jsd", "semantic_distance"])]
        lines += [
            "## Paired differences at the final round (same initial populations)",
            "",
            "Mean of (A - B) across replicates with 95% bootstrap CI over replicate pairs.",
            "",
            key.round(4).to_markdown(index=False) if _has_tabulate() else key.round(4).to_string(index=False),
            "",
        ]
    lines += ["## Figures", ""]
    for title, p in figs:
        if p is not None:
            lines += [f"### {title}", "", f"![{title}](figures/{p.name})", ""]
    lines += [
        "## Files",
        "",
        "- `analysis/metrics.parquet|csv` - every metric, run and round (long format)",
        "- `analysis/condition_trajectories.csv` - mean and run-bootstrap CI per condition and round",
        "- `analysis/run_summary.csv` - initial/final values per run",
        "- `analysis/paired_differences.csv`, `analysis/mixed_models.txt` - statistical comparisons",
        "- `analysis/topic_terms.csv`, `analysis/topic_lifecycle.csv` - shared topics and their entry/extinction",
        "- `analysis/transmission_events.parquet` - word adoption events with prior exposures",
        "",
    ]
    report = exp_dir / "report" / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    log(f"report written to {report}")
    return report


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except ImportError:
        return False
