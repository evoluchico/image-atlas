"""Command-line interface: `llm-society <command>`."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help="LLM society consensus experiments.")

DATA_ROOT_HELP = "Root directory holding experiment folders (default: the config's experiment.output_dir)."


def _load(config: Path):
    from .config import load_config

    try:
        return load_config(config)
    except Exception as exc:
        typer.secho(f"invalid configuration {config}:\n{exc}", fg="red", err=True)
        raise typer.Exit(2)


def _print_cost(cfg) -> None:
    from .cost import estimate

    est = estimate(cfg)
    typer.echo(est.table.to_string(index=False))
    typer.echo("")
    for k, v in est.totals.items():
        typer.echo(f"  {k:18s} {v:,}" if isinstance(v, int) else f"  {k:18s} {v}")
    typer.echo("  (approximate; thinking tokens of reasoning models are not included)")


@app.command()
def validate(config: Path):
    """Check a configuration without making any API calls."""
    from .environments.network import edge_probability
    from .simulation.population import generate_population

    cfg = _load(config)
    typer.echo(f"experiment      {cfg.experiment_id}")
    typer.echo(f"topic           {cfg.topic.id}: {cfg.topic.opening_question}")
    typer.echo(f"dimensions      {', '.join(cfg.dimension_names)}")
    typer.echo(f"environments    {', '.join(cfg.all_environments())}")
    typer.echo(f"population      {cfg.population_sizes} x {cfg.replicates} replicates, {cfg.max_rounds} rounds")
    typer.echo(f"probe rounds    {cfg.probe_rounds()}")
    for key, m in cfg.model_configs().items():
        typer.echo(f"model[{key}]  {m.provider}/{m.model}")
    if "erdos_renyi" in cfg.environments:
        for n in cfg.population_sizes:
            typer.echo(f"ER N={n:<4d}       p = {edge_probability(n, cfg.network.target_mean_degree):.4f}")
    for n in cfg.population_sizes:
        generate_population(cfg, n, 0)
    typer.secho("configuration OK", fg="green")


@app.command("estimate-cost")
def estimate_cost(config: Path):
    """Estimate the number of LLM calls and tokens before running."""
    _print_cost(_load(config))


@app.command()
def run(
    config: Path,
    data_root: Optional[Path] = typer.Option(None, help=DATA_ROOT_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan, generate populations and estimate cost; no LLM calls."),
    only: Optional[list[str]] = typer.Option(None, help="Run only these run IDs (repeatable)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation before paid API calls."),
):
    """Run (or continue) every run of an experiment."""
    from .simulation.experiment import ConfigFrozenError, ExperimentRunner

    cfg = _load(config)
    runner = ExperimentRunner(cfg, config_text=config.read_text(encoding="utf-8"), data_root=data_root)
    _print_cost(cfg)
    if dry_run or cfg.dry_run:
        runner.prepare()
        pending = [r for r, s in runner.status().items() if s != "complete"]
        typer.echo(f"\ndry run: {len(runner.plan())} runs planned, {len(pending)} not complete; "
                   f"populations written to {runner.exp_dir / 'populations'}")
        return
    paid = any(m.provider != "mock" for m in cfg.model_configs().values())
    if paid and not yes:
        typer.confirm("\nProceed with these API calls?", abort=True)
    try:
        results = asyncio.run(runner.run(only=only))
    except ConfigFrozenError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(2)
    failed = [r for r, s in results.items() if s != "complete"]
    typer.echo(f"\n{len(results) - len(failed)}/{len(results)} runs complete; data in {runner.exp_dir}")
    if failed:
        typer.secho(f"failed: {', '.join(failed)}", fg="red")
        raise typer.Exit(1)


def _find_run(run_id: str, data_root: Path, experiment: Optional[str]) -> Path:
    base = data_root / experiment if experiment else data_root
    matches = sorted(base.glob(f"runs/{run_id}")) if experiment else sorted(base.glob(f"*/runs/{run_id}"))
    if not matches:
        typer.secho(f"run {run_id} not found under {base}", fg="red", err=True)
        raise typer.Exit(2)
    if len(matches) > 1:
        typer.secho(f"run {run_id} exists in several experiments; pass --experiment", fg="red", err=True)
        raise typer.Exit(2)
    return matches[0].parent.parent


@app.command()
def resume(
    run_id: str,
    experiment: Optional[str] = typer.Option(None, help="Experiment ID containing the run."),
    data_root: Path = typer.Option(Path("data/experiments"), help="Root directory holding experiment folders."),
    yes: bool = typer.Option(False, "--yes", "-y"),
):
    """Resume an interrupted run from its latest checkpoint (use `run` to resume all)."""
    from .config import load_config
    from .simulation.experiment import ExperimentRunner

    exp_dir = _find_run(run_id, data_root, experiment)
    cfg = load_config(exp_dir / "config.yaml")
    runner = ExperimentRunner(cfg, data_root=exp_dir.parent)
    paid = any(m.provider != "mock" for m in cfg.model_configs().values())
    if paid and not yes:
        typer.confirm(f"Resume {run_id} with API calls?", abort=True)
    results = asyncio.run(runner.run(only=[run_id]))
    if results.get(run_id) != "complete":
        raise typer.Exit(1)
    typer.secho(f"{run_id} complete", fg="green")


@app.command()
def status(experiment: str, data_root: Path = typer.Option(Path("data/experiments"))):
    """Show which runs of an experiment are complete, incomplete, failed or pending."""
    from .analysis.data import experiment_config, resolve_experiment_dir
    from .simulation.experiment import ExperimentRunner

    exp_dir = resolve_experiment_dir(experiment, data_root)
    runner = ExperimentRunner(experiment_config(exp_dir), data_root=exp_dir.parent)
    for run_id, s in runner.status().items():
        typer.echo(f"{s:11s} {run_id}")


@app.command()
def analyse(
    experiment: str,
    data_root: Path = typer.Option(Path("data/experiments"), help="Root directory holding experiment folders."),
):
    """Run post-hoc analyses (stance, lexical, semantic, topic, network, transmission)."""
    from .analysis.data import resolve_experiment_dir
    from .analysis.pipeline import AnalysisPipeline

    exp_dir = resolve_experiment_dir(experiment, data_root)
    AnalysisPipeline(exp_dir).run()
    typer.secho(f"analysis written to {exp_dir / 'analysis'}", fg="green")


@app.command()
def report(
    experiment: str,
    data_root: Path = typer.Option(Path("data/experiments")),
    reanalyse: bool = typer.Option(False, help="Recompute the analysis first."),
):
    """Create summary tables, figures and report.md."""
    from .analysis.data import resolve_experiment_dir
    from .analysis.report import build_report

    build_report(resolve_experiment_dir(experiment, data_root), reanalyse=reanalyse)


@app.command("estimate-stance")
def estimate_stance(
    experiment: str,
    provider: str = typer.Option(..., help="mock | openai | anthropic | local"),
    model: str = typer.Option(..., help="Classifier model identifier."),
    window: int = typer.Option(1, help="Rate each agent from its messages in the last WINDOW rounds."),
    data_root: Path = typer.Option(Path("data/experiments")),
    yes: bool = typer.Option(False, "--yes", "-y"),
):
    """Independent stance estimates from agents' messages (post hoc LLM classifier)."""
    from .analysis.data import experiment_config, resolve_experiment_dir
    from .config.schema import ModelConfig
    from .measurement.external_stance import estimate_experiment

    exp_dir = resolve_experiment_dir(experiment, data_root)
    if provider != "mock" and not yes:
        typer.confirm("This makes one classifier call per agent per round. Proceed?", abort=True)
    df = asyncio.run(estimate_experiment(exp_dir, experiment_config(exp_dir), ModelConfig(provider=provider, model=model), window))
    typer.secho(f"{len(df)} rows written to {exp_dir / 'analysis' / 'external_stance.parquet'}", fg="green")


@app.command()
def transcript(
    run_id: str,
    experiment: Optional[str] = typer.Option(None),
    data_root: Path = typer.Option(Path("data/experiments")),
    round: Optional[int] = typer.Option(None, "--round", help="Only this round."),
):
    """Print a run's conversation for manual reading."""
    import pandas as pd

    exp_dir = _find_run(run_id, data_root, experiment)
    run_dir = exp_dir / "runs" / run_id
    msgs = pd.read_parquet(run_dir / "messages.parquet").sort_values("sequence")
    names = dict(pd.read_parquet(run_dir / "agents.parquet")[["agent_id", "public_name"]].values)
    info = json.loads((run_dir / "run.json").read_text())
    typer.echo(f"# {run_id} ({info['status']}, {info['rounds_completed']} rounds)\n")
    for _, m in msgs.iterrows():
        if round is not None and m["round"] != round:
            continue
        to = f" -> {names.get(m['recipient_id'], m['recipient_id'])}" if isinstance(m["recipient_id"], str) else ""
        typer.echo(f"[r{m['round']}] {names[m['sender_id']]}{to} ({m['word_count']} words):\n{m['text']}\n")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
