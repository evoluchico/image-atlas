"""Spec §64: N=10, 3 rounds, 1 replicate, all three environments, mock LLM,
through the entire analysis pipeline and report."""

import pandas as pd
import yaml
from typer.testing import CliRunner

from conftest import EXPERIMENTS, run_experiment
from llm_society.analysis.pipeline import AnalysisPipeline
from llm_society.analysis.report import build_report
from llm_society.cli import app
from llm_society.config import config_from_dict


def test_full_pipeline(make_cfg, tmp_path):
    cfg = make_cfg(population_sizes=[10], rounds=3, replicates=1, controls={"no_interaction": True})
    runner, results = run_experiment(cfg, tmp_path)
    assert list(results.values()) == ["complete"] * 4
    metrics = AnalysisPipeline(runner.exp_dir, log=lambda s: None).run()
    names = set(metrics["metric_name"])
    for m in ["stance_distance", "convergence_ratio", "centroid_displacement", "cluster_count", "lexical_jsd",
              "jaccard_top20", "tfidf_cosine", "semantic_distance", "topic_jsd", "topic_entropy",
              "neighbour_distance", "morans_i", "interaction_reciprocity", "adoption_rate_exposed"]:
        assert m in names, m
    report = build_report(runner.exp_dir, log=lambda s: None)
    assert report.exists()
    figs = {p.name for p in (runner.exp_dir / "report" / "figures").glob("*.png")}
    assert {"fig1_stance_convergence.png", "fig5_final_matrix.png", "fig7_network_snapshots.png"} <= figs
    summary = pd.read_csv(runner.exp_dir / "analysis" / "run_summary.csv")
    assert len(summary) == 4
    manifest = (runner.exp_dir / "manifest.json").read_text()
    assert "master_seed" in manifest and "package_versions" in manifest


def test_other_topic_through_configuration_only(tmp_path):
    d = yaml.safe_load((EXPERIMENTS / "ai_society_example.yaml").read_text())
    d.update(population_sizes=[8], replicates=1, rounds=2)
    cfg = config_from_dict(d)
    runner, results = run_experiment(cfg, tmp_path)
    assert set(results.values()) == {"complete"}
    probes = pd.read_parquet(runner.exp_dir / "runs" / "forum_N008_r000" / "probes.parquet")
    assert set(probes["stance_dimension"]) == set(d["stance_dimensions"])


def test_cli_run_analyse_report(tmp_path):
    d = yaml.safe_load((EXPERIMENTS / "climate_consensus_dev.yaml").read_text())
    d.update(population_sizes=[8], replicates=1, rounds=2)
    d["experiment"]["output_dir"] = str(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(d))
    cli = CliRunner()
    for args in (["validate", str(cfg_path)], ["estimate-cost", str(cfg_path)], ["run", str(cfg_path)],
                 ["analyse", d["experiment"]["name"], "--data-root", str(tmp_path)],
                 ["report", d["experiment"]["name"], "--data-root", str(tmp_path)],
                 ["transcript", "forum_N008_r000", "--data-root", str(tmp_path)],
                 ["resume", "forum_N008_r000", "--data-root", str(tmp_path)]):
        res = cli.invoke(app, args)
        assert res.exit_code == 0, (args, res.output)
