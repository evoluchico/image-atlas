import json

import pandas as pd

from conftest import run_experiment


def _tables(runner, run_id):
    d = runner.exp_dir / "runs" / run_id
    return {t: pd.read_parquet(d / f"{t}.parquet") for t in ["messages", "probes", "agents", "network_edges", "exposures"]}


def _strip(df):
    return df.drop(columns=[c for c in ("timestamp",) if c in df.columns]).reset_index(drop=True)


def test_same_seed_same_population_network_and_routing(make_cfg, tmp_path):
    cfg = make_cfg()
    a, _ = run_experiment(cfg, tmp_path / "a")
    b, _ = run_experiment(cfg, tmp_path / "b")
    for run_id in ["forum_N008_r000", "open_dyadic_N008_r000", "erdos_renyi_N008_r000"]:
        ta, tb = _tables(a, run_id), _tables(b, run_id)
        for name in ta:
            pd.testing.assert_frame_equal(_strip(ta[name]), _strip(tb[name]))


def test_different_seed_changes_routing(make_cfg, tmp_path):
    a, _ = run_experiment(make_cfg(), tmp_path / "a")
    b, _ = run_experiment(make_cfg(seed=999), tmp_path / "b")
    ra = _tables(a, "open_dyadic_N008_r000")["messages"]["recipient_id"].tolist()
    rb = _tables(b, "open_dyadic_N008_r000")["messages"]["recipient_id"].tolist()
    assert ra != rb


def test_paired_initial_profiles_identical_across_environments(make_cfg, tmp_path):
    cfg = make_cfg(replicates=2, controls={"no_interaction": True})
    runner, _ = run_experiment(cfg, tmp_path)
    for rep in range(2):
        ref = None
        for env in cfg.all_environments():
            agents = _tables(runner, f"{env}_N008_r{rep:03d}")["agents"]
            cols = agents[["agent_id", "initial_latent_position", "initial_stance_vector", "system_prompt"]]
            if ref is None:
                ref = cols
            else:
                pd.testing.assert_frame_equal(ref, cols)
        # round-0 probes identical too: agents start from identical states
        p0 = [
            _tables(runner, f"{env}_N008_r{rep:03d}")["probes"].query("round == 0")[["agent_id", "stance_dimension", "value"]].reset_index(drop=True)
            for env in cfg.all_environments()
        ]
        for p in p0[1:]:
            pd.testing.assert_frame_equal(p0[0], p)
    pops = sorted((runner.exp_dir / "populations").glob("*.json"))
    assert len(pops) == 2
    assert json.loads(pops[0].read_text())["profiles"] != json.loads(pops[1].read_text())["profiles"]
