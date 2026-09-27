import networkx as nx
import numpy as np
import pandas as pd
import pytest

from conftest import run_experiment
from llm_society.environments.network import edge_probability, generate_er_graph


def load(runner, env):
    d = runner.exp_dir / "runs" / f"{env}_N008_r000"
    return {t: pd.read_parquet(d / f"{t}.parquet") for t in ["messages", "exposures", "probes", "agents", "network_edges", "generation_events"]}


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    from conftest import base_dict
    from llm_society.config import config_from_dict

    d = base_dict()
    d["controls"]["no_interaction"] = True
    cfg = config_from_dict(d)
    runner, results = run_experiment(cfg, tmp_path_factory.mktemp("data"))
    assert set(results.values()) == {"complete"}
    return cfg, runner


def test_forum_every_agent_can_receive_public_messages(experiment):
    cfg, runner = experiment
    t = load(runner, "forum")
    msgs, exp = t["messages"], t["exposures"]
    r1 = set(msgs[msgs["round"] == 1]["message_id"])
    for agent in t["agents"]["agent_id"]:
        seen = set(exp[(exp["round"] == 2) & (exp["phase"] == "generation") & (exp["agent_id"] == agent)]["message_id"])
        assert r1 <= seen  # all round-1 posts (incl. own) fit in the token budget
    assert msgs["recipient_id"].isna().all()
    assert (msgs["channel"] == "forum").all()


def test_open_dyadic_never_sends_to_self(experiment):
    cfg, runner = experiment
    m = load(runner, "open_dyadic")["messages"]
    assert m["recipient_id"].notna().all()
    assert (m["recipient_id"] != m["sender_id"]).all()


def test_network_messages_only_to_neighbours(experiment):
    cfg, runner = experiment
    t = load(runner, "erdos_renyi")
    edges = {frozenset(e) for e in t["network_edges"][["source", "target"]].itertuples(index=False)}
    assert edges
    for s, r in t["messages"][["sender_id", "recipient_id"]].itertuples(index=False):
        assert frozenset((s, r)) in edges


def test_dyadic_agents_only_see_own_and_received(experiment):
    cfg, runner = experiment
    for env in ["open_dyadic", "erdos_renyi", "no_interaction"]:
        t = load(runner, env)
        m = t["messages"].set_index("message_id")
        for row in t["exposures"].itertuples():
            msg = m.loc[row.message_id]
            assert msg["sender_id"] == row.agent_id or (env != "no_interaction" and msg["recipient_id"] == row.agent_id)


@pytest.mark.parametrize("env", ["forum", "open_dyadic", "erdos_renyi", "no_interaction"])
def test_exactly_n_messages_per_round(experiment, env):
    cfg, runner = experiment
    m = load(runner, env)["messages"]
    counts = m.groupby("round").size()
    assert list(counts.index) == list(range(1, cfg.rounds + 1))
    assert (counts == 8).all()
    assert m.groupby(["round", "sender_id"]).size().max() == 1


@pytest.mark.parametrize("env", ["forum", "open_dyadic", "erdos_renyi"])
def test_synchrony_no_same_round_visibility(experiment, env):
    cfg, runner = experiment
    t = load(runner, env)
    rounds = t["messages"].set_index("message_id")["round"]
    gen = t["exposures"][t["exposures"]["phase"] == "generation"]
    assert (gen["message_id"].map(rounds) < gen["round"]).all()
    probe = t["exposures"][t["exposures"]["phase"] == "probe"]
    assert (probe["message_id"].map(rounds) <= probe["round"]).all()


def test_asynchronous_forum_sees_same_round_posts(make_cfg, tmp_path):
    cfg = make_cfg(environments=["forum"], forum={"scheduler": "asynchronous"}, rounds=2)
    runner, _ = run_experiment(cfg, tmp_path)
    t = load(runner, "forum")
    rounds = t["messages"].set_index("message_id")["round"]
    gen = t["exposures"][t["exposures"]["phase"] == "generation"]
    assert (gen["message_id"].map(rounds) == gen["round"]).any()
    seq = t["messages"].set_index("message_id")["sequence"]
    own_seq = t["messages"].set_index("generation_event_id")["sequence"]
    assert (gen["message_id"].map(seq) < gen["event_id"].map(own_seq)).all()


def test_exposure_matching_reduces_forum_exposure(make_cfg, tmp_path):
    cfg = make_cfg(environments=["forum"], population_sizes=[20], exposure_matching={"enabled": True, "posts_per_round": 1.0})
    runner, _ = run_experiment(cfg, tmp_path)
    d = runner.exp_dir / "runs" / "forum_N020_r000"
    exp = pd.read_parquet(d / "exposures.parquet")
    msgs = pd.read_parquet(d / "messages.parquet").set_index("message_id")
    gen = exp[(exp["phase"] == "generation") & (exp["round"] == 2)]
    others = gen[gen["message_id"].map(msgs["sender_id"]) != gen["agent_id"]]
    assert others.groupby("agent_id").size().reindex(msgs["sender_id"].unique(), fill_value=0).mean() < 4


def test_er_edge_probability_scales_with_n(make_cfg):
    cfg = make_cfg()
    assert edge_probability(10, 6) == pytest.approx(6 / 9)
    assert edge_probability(100, 6) == pytest.approx(6 / 99)
    rng = np.random.default_rng(0)
    degs = []
    for _ in range(20):
        g, _ = generate_er_graph(100, cfg.network, rng)
        assert nx.is_connected(g)
        degs.append(2 * g.number_of_edges() / 100)
    assert 5.0 < np.mean(degs) < 7.0
