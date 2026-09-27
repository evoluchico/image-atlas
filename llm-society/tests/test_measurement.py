import gzip
import json

import numpy as np
import pandas as pd

from conftest import run_experiment
from llm_society.measurement.probes import parse_probe
from llm_society.measurement.lexical import mean_pairwise_jsd, mattr


def test_probe_prompts_never_enter_conversational_memory(make_cfg, tmp_path):
    cfg = make_cfg()
    runner, _ = run_experiment(cfg, tmp_path)
    for env in cfg.all_environments():
        d = runner.exp_dir / "runs" / f"{env}_N008_r000"
        msg_ids = set(pd.read_parquet(d / "messages.parquet")["message_id"])
        ckpt = json.loads((d / "checkpoint.json").read_text())
        for mem in ckpt["memories"].values():
            ids = mem["own_messages"] + mem["received_messages"] + mem["public_messages_seen"]
            assert set(ids) <= msg_ids
        raw = sorted((d / "raw").glob("*.jsonl.gz"))
        assert raw
        for f in raw:
            for line in gzip.open(f, "rt").read().splitlines():
                rec = json.loads(line)
                if rec["phase"] == "generation":
                    assert "Return only JSON" not in rec["context"]
                    assert "report your" not in rec["context"]
        texts = pd.read_parquet(d / "messages.parquet")["text"]
        assert not texts.str.contains("Return only JSON").any()


def test_probe_rounds_and_values(make_cfg, tmp_path):
    cfg = make_cfg(environments=["forum"])
    runner, _ = run_experiment(cfg, tmp_path)
    p = pd.read_parquet(runner.exp_dir / "runs" / "forum_N008_r000" / "probes.parquet")
    assert sorted(p["round"].unique()) == [0, 1, 2, 3]
    assert (p["parse_status"] == "ok").all()
    assert p["value"].between(0, 100).all()


def test_final_only_probes(make_cfg, tmp_path):
    cfg = make_cfg(environments=["forum"], measurement={"probe_frequency": "final_only"})
    runner, _ = run_experiment(cfg, tmp_path)
    p = pd.read_parquet(runner.exp_dir / "runs" / "forum_N008_r000" / "probes.parquet")
    assert sorted(p["round"].unique()) == [0, 3]


def test_probe_parsing(make_cfg):
    dims = make_cfg().stance_dimensions
    names = list(dims)
    good = json.dumps({n: 40 for n in names})
    assert parse_probe(good, dims) == ({n: 40.0 for n in names}, "ok")
    vals, status = parse_probe("Sure! Here you go:\n```json\n" + good + "\n```", dims)
    assert status == "extracted" and vals[names[0]] == 40
    vals, status = parse_probe(", ".join(f"{n}: 55" for n in names), dims)
    assert status == "regex" and vals[names[-1]] == 55
    vals, status = parse_probe(json.dumps({n: 140 for n in names}), dims)
    assert status == "ok+clamped" and vals[names[0]] == 100
    assert parse_probe("I'd rather not say.", dims) == (None, "failed")
    assert parse_probe(json.dumps({names[0]: 3}), dims)[1] == "failed"


def test_malformed_probe_is_repaired_and_marked(make_cfg, tmp_path):
    from llm_society.llm.mock_client import MockLLMClient

    class Chatty(MockLLMClient):
        async def generate(self, system, messages, cfg):
            resp = await super().generate(system, messages, cfg)
            if "Return only JSON" in messages[-1].content and "Convert the following" not in messages[-1].content:
                resp.text = "My views, roughly: " + resp.text.replace('"', "'").replace(":", " is")
            return resp

    cfg = make_cfg(environments=["forum"], rounds=1)
    runner, _ = run_experiment(cfg, tmp_path, clients={"default": Chatty()})
    p = pd.read_parquet(runner.exp_dir / "runs" / "forum_N008_r000" / "probes.parquet")
    assert set(p["parse_status"]) == {"repaired"}
    assert p["raw_probe_response"].str.contains("My views").all()


def test_jsd_and_mattr():
    P = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1.0]])
    assert mean_pairwise_jsd(P[:2]) == 0
    assert np.isclose(mean_pairwise_jsd(P[1:]), 1.0)
    assert mattr(["a"] * 100) == 1 / 50


def test_memory_token_budget_keeps_recent_and_own():
    from llm_society.config.schema import MemoryConfig
    from llm_society.simulation.memory import AgentMemory, MemoryPolicy

    mem = AgentMemory("A", {})
    mem.add_own("own0")
    for i in range(1, 20):
        mem.receive(f"m{i}", public=True)
    seq = {"own0": 0, **{f"m{i}": i for i in range(1, 20)}}
    chosen = MemoryPolicy(MemoryConfig(strategy="token_budget", reserve_own_messages=1)).select(mem, seq.get, lambda m: 10, 50)
    assert chosen == ["own0", "m16", "m17", "m18", "m19"]
    chosen = MemoryPolicy(MemoryConfig(strategy="recent_messages", recent_messages=3)).select(mem, seq.get, lambda m: 10, 0)
    assert chosen == ["m17", "m18", "m19"]
