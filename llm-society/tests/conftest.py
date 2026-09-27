import asyncio
from pathlib import Path

import pytest
import yaml

from llm_society.config import config_from_dict

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"


def base_dict() -> dict:
    d = yaml.safe_load((EXPERIMENTS / "climate_consensus_dev.yaml").read_text())
    d["experiment"]["name"] = "test_exp"
    d["population_sizes"] = [8]
    d["replicates"] = 1
    d["rounds"] = 3
    d["controls"]["no_interaction"] = False
    d["runtime"] = {"max_concurrency": 32, "max_retries": 0}
    d["analysis"]["bootstrap_samples"] = 200
    return d


@pytest.fixture
def make_cfg():
    def _make(**overrides):
        d = base_dict()
        for key, value in overrides.items():
            target = d
            parts = key.split("__")
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = value
        return config_from_dict(d)

    return _make


def run_experiment(cfg, root, clients=None):
    from llm_society.simulation.experiment import ExperimentRunner

    runner = ExperimentRunner(cfg, data_root=root, clients=clients, log=lambda s: None)
    results = asyncio.run(runner.run())
    return runner, results
