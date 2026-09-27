from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_society.config import load_config
from llm_society.cost import estimate

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"


def test_shipped_configs_validate():
    for f in EXPERIMENTS.glob("*.yaml"):
        cfg = load_config(f)
        assert cfg.stance_dimensions


def test_generation_count_matches_spec_example():
    cfg = load_config(EXPERIMENTS / "climate_consensus_v1.yaml")
    assert estimate(cfg).totals["generation_calls"] == 5 * 12 * 160 * 3 == 28_800


def test_invalid_configs_rejected(make_cfg):
    with pytest.raises(ValidationError):
        make_cfg(environments=["forum", "chatroom"])
    with pytest.raises(ValidationError):
        make_cfg(population_sizes=[5], network={"target_mean_degree": 6})
    with pytest.raises(ValidationError):
        make_cfg(typo_key=1)
    with pytest.raises(ValidationError):
        make_cfg(stopping={"type": "convergence"}, measurement={"probe_frequency": "final_only"})


def test_aliases(make_cfg):
    cfg = make_cfg(environments=["forum", "network_dyadic"], network={"require_connected_graph": False})
    assert cfg.environments == ["forum", "erdos_renyi"]
    assert cfg.network.require_connected is False


def test_anthropic_adapter_omits_sampling_for_models_that_reject_it():
    pytest.importorskip("anthropic")
    from llm_society.llm.anthropic_client import AnthropicClient

    assert AnthropicClient("claude-opus-5", api_key_env=None).supports_temperature is False
    assert AnthropicClient("claude-haiku-4-5").supports_temperature is True
