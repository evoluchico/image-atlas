import asyncio

import pandas as pd
import pytest

from conftest import run_experiment
from llm_society.llm.base import PermanentLLMError
from llm_society.llm.mock_client import MockLLMClient
from llm_society.simulation.experiment import ConfigFrozenError, ExperimentRunner


class CrashingMock(MockLLMClient):
    """Fails permanently once `crash_after` calls have been made."""

    def __init__(self, crash_after):
        super().__init__()
        self.crash_after = crash_after

    async def generate(self, system, messages, cfg):
        if self.calls >= self.crash_after:
            self.calls += 1
            raise PermanentLLMError("simulated outage")
        return await super().generate(system, messages, cfg)


def _data(exp_dir, run_id):
    d = exp_dir / "runs" / run_id
    out = {}
    for t in ["messages", "probes", "exposures", "generation_events"]:
        df = pd.read_parquet(d / f"{t}.parquet")
        out[t] = df.drop(columns=[c for c in ("timestamp", "from_cache") if c in df.columns]).reset_index(drop=True)
    return out


@pytest.mark.parametrize("env", ["forum", "open_dyadic", "erdos_renyi"])
def test_interrupted_run_resumes_without_repeating_completed_rounds(make_cfg, tmp_path, env):
    cfg = make_cfg(environments=[env])
    n = 8
    # Probe r0 (8) + round 1 (8) + probe r1 (8) + 5 calls into round 2.
    crash = CrashingMock(crash_after=3 * n + 5)
    runner = ExperimentRunner(cfg, data_root=tmp_path / "x", clients={"default": crash}, log=lambda s: None)
    results = asyncio.run(runner.run())
    assert results[f"{env}_N008_r000"] == "failed"

    resumed = MockLLMClient()
    runner2 = ExperimentRunner(cfg, data_root=tmp_path / "x", clients={"default": resumed}, log=lambda s: None)
    results = asyncio.run(runner2.run())
    assert results[f"{env}_N008_r000"] == "complete"
    # Completed rounds are not regenerated; the 5 successful round-2 calls are
    # answered from the partial-round cache.
    total = n * (cfg.rounds + len(cfg.probe_rounds()))
    assert resumed.calls == total - (3 * n) - 5

    reference, _ = run_experiment(cfg, tmp_path / "ref")
    a = _data(runner2.exp_dir, f"{env}_N008_r000")
    b = _data(reference.exp_dir, f"{env}_N008_r000")
    for t in a:
        pd.testing.assert_frame_equal(a[t], b[t])


def test_completed_runs_are_skipped(make_cfg, tmp_path):
    cfg = make_cfg(environments=["forum"])
    run_experiment(cfg, tmp_path)
    client = MockLLMClient()
    run_experiment(cfg, tmp_path, clients={"default": client})
    assert client.calls == 0


def test_config_is_frozen(make_cfg, tmp_path):
    run_experiment(make_cfg(environments=["forum"]), tmp_path)
    with pytest.raises(ConfigFrozenError):
        run_experiment(make_cfg(environments=["forum"], rounds=4), tmp_path)
