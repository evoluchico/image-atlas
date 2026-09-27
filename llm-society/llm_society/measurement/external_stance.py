"""Independent (external) stance estimation from agents' discussion messages.

Runs strictly after simulation. A separate classifier reads what an agent
actually *wrote* (not its private probe answers) and rates it on the
configured stance dimensions. Results are stored separately from the
self-reports (`analysis/external_stance.parquet`, same long layout as the
probes table plus `source`), so agreement between the two can be studied.

The default estimator is an LLM classifier using any `LLMClient`; other
estimators (supervised, embedding-based, human annotation imports) only need
to implement `estimate(texts) -> dict | None`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd

from ..config.schema import ExperimentConfig, ModelConfig
from ..llm import GenerationConfig, LLMClient, Message, create_client, generate_with_retry
from ..prompts.probe import probe_instruction
from ..storage.parquet import write_frame
from .probes import parse_probe

CLASSIFIER_SYSTEM = (
    "You are an annotator. You read messages written by one participant in a discussion and "
    "estimate the positions that the messages express. Rate only what the text expresses; "
    "if a dimension is not addressed, give your best estimate from the overall stance."
)


class StanceEstimator(Protocol):
    name: str

    async def estimate(self, texts: list[tuple[int, str]]) -> tuple[dict[str, float] | None, str, str]: ...


class LLMStanceEstimator:
    name = "llm_classifier"

    def __init__(self, cfg: ExperimentConfig, client: LLMClient, max_output_tokens: int = 2000):
        self.cfg = cfg
        self.client = client
        self.max_output_tokens = max_output_tokens

    async def estimate(self, texts):
        history = "\n\n".join(f"[Round {r}] {t}" for r, t in texts)
        instruction = probe_instruction(self.cfg, (
            "Rate the position expressed by the messages above on the following questions, "
            "on a scale from {scale_min} to {scale_max}.\n\n{dimension_list}\n\nReturn only JSON.\n\n{json_template}"
        ))
        user = f"Discussion question: {self.cfg.topic.opening_question}\n\nMessages:\n\n{history}\n\n{instruction}"
        resp = await generate_with_retry(
            self.client, CLASSIFIER_SYSTEM, [Message("user", user)], GenerationConfig(0.0, self.max_output_tokens)
        )
        values, status = parse_probe(resp.text, self.cfg.stance_dimensions)
        return values, status, resp.text


async def estimate_experiment(
    exp_dir: Path,
    cfg: ExperimentConfig,
    model: ModelConfig,
    window: int = 1,
    max_concurrency: int = 8,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Rate each agent at each round from its messages in rounds (t - window, t]."""
    from ..analysis.data import load_runs

    est = LLMStanceEstimator(cfg, create_client(model))
    sem = asyncio.Semaphore(max_concurrency)
    rows = []

    async def one(run_id, agent, t, texts):
        async with sem:
            values, status, raw = await est.estimate(texts)
        for d in cfg.stance_dimensions:
            rows.append({"run_id": run_id, "round": t, "agent_id": agent, "stance_dimension": d,
                         "value": values[d] if values else float("nan"), "parse_status": status,
                         "raw_response": raw, "source": f"{est.name}:{model.provider}/{model.model}"})

    for run in load_runs(exp_dir):
        m = run.messages
        tasks = []
        for t in sorted(m["round"].unique()):
            win = m[(m["round"] > t - window) & (m["round"] <= t)]
            for agent, grp in win.groupby("sender_id"):
                texts = list(zip(grp["round"], grp["text"]))
                tasks.append(one(run.run_id, agent, int(t), texts))
        await asyncio.gather(*tasks)
        log(f"  {run.run_id}: {len(tasks)} agent-round estimates")
    df = pd.DataFrame(rows).sort_values(["run_id", "round", "agent_id", "stance_dimension"])
    write_frame(exp_dir / "analysis" / "external_stance.parquet", df)
    return df
