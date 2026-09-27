"""Pre-flight estimate of LLM calls and tokens for an experiment.

The estimate is deliberately simple and transparent: each message is assumed
to be close to the word limit, and the context an agent sees at round t is
the number of messages it could have received by then, capped by the memory
policy. Thinking tokens (for models that think) are not included.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config.schema import ExperimentConfig
from .prompts.agent import PromptBuilder
from .prompts.probe import probe_instruction
from .simulation.population import generate_population

TOKENS_PER_WORD = 1.35
PROBE_OUTPUT_TOKENS = 60


@dataclass
class CostEstimate:
    table: pd.DataFrame
    totals: dict


def _visible_messages(cfg: ExperimentConfig, env: str, n: int, t: int) -> float:
    """Expected number of messages available to one agent at the start of round t."""
    past = max(t - 1, 0)
    if env == "forum":
        if cfg.exposure_matching.enabled:
            return past * (1 + cfg.exposure_matching.posts_per_round)
        return past * n
    if env in ("open_dyadic", "erdos_renyi"):
        return past * 2  # own + (on average) one received per round
    return past


def estimate(cfg: ExperimentConfig) -> CostEstimate:
    builder = PromptBuilder(cfg)
    msg_tokens = min(cfg.generation.max_output_tokens, cfg.generation.message_word_limit * TOKENS_PER_WORD)
    item_tokens = msg_tokens + 12  # "[Round t] Participant k: " prefix
    sample = generate_population(cfg, min(cfg.population_sizes), 0).profiles[0]
    sys_tokens = builder.estimate_tokens(builder.system_prompt(sample))
    frame_tokens = builder.estimate_tokens(builder.context_block([], sample.agent_id, "forum")) + 40
    probe_tokens = builder.estimate_tokens(probe_instruction(cfg, builder.probe_template))
    mem = cfg.memory
    if mem.strategy == "token_budget":
        cap = max(0.0, (mem.max_context_tokens - sys_tokens - 400) / item_tokens)
    elif mem.strategy == "recent_messages":
        cap = float(mem.recent_messages)
    else:
        cap = float("inf")
    probe_rounds = cfg.probe_rounds()
    T = cfg.max_rounds

    rows = []
    for env in cfg.all_environments():
        for n in cfg.population_sizes:
            gen_in = sum(
                n * (sys_tokens + frame_tokens + min(cap, _visible_messages(cfg, env, n, t)) * item_tokens)
                for t in range(1, T + 1)
            )
            probe_in = sum(
                n * (sys_tokens + frame_tokens + probe_tokens + min(cap, _visible_messages(cfg, env, n, t + 1)) * item_tokens)
                for t in probe_rounds
            )
            gen_calls = n * T
            probe_calls = n * len(probe_rounds)
            rows.append(
                {
                    "environment": env,
                    "population_size": n,
                    "replicates": cfg.replicates,
                    "generation_calls": gen_calls * cfg.replicates,
                    "probe_calls": probe_calls * cfg.replicates,
                    "input_tokens": (gen_in + probe_in) * cfg.replicates,
                    "output_tokens": (gen_calls * msg_tokens + probe_calls * PROBE_OUTPUT_TOKENS) * cfg.replicates,
                }
            )
    table = pd.DataFrame(rows)
    table[["input_tokens", "output_tokens"]] = table[["input_tokens", "output_tokens"]].round().astype(int)
    totals = {
        "runs": len(cfg.all_environments()) * len(cfg.population_sizes) * cfg.replicates,
        "generation_calls": int(table["generation_calls"].sum()),
        "probe_calls": int(table["probe_calls"].sum()),
        "total_calls": int(table["generation_calls"].sum() + table["probe_calls"].sum()),
        "input_tokens": int(table["input_tokens"].sum()),
        "output_tokens": int(table["output_tokens"].sum()),
    }
    p = cfg.pricing
    if p.input_per_mtok is not None and p.output_per_mtok is not None:
        totals["estimated_usd"] = round(
            totals["input_tokens"] / 1e6 * p.input_per_mtok + totals["output_tokens"] / 1e6 * p.output_per_mtok, 2
        )
    return CostEstimate(table, totals)
