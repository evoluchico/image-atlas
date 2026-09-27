"""Deterministic mock LLM for tests, CI and engine development.

The mock never touches the network. Its output is a pure function of the
prompt, so identical prompts always give identical responses. To make the
analysis pipeline exercise something non-trivial, it implements a toy
bag-of-words opinion dynamic:

* The agent's prior opinion is parsed from the position description in the
  system prompt (verbal or numeric format produced by `PromptBuilder`).
* Every visible message's opinion is inferred from "stance words" drawn from
  a vocabulary graded from adaptation-/cost-oriented (low) to
  mitigation-/urgency-oriented (high).
* The current opinion is a weighted mean of the prior and the observed
  opinions, and the agent writes a message whose stance words encode it.

So the forum (many exposures) converges faster than dyadic conditions, and
the no-interaction control stays put. None of this says anything about real
LLM behaviour; it only ensures the plumbing is tested end to end.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np

from .base import GenerationConfig, LLMResponse, Message

# Stance vocabulary: bin b encodes opinion (b + 0.5) / len(BINS).
STANCE_BINS: list[list[str]] = [
    ["overreaction", "alarmism", "affordability", "sovereignty"],
    ["adaptation", "resilience", "seawalls", "prosperity"],
    ["pragmatism", "innovation", "markets", "competitiveness"],
    ["balance", "tradeoffs", "gradual", "infrastructure"],
    ["efficiency", "incentives", "transition", "cooperation"],
    ["renewables", "emissions", "standards", "investment"],
    ["decarbonisation", "carbon", "regulation", "targets"],
    ["emergency", "urgency", "mobilisation", "phaseout"],
]
WORD_TO_BIN = {w: i for i, ws in enumerate(STANCE_BINS) for w in ws}
FILLER = (
    "i think we should consider the evidence and what it means for communities "
    "people policy future costs benefits long term practical honest debate "
    "clearly important question matters because families jobs energy climate"
).split()

LEVEL_VALUES = {"strongly_low": 10.0, "somewhat_low": 30.0, "balanced": 50.0, "somewhat_high": 70.0, "strongly_high": 90.0}
_NUMERIC_RE = re.compile(r"^- (?P<label>[^:\n]+?)[^:\n]*?:\s*(?P<value>\d+(?:\.\d+)?) on a 0", re.M)
_VERBAL_RE = re.compile(
    r'^- (?P<label>[^(\n]+?) \(from "(?P<low>[^"]+)" to "(?P<high>[^"]+)"\):\s*(?P<phrase>[^\n]+)$', re.M
)
_JSON_KEY_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*<')


def _seed_from(*parts: str) -> int:
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


def parse_prior(system_prompt: str) -> list[float]:
    """Per-dimension prior values (0-100) from a rendered position description."""
    numeric = [float(m.group("value")) for m in _NUMERIC_RE.finditer(system_prompt)]
    if numeric:
        return numeric
    values = []
    for m in _VERBAL_RE.finditer(system_prompt):
        phrase, low, high = m.group("phrase"), m.group("low"), m.group("high")
        if "balanced" in phrase:
            values.append(LEVEL_VALUES["balanced"])
            continue
        side = "high" if f'"{high}"' in phrase else "low"
        strength = "strongly" if "strongly" in phrase else "somewhat"
        values.append(LEVEL_VALUES[f"{strength}_{side}"])
    return values or [50.0]


def word_opinions(text: str) -> list[float]:
    words = re.findall(r"[a-z]+", text.lower())
    return [(WORD_TO_BIN[w] + 0.5) / len(STANCE_BINS) for w in words if w in WORD_TO_BIN]


class MockLLMClient:
    provider = "mock"

    def __init__(self, model: str = "mock-deterministic-v1", social_weight: float = 0.25, **_: Any):
        self.model = model
        # Weight of the prior relative to one observed message (in word units).
        self.prior_weight_words = 30.0
        self.social_weight = social_weight
        self.calls = 0

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "deterministic": True}

    def _opinion(self, system_prompt: str, user_text: str) -> tuple[float, float, list[float]]:
        prior_dims = parse_prior(system_prompt)
        prior = float(np.mean(prior_dims)) / 100.0
        history = "\n".join(re.findall(r"^\[Round \d+\].*$", user_text, re.M))
        observed = word_opinions(history)
        w_prior = self.prior_weight_words
        w_obs = self.social_weight
        current = (w_prior * prior + w_obs * sum(observed)) / (w_prior + w_obs * len(observed))
        return prior, current, prior_dims

    async def generate(
        self, system_prompt: str, messages: list[Message], generation_config: GenerationConfig
    ) -> LLMResponse:
        self.calls += 1
        user_text = "\n".join(m.content for m in messages)
        rng = np.random.default_rng(_seed_from(system_prompt, user_text, str(generation_config.temperature)))
        prior, current, prior_dims = self._opinion(system_prompt, user_text)

        probe_keys = _JSON_KEY_RE.findall(user_text)
        if "Return only JSON" in user_text and probe_keys:
            shift = (current - prior) * 100.0
            vals = {}
            for i, key in enumerate(probe_keys):
                base = prior_dims[i] if i < len(prior_dims) else prior * 100
                noise = 0.0 if not generation_config.temperature else rng.normal(0, 2.0)
                vals[key] = int(round(float(np.clip(base + shift + noise, 0, 100))))
            text = json.dumps(vals)
        elif "Convert the following" in user_text:
            # Repair request: pull "key ... number" pairs out of the text.
            keys = re.search(r"numeric keys: ([^.]+)\.", user_text)
            body = user_text.split("---", 1)[-1]
            vals = {}
            for k in (keys.group(1).split(", ") if keys else []):
                m = re.search(re.escape(k) + r"\D{0,15}?(-?\d+(?:\.\d+)?)", body)
                vals[k] = float(m.group(1)) if m else None
            text = json.dumps(vals)
        else:
            text = self._message(rng, current)

        prompt_tokens = (len(system_prompt) + len(user_text)) // 4
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(text) // 4,
            finish_reason="stop",
            latency_s=0.0,
            request={"model": self.model, "temperature": generation_config.temperature},
            raw={"mock": True, "opinion": round(current, 4)},
        )

    def _message(self, rng: np.random.Generator, opinion: float) -> str:
        n_bins = len(STANCE_BINS)
        centre = opinion * n_bins - 0.5
        words: list[str] = []
        for _ in range(int(rng.integers(35, 60))):
            if rng.random() < 0.35:
                b = int(np.clip(round(centre + rng.normal(0, 0.6)), 0, n_bins - 1))
                words.append(str(rng.choice(STANCE_BINS[b])))
            else:
                words.append(str(rng.choice(FILLER)))
        words[0] = words[0].capitalize()
        return " ".join(words) + "."
