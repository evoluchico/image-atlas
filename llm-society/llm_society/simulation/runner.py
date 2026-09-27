"""SimulationRunner: executes one run (one environment x population x replicate).

Timeline of a run:

    initialisation -> round 0 probe -> round 1 -> round 1 probe -> ... -> round T -> final probe

Within a round every agent writes exactly one message. Agents in the same
batch generate from the same snapshot of the world concurrently; messages are
routed only after the whole batch has finished, so API completion order can
never determine who influences whom.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..config.schema import ExperimentConfig
from ..environments.base import Environment, OutgoingMessage
from ..environments.registry import get_environment_class
from ..llm.base import GenerationConfig, LLMClient, LLMResponse, Message, generate_with_retry
from ..measurement.probes import parse_probe
from ..measurement.stance import mean_pairwise_distance
from ..prompts.agent import ContextItem, PromptBuilder
from ..prompts.probe import probe_prompt, repair_prompt
from ..seeds import derive_seed, make_rng
from ..storage import schemas
from ..storage.checkpoints import ResponseCache, load_checkpoint, request_key, save_checkpoint
from ..storage.parquet import atomic_write_json, write_records
from .agent import Agent
from .events import EventLogger, utcnow
from .memory import AgentMemory, MemoryPolicy
from .population import Population

# Tokens kept free for the framing text and task instruction around the history.
CONTEXT_OVERHEAD_TOKENS = 400


@dataclass(frozen=True)
class RunSpec:
    experiment_id: str
    environment: str
    population_size: int
    replicate: int

    @property
    def condition_id(self) -> str:
        return f"{self.environment}_N{self.population_size:03d}"

    @property
    def run_id(self) -> str:
        return f"{self.condition_id}_r{self.replicate:03d}"

    @property
    def population_id(self) -> str:
        return f"pop_N{self.population_size:03d}_r{self.replicate:03d}"


class RunFailed(RuntimeError):
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SimulationRunner:
    def __init__(
        self,
        cfg: ExperimentConfig,
        spec: RunSpec,
        population: Population,
        run_dir: Path,
        clients: dict[str, LLMClient],
        semaphore: asyncio.Semaphore | None = None,
        log: Callable[[str], None] = print,
        extra_run_info: dict | None = None,
    ):
        if population.population_size != spec.population_size or population.replicate != spec.replicate:
            raise ValueError("population does not match run spec")
        self.cfg = cfg
        self.spec = spec
        self.population = population
        self.run_dir = Path(run_dir)
        self.clients = clients
        self.semaphore = semaphore or asyncio.Semaphore(cfg.runtime.max_concurrency)
        self.log = log
        self.extra_run_info = extra_run_info or {}
        self.builder = PromptBuilder(cfg)
        self.policy = MemoryPolicy(cfg.memory)
        self.probe_rounds = set(cfg.probe_rounds())
        self.max_rounds = cfg.max_rounds

        n, r, e = spec.population_size, spec.replicate, spec.environment
        self.seeds = {
            "master": cfg.seed,
            "profile": population.seed,
            "network": derive_seed(cfg.seed, "network", n, r),
            "routing": derive_seed(cfg.seed, "routing", n, r, e),
            "ordering": derive_seed(cfg.seed, "ordering", n, r, e),
        }
        self.agents: dict[str, Agent] = {
            p.agent_id: Agent.create(p, self.builder.system_prompt(p)) for p in population.profiles
        }
        self.agent_order = [p.agent_id for p in population.profiles]
        self.names = {p.agent_id: p.public_name for p in population.profiles}
        self.env: Environment = get_environment_class(e)(
            cfg, make_rng(cfg.seed, "routing", n, r, e), make_rng(cfg.seed, "ordering", n, r, e)
        )
        self.events = EventLogger(spec.run_id, self.run_dir, cfg.runtime.save_raw)
        self.messages: dict[str, dict] = {}
        self.round_completed = -1
        self.next_sequence = 0
        self.distance_history: dict[int, float] = {}
        self.stable_rounds = 0
        self.stopped_early = False
        self.start_time: str | None = None
        self.status = "pending"

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def run(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = load_checkpoint(self.run_dir)
        if ckpt and ckpt.get("status") == "complete":
            self.log(f"[{self.spec.run_id}] already complete; skipping")
            return json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
        if ckpt:
            self._restore(ckpt)
            self.log(f"[{self.spec.run_id}] resuming after round {self.round_completed}")
        else:
            self._initialise()
        self.cache = ResponseCache(self.run_dir)
        self.status = "running"
        self._write_run_info()
        try:
            if self.round_completed < 0:
                await self._probe_round(0)
                self.round_completed = 0
                self._checkpoint()
            while self.round_completed < self.max_rounds and not self.stopped_early:
                t = self.round_completed + 1
                await self._play_round(t)
                if t in self.probe_rounds:
                    await self._probe_round(t)
                self.round_completed = t
                self._update_stopping(t)
                self._checkpoint()
                self.log(f"[{self.spec.run_id}] round {t}/{self.max_rounds} done")
        except Exception as exc:
            self.status = "failed"
            self._write_run_info(error=f"{type(exc).__name__}: {exc}")
            raise RunFailed(f"{self.spec.run_id}: {exc}") from exc
        self.status = "complete"
        self._checkpoint()
        return self._write_run_info()

    def _initialise(self) -> None:
        self.start_time = utcnow()
        net_rng = make_rng(self.cfg.seed, "network", self.spec.population_size, self.spec.replicate)
        self.env.initialise(self.agent_order, net_rng)
        self._write_static_tables()

    def _write_static_tables(self) -> None:
        agents_rows = []
        for aid in self.agent_order:
            a = self.agents[aid]
            agents_rows.append(
                {
                    "run_id": self.spec.run_id,
                    "agent_id": aid,
                    "agent_index": a.profile.agent_index,
                    "public_name": a.profile.public_name,
                    "initial_latent_position": a.profile.latent_position,
                    "initial_stance_vector": json.dumps(a.profile.stance),
                    "model_assignment": a.model_key,
                    "model": self.clients[a.model_key].model,
                    "system_prompt": a.system_prompt,
                }
            )
        write_records(self.run_dir / "agents.parquet", agents_rows, schemas.AGENTS)
        edges = self.env.network_edges() or []
        write_records(
            self.run_dir / "network_edges.parquet",
            [{"run_id": self.spec.run_id, "source": u, "target": v} for u, v in edges],
            schemas.NETWORK_EDGES,
        )

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
    def _checkpoint(self) -> None:
        self.events.flush(self.round_completed)
        state = {
            "status": self.status,
            "run_id": self.spec.run_id,
            "round_completed": self.round_completed,
            "next_sequence": self.next_sequence,
            "start_time": self.start_time,
            "environment": self.env.state_dict(),
            "memories": {aid: a.memory.to_dict() for aid, a in self.agents.items()},
            "distance_history": self.distance_history,
            "stable_rounds": self.stable_rounds,
            "stopped_early": self.stopped_early,
            "saved_at": utcnow(),
        }
        save_checkpoint(self.run_dir, state)
        if hasattr(self, "cache"):
            self.cache.entries.clear()

    def _restore(self, ckpt: dict) -> None:
        self.round_completed = ckpt["round_completed"]
        self.next_sequence = ckpt["next_sequence"]
        self.start_time = ckpt["start_time"]
        self.env.load_state_dict(ckpt["environment"])
        for aid, mem in ckpt["memories"].items():
            self.agents[aid].memory = AgentMemory.from_dict(mem)
        self.distance_history = {int(k): v for k, v in ckpt.get("distance_history", {}).items()}
        self.stable_rounds = ckpt.get("stable_rounds", 0)
        self.stopped_early = ckpt.get("stopped_early", False)
        self.events.restore(self.round_completed)
        self.messages = {m["message_id"]: m for m in self.events.records["messages"]}

    def _write_run_info(self, error: str | None = None) -> dict:
        model_keys = sorted({a.model_key for a in self.agents.values()})
        info = {
            "run_id": self.spec.run_id,
            "experiment_id": self.spec.experiment_id,
            "condition_id": self.spec.condition_id,
            "replicate": self.spec.replicate,
            "population_id": self.spec.population_id,
            "seed": self.seeds["routing"],
            "seeds": self.seeds,
            "environment_type": self.spec.environment,
            "environment": self.env.describe(),
            "population_size": self.spec.population_size,
            "num_rounds": self.max_rounds,
            "rounds_completed": max(self.round_completed, 0),
            "stopped_early": self.stopped_early,
            "model": ",".join(self.clients[k].model for k in model_keys),
            "models": {k: self.clients[k].describe() for k in model_keys},
            "model_parameters": {
                "generation": self.cfg.generation.model_dump(),
                "measurement": self.cfg.measurement.model_dump(),
                "memory": self.cfg.memory.model_dump(),
            },
            "topic_id": self.cfg.topic.id,
            "start_time": self.start_time,
            "end_time": utcnow() if self.status in ("complete", "failed") else None,
            "status": self.status,
            "error": error,
            "n_messages": len(self.events.records["messages"]),
            **self.extra_run_info,
        }
        atomic_write_json(self.run_dir / "run.json", info)
        return info

    # ------------------------------------------------------------------
    # context & LLM calls
    # ------------------------------------------------------------------
    def _item(self, mid: str) -> ContextItem:
        m = self.messages[mid]
        rid = m["recipient_id"]
        return ContextItem(
            message_id=mid,
            round=m["round"],
            sender_id=m["sender_id"],
            sender_name=self.names[m["sender_id"]],
            recipient_id=rid,
            recipient_name=self.names.get(rid) if rid else None,
            channel=m["channel"],
            text=m["text"],
        )

    def _context(self, agent: Agent) -> tuple[list[ContextItem], int, int]:
        framing = self.env.framing
        budget = (
            self.cfg.memory.max_context_tokens
            - self.builder.estimate_tokens(agent.system_prompt)
            - CONTEXT_OVERHEAD_TOKENS
        )

        def cost(mid: str) -> int:
            return self.builder.estimate_tokens(self.builder.render_item(self._item(mid), agent.agent_id, framing))

        ids = self.policy.select(agent.memory, lambda mid: self.messages[mid]["sequence"], cost, max(budget, 0))
        items = [self._item(mid) for mid in ids]
        tokens = sum(cost(mid) for mid in ids)
        return items, len(set(agent.memory.available())), tokens

    async def _call(
        self, agent: Agent, phase: str, round_id: int, system: str, user: str, gen: GenerationConfig, attempt: int = 0
    ) -> tuple[LLMResponse, bool]:
        client = self.clients[agent.model_key]
        key = request_key(
            {
                "run": self.spec.run_id,
                "phase": phase,
                "round": round_id,
                "agent": agent.agent_id,
                "attempt": attempt,
                "model": client.model,
                "system": system,
                "user": user,
                "gen": asdict(gen),
            }
        )
        cached = self.cache.get(key)
        if cached is not None:
            resp, from_cache = LLMResponse.from_dict(cached), True
        else:
            def on_retry(n, delay, exc):
                self.log(f"[{self.spec.run_id}] {agent.agent_id} {phase} r{round_id}: retry {n} in {delay:.1f}s ({exc})")

            async with self.semaphore:
                resp = await generate_with_retry(
                    client,
                    system,
                    [Message("user", user)],
                    gen,
                    max_retries=self.cfg.runtime.max_retries,
                    backoff_base_s=self.cfg.runtime.backoff_base_s,
                    backoff_max_s=self.cfg.runtime.backoff_max_s,
                    on_retry=on_retry,
                )
            self.cache.put(key, resp.to_dict())
            from_cache = False
        self.events.add_raw(
            {
                "run_id": self.spec.run_id,
                "round": round_id,
                "agent_id": agent.agent_id,
                "phase": phase,
                "attempt": attempt,
                "from_cache": from_cache,
                "system_prompt": system,
                "context": user,
                "generation_config": asdict(gen),
                "response": resp.to_dict(),
            }
        )
        return resp, from_cache

    def _gen_config(self, phase: str) -> GenerationConfig:
        if phase == "generation":
            g = self.cfg.generation
            return GenerationConfig(g.temperature, g.max_output_tokens, seed=g.seed, top_p=g.top_p)
        m = self.cfg.measurement
        return GenerationConfig(m.temperature, m.max_output_tokens, seed=self.cfg.generation.seed)

    # ------------------------------------------------------------------
    # conversational rounds
    # ------------------------------------------------------------------
    async def _generate_message(self, aid: str, t: int, recipient: str | None) -> dict:
        agent = self.agents[aid]
        items, n_available, ctx_tokens = self._context(agent)
        first = not agent.memory.own_messages
        user = self.builder.generation_prompt(
            items, aid, self.env.framing, t, self.names.get(recipient) if recipient else None, first, self.names[aid]
        )
        gen = self._gen_config("generation")
        resp, from_cache = await self._call(agent, "generation", t, agent.system_prompt, user, gen)
        empty_retries = 0
        while not resp.text.strip() and resp.finish_reason != "refusal" and empty_retries < self.cfg.runtime.empty_response_retries:
            empty_retries += 1
            resp, from_cache = await self._call(agent, "generation", t, agent.system_prompt, user, gen, attempt=empty_retries)
        text = resp.text.strip()
        event_id = f"g_r{t:03d}_{aid}"
        mid = f"m_r{t:03d}_{aid}"
        words = len(text.split())
        record = {
            "run_id": self.spec.run_id,
            "message_id": mid,
            "round": t,
            "sequence": None,
            "sender_id": aid,
            "recipient_id": recipient,
            "channel": self.env.channel,
            "text": text,
            "timestamp": utcnow(),
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "word_count": words,
            "over_word_limit": words > self.cfg.generation.message_word_limit,
            "finish_reason": resp.finish_reason,
            "model": resp.model,
            "empty_retries": empty_retries,
            "generation_event_id": event_id,
        }
        event = self._event_row(event_id, t, aid, "generation", recipient, n_available, items, ctx_tokens, agent.system_prompt + user, resp, from_cache)
        exposures = [self._exposure_row(t, aid, i.message_id, "generation", event_id) for i in items]
        return {"outgoing": OutgoingMessage(mid, t, aid, recipient, text), "record": record, "event": event, "exposures": exposures}

    def _event_row(self, event_id, t, aid, phase, recipient, n_available, items, ctx_tokens, prompt, resp, from_cache) -> dict:
        return {
            "run_id": self.spec.run_id,
            "event_id": event_id,
            "round": t,
            "agent_id": aid,
            "phase": phase,
            "recipient_id": recipient,
            "n_available": n_available,
            "n_visible": len(items),
            "context_tokens_est": ctx_tokens,
            "prompt_sha256": _sha(prompt),
            "model": resp.model,
            "notes": ",".join(resp.notes) if resp.notes else None,
            "from_cache": from_cache,
            "timestamp": utcnow(),
        }

    def _exposure_row(self, t, aid, mid, phase, event_id) -> dict:
        return {"run_id": self.spec.run_id, "round": t, "agent_id": aid, "message_id": mid, "phase": phase, "event_id": event_id}

    async def _gather(self, coros) -> list:
        # Let every call finish (successes land in the response cache) before
        # surfacing the first failure.
        results = await asyncio.gather(*coros, return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                raise r
        return results

    async def _play_round(self, t: int) -> None:
        recipients = self.env.select_recipients(t)
        public = self.env.channel == "forum"
        for batch in self.env.plan_round(t):
            results = await self._gather([self._generate_message(aid, t, recipients[aid]) for aid in batch])
            # Log in agent order, independent of completion order.
            for r in results:
                self.events.add("generation_events", [r["event"]])
                self.events.add("exposures", r["exposures"])
            routing = self.env.route_messages([r["outgoing"] for r in results], t)
            by_id = {r["record"]["message_id"]: r["record"] for r in results}
            ordered = []
            for mid in routing.ordered_ids:
                rec = by_id[mid]
                rec["sequence"] = self.next_sequence
                self.next_sequence += 1
                self.messages[mid] = rec
                ordered.append(rec)
                self.agents[rec["sender_id"]].memory.add_own(mid)
            self.events.add("messages", ordered)
            for recipient_id, mid in routing.deliveries:
                self.agents[recipient_id].memory.receive(mid, public)

    # ------------------------------------------------------------------
    # private probes
    # ------------------------------------------------------------------
    async def _probe_agent(self, aid: str, t: int) -> dict:
        agent = self.agents[aid]
        items, n_available, ctx_tokens = self._context(agent)
        user = probe_prompt(self.builder, items, aid, self.env.framing, self.names[aid])
        gen = self._gen_config("probe")
        resp, from_cache = await self._call(agent, "probe", t, agent.system_prompt, user, gen)
        dims = self.cfg.stance_dimensions
        values, status = parse_probe(resp.text, dims)
        raw_text = resp.text
        if values is None and self.cfg.measurement.repair_malformed and resp.text.strip():
            fix, _ = await self._call(
                agent, "probe_repair", t, "You convert text into JSON.", repair_prompt(self.cfg, resp.text), gen
            )
            values, fix_status = parse_probe(fix.text, dims)
            status = "repaired" if values is not None else "failed"
            raw_text = resp.text + "\n\n[repair response]\n" + fix.text
        event_id = f"p_r{t:03d}_{aid}"
        rows = [
            {
                "run_id": self.spec.run_id,
                "round": t,
                "agent_id": aid,
                "stance_dimension": d,
                "value": float(values[d]) if values is not None else float("nan"),
                "parse_status": status,
                "raw_probe_response": raw_text,
                "event_id": event_id,
            }
            for d in dims
        ]
        event = self._event_row(event_id, t, aid, "probe", None, n_available, items, ctx_tokens, agent.system_prompt + user, resp, from_cache)
        exposures = [self._exposure_row(t, aid, i.message_id, "probe", event_id) for i in items]
        return {"rows": rows, "event": event, "exposures": exposures}

    async def _probe_round(self, t: int) -> None:
        results = await self._gather([self._probe_agent(aid, t) for aid in self.agent_order])
        for r in results:
            self.events.add("probes", r["rows"])
            self.events.add("generation_events", [r["event"]])
            self.events.add("exposures", r["exposures"])

    # ------------------------------------------------------------------
    # optional convergence stopping
    # ------------------------------------------------------------------
    def _update_stopping(self, t: int) -> None:
        if self.cfg.stopping.type != "convergence" or t not in self.probe_rounds:
            return
        dims = self.cfg.stance_dimensions
        rows = [r for r in self.events.records["probes"] if r["round"] == t]
        by_agent: dict[str, dict[str, float]] = {}
        for r in rows:
            by_agent.setdefault(r["agent_id"], {})[r["stance_dimension"]] = r["value"]
        mat = np.array(
            [[(v[d] - dims[d].min) / (dims[d].max - dims[d].min) for d in dims] for v in by_agent.values()], float
        )
        mat = mat[~np.isnan(mat).any(axis=1)]
        self.distance_history[t] = mean_pairwise_distance(mat)
        prev = self.distance_history.get(t - 1)
        if prev is None or prev == 0:
            return
        change = abs(self.distance_history[t] - prev) / prev
        self.stable_rounds = self.stable_rounds + 1 if change < self.cfg.stopping.threshold else 0
        if self.stable_rounds >= self.cfg.stopping.consecutive_rounds:
            self.stopped_early = True
            self.log(f"[{self.spec.run_id}] convergence stopping rule met at round {t}")
