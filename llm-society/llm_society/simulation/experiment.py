"""ExperimentRunner: plans and executes every run of an experiment.

Directory layout:

    <output_dir>/<experiment_id>/
        config.yaml               # exact file used (frozen)
        config.resolved.json      # resolved config incl. defaults + hash
        manifest.json
        populations/pop_N050_r007.json
        runs/<run_id>/{run.json, checkpoint.json, *.parquet, raw/}
        analysis/, report/        # written by `analyse` / `report`
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable

from ..config.loader import config_hash, dump_config_yaml
from ..config.schema import ExperimentConfig
from ..llm import LLMClient, create_client
from ..storage.checkpoints import load_checkpoint
from ..storage.manifest import write_manifest
from ..storage.parquet import atomic_write_bytes, atomic_write_json
from .events import utcnow
from .population import Population, generate_population
from .runner import RunFailed, RunSpec, SimulationRunner


class ConfigFrozenError(RuntimeError):
    pass


def experiment_dir(cfg: ExperimentConfig, data_root: str | Path | None = None) -> Path:
    return Path(data_root or cfg.experiment.output_dir) / cfg.experiment_id


class ExperimentRunner:
    def __init__(
        self,
        cfg: ExperimentConfig,
        config_text: str | None = None,
        data_root: str | Path | None = None,
        clients: dict[str, LLMClient] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.cfg = cfg
        self.config_text = config_text
        self.exp_dir = experiment_dir(cfg, data_root)
        self._clients = clients
        self.log = log

    # ------------------------------------------------------------------
    def plan(self) -> list[RunSpec]:
        """Runs ordered replicate -> population size -> environment, so each
        paired set of environments completes together."""
        return [
            RunSpec(self.cfg.experiment_id, env, n, rep)
            for rep in range(self.cfg.replicates)
            for n in self.cfg.population_sizes
            for env in self.cfg.all_environments()
        ]

    def run_dir(self, spec: RunSpec) -> Path:
        return self.exp_dir / "runs" / spec.run_id

    def population_path(self, n: int, rep: int) -> Path:
        return self.exp_dir / "populations" / f"pop_N{n:03d}_r{rep:03d}.json"

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        """Freeze the config and generate (or load) the paired populations."""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        resolved = self.exp_dir / "config.resolved.json"
        h = config_hash(self.cfg)
        if resolved.exists():
            prev = json.loads(resolved.read_text(encoding="utf-8"))
            if prev.get("config_hash") != h:
                raise ConfigFrozenError(
                    f"{self.exp_dir} was created with a different configuration (hash "
                    f"{prev.get('config_hash', '?')[:12]} vs {h[:12]}). Parameters of an experiment "
                    "must not change mid-batch: give the new configuration a new experiment.name "
                    "(e.g. bump _v1 to _v2)."
                )
        else:
            atomic_write_json(resolved, {"config_hash": h, "created": utcnow(), "config": self.cfg.model_dump(mode="json")})
            text = self.config_text if self.config_text is not None else dump_config_yaml(self.cfg)
            atomic_write_bytes(self.exp_dir / "config.yaml", text.encode("utf-8"))
        for rep in range(self.cfg.replicates):
            for n in self.cfg.population_sizes:
                self.population(n, rep)

    def population(self, n: int, rep: int) -> Population:
        path = self.population_path(n, rep)
        if path.exists():
            return Population.load(path)
        pop = generate_population(self.cfg, n, rep)
        pop.save(path)
        return pop

    def clients(self) -> dict[str, LLMClient]:
        if self._clients is None:
            self._clients = {k: create_client(m) for k, m in self.cfg.model_configs().items()}
        return self._clients

    def status(self) -> dict[str, str]:
        out = {}
        for spec in self.plan():
            ckpt = load_checkpoint(self.run_dir(spec))
            run_json = self.run_dir(spec) / "run.json"
            if ckpt is None:
                out[spec.run_id] = "pending"
            elif ckpt.get("status") == "complete":
                out[spec.run_id] = "complete"
            elif run_json.exists() and json.loads(run_json.read_text()).get("status") == "failed":
                out[spec.run_id] = "failed"
            else:
                out[spec.run_id] = "incomplete"
        return out

    # ------------------------------------------------------------------
    async def run(self, only: list[str] | None = None) -> dict:
        self.prepare()
        clients = self.clients()
        semaphore = asyncio.Semaphore(self.cfg.runtime.max_concurrency)
        results: dict[str, str] = {}
        started = utcnow()
        specs = [s for s in self.plan() if only is None or s.run_id in only]
        for i, spec in enumerate(specs, 1):
            self.log(f"=== run {i}/{len(specs)}: {spec.run_id}")
            runner = SimulationRunner(
                self.cfg,
                spec,
                self.population(spec.population_size, spec.replicate),
                self.run_dir(spec),
                clients,
                semaphore=semaphore,
                log=self.log,
                extra_run_info={"config_hash": config_hash(self.cfg)},
            )
            try:
                await runner.run()
                results[spec.run_id] = "complete"
            except RunFailed as exc:
                self.log(f"!!! {exc} (checkpoint kept; resume with `llm-society resume {spec.run_id}`)")
                results[spec.run_id] = "failed"
        self.write_manifest(started)
        return results

    def write_manifest(self, started: str | None = None) -> dict:
        status = self.status()
        observed_models = set()
        for spec in self.plan():
            rj = self.run_dir(spec) / "run.json"
            if rj.exists():
                info = json.loads(rj.read_text(encoding="utf-8"))
                observed_models.add(info.get("model"))
        return write_manifest(
            self.exp_dir,
            experiment_id=self.cfg.experiment_id,
            configuration_file="config.yaml",
            config_hash=config_hash(self.cfg),
            model_identifiers={k: m.model_dump(mode="json") for k, m in self.cfg.model_configs().items()},
            models_reported=sorted(m for m in observed_models if m),
            timestamp=utcnow(),
            run_started=started,
            master_seed=self.cfg.seed,
            n_runs_planned=len(status),
            n_runs_successful=sum(v == "complete" for v in status.values()),
            n_runs_failed=sum(v == "failed" for v in status.values()),
            n_runs_incomplete=sum(v in ("incomplete", "pending") for v in status.values()),
            run_status=status,
        )
