"""Pydantic schema for a complete experiment configuration.

Everything that defines an experiment (topic, stance dimensions, population
sizes, environments, model, prompts, measurement, analysis) lives here and is
loaded from a single YAML file. Unknown keys are rejected so that a typo can
never silently fall back to a default in a frozen (pre-registered) config.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

# Canonical environment names and accepted aliases. The simulation engine looks
# environments up in `llm_society.environments.registry` by canonical name.
ENVIRONMENT_ALIASES = {
    "forum": "forum",
    "public_forum": "forum",
    "open_dyadic": "open_dyadic",
    "dyadic": "open_dyadic",
    "unrestricted_dyadic": "open_dyadic",
    "erdos_renyi": "erdos_renyi",
    "network": "erdos_renyi",
    "network_dyadic": "erdos_renyi",
    "no_interaction": "no_interaction",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ExperimentMeta(StrictModel):
    name: str
    description: str = ""
    output_dir: str = "data/experiments"

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", v):
            raise ValueError("experiment.name may only contain letters, digits, '_', '-', '.'")
        return v


class TopicConfig(StrictModel):
    id: str
    title: str | None = None
    opening_question: str

    @model_validator(mode="after")
    def _default_title(self) -> "TopicConfig":
        if self.title is None:
            self.title = self.id.replace("_", " ")
        self.opening_question = " ".join(self.opening_question.split())
        return self


class StanceDimension(StrictModel):
    """One numeric attitude dimension.

    `label` is a short human-readable name used in the agent's position
    description; `description` is the full question used by the probe.
    `low_anchor`/`high_anchor` describe what the ends of the scale mean.
    `loading` controls how strongly the latent orientation maps onto this
    dimension during initialisation (sign flips direction).
    """

    min: float = 0.0
    max: float = 100.0
    label: str | None = None
    description: str | None = None
    low_anchor: str = "very low"
    high_anchor: str = "very high"
    loading: float = 1.0

    @model_validator(mode="after")
    def _check_range(self) -> "StanceDimension":
        if self.max <= self.min:
            raise ValueError("stance dimension max must exceed min")
        return self


class NetworkConfig(StrictModel):
    type: Literal["erdos_renyi"] = "erdos_renyi"
    target_mean_degree: float = 6.0
    require_connected: bool = Field(
        default=True, validation_alias=AliasChoices("require_connected", "require_connected_graph")
    )
    max_resample_attempts: int = 10_000
    recipient_selection: str = Field(
        default="uniform_random_neighbor",
        validation_alias=AliasChoices("recipient_selection", "network_recipient_selection"),
    )


class ForumConfig(StrictModel):
    scheduler: Literal["synchronous", "asynchronous"] = Field(
        default="synchronous", validation_alias=AliasChoices("scheduler", "forum_scheduler")
    )


class DyadicConfig(StrictModel):
    recipient_selection: str = Field(
        default="uniform_random",
        validation_alias=AliasChoices("recipient_selection", "dyadic_recipient_selection"),
    )


class ExposureMatchingConfig(StrictModel):
    """Robustness option: forum agents receive only a random subset of posts.

    Each (post, other agent) pair is delivered independently with probability
    `posts_per_round / (N - 1)`, so the expected number of new incoming posts
    per agent per round equals `posts_per_round` (1.0 matches the dyadic
    conditions, where each agent receives one message per round on average).
    """

    enabled: bool = False
    posts_per_round: float = 1.0


class InitialisationConfig(StrictModel):
    method: Literal["stratified", "uniform", "normal", "bimodal", "clustered", "custom_file"] = "stratified"
    noise_sd: float = 0.10
    normal_sd: float = 0.35
    bimodal_separation: float = 0.6
    bimodal_sd: float = 0.15
    n_clusters: int = 3
    cluster_sd: float = 0.08
    custom_file: str | None = None

    @model_validator(mode="after")
    def _custom_needs_file(self) -> "InitialisationConfig":
        if self.method == "custom_file" and not self.custom_file:
            raise ValueError("initialisation.method=custom_file requires initialisation.custom_file")
        return self


class ModelConfig(StrictModel):
    provider: Literal["mock", "openai", "anthropic", "local"] = "mock"
    model: str = "mock-deterministic-v1"
    base_url: str | None = None
    api_key_env: str | None = None
    # None = let the adapter decide (some models reject sampling parameters).
    supports_temperature: bool | None = None
    timeout_s: float = 120.0
    # Passed through verbatim to the provider request body.
    extra_params: dict[str, Any] = Field(default_factory=dict)


class PopulationConfig(StrictModel):
    model_assignment: Literal["homogeneous", "random", "stratified", "custom"] = "homogeneous"
    # For heterogeneous populations: named models and (for `random`) weights.
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    # For `custom`: CSV with columns agent_index, model_key.
    custom_file: str | None = None


class GenerationSettings(StrictModel):
    temperature: float = 0.7
    max_output_tokens: int = 200
    message_word_limit: int = 120
    seed: int | None = None
    top_p: float | None = None


class MeasurementConfig(StrictModel):
    probe_frequency: Literal["every_round", "selected_rounds", "final_only"] = "every_round"
    selected_rounds: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    max_output_tokens: int = 200
    repair_malformed: bool = True


class MemoryConfig(StrictModel):
    strategy: Literal["full", "recent_messages", "token_budget", "summarised"] = "token_budget"
    max_context_tokens: int = 12_000
    recent_messages: int = 20
    # Own most-recent messages kept in context before filling with others.
    reserve_own_messages: int = 2
    chars_per_token: float = 4.0

    @field_validator("strategy")
    @classmethod
    def _not_yet(cls, v: str) -> str:
        if v == "summarised":
            raise ValueError("memory.strategy=summarised is reserved for a future version")
        return v


class AnalysisConfig(StrictModel):
    stance_convergence: bool = True
    lexical_convergence: bool = True
    semantic_convergence: bool = True
    topic_convergence: bool = True
    network_metrics: bool = True
    transmission: bool = True
    lexical_top_k: list[int] = Field(default_factory=lambda: [20, 50, 100])
    lexical_window: int = 1
    remove_stopwords: bool = True
    lemmatise: bool = False
    embedding_backend: Literal["auto", "sentence_transformers", "tfidf_svd"] = "auto"
    embedding_model: str = "all-MiniLM-L6-v2"
    topic_model: Literal["nmf", "lda"] = "nmf"
    n_topics: int = 12
    topic_presence_threshold: float = 0.05
    bootstrap_samples: int = 2000
    max_clusters: int = 6


class ControlsConfig(StrictModel):
    no_interaction: bool = False


class StoppingConfig(StrictModel):
    type: Literal["fixed_rounds", "convergence"] = "fixed_rounds"
    threshold: float = 0.10
    consecutive_rounds: int = 3
    max_rounds: int | None = None


class RuntimeConfig(StrictModel):
    max_concurrency: int = 16
    max_retries: int = 6
    backoff_base_s: float = 2.0
    backoff_max_s: float = 60.0
    empty_response_retries: int = 2
    save_raw: bool = True


class PricingConfig(StrictModel):
    input_per_mtok: float | None = None
    output_per_mtok: float | None = None


class PromptsConfig(StrictModel):
    agent_system_template: str | None = None
    probe_template: str | None = None
    position_format: Literal["verbal", "numeric", "both"] = "verbal"
    participant_label: str = "Participant {n}"


class ExperimentConfig(StrictModel):
    experiment: ExperimentMeta
    topic: TopicConfig
    stance_dimensions: dict[str, StanceDimension]
    population_sizes: list[int] = Field(default_factory=lambda: [10, 50, 100])
    environments: list[str] = Field(default_factory=lambda: ["forum", "open_dyadic", "erdos_renyi"])
    replicates: int = 3
    rounds: int = 12
    seed: int = 12345
    dry_run: bool = False

    network: NetworkConfig = Field(default_factory=NetworkConfig)
    forum: ForumConfig = Field(default_factory=ForumConfig)
    dyadic: DyadicConfig = Field(default_factory=DyadicConfig)
    exposure_matching: ExposureMatchingConfig = Field(default_factory=ExposureMatchingConfig)
    initialisation: InitialisationConfig = Field(default_factory=InitialisationConfig)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    measurement: MeasurementConfig = Field(default_factory=MeasurementConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    controls: ControlsConfig = Field(default_factory=ControlsConfig)
    stopping: StoppingConfig = Field(default_factory=StoppingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)

    @field_validator("stance_dimensions")
    @classmethod
    def _dims_nonempty(cls, v: dict[str, StanceDimension]) -> dict[str, StanceDimension]:
        if not v:
            raise ValueError("at least one stance dimension is required")
        for name in v:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"stance dimension name {name!r} must be an identifier")
        return v

    @field_validator("population_sizes")
    @classmethod
    def _sizes(cls, v: list[int]) -> list[int]:
        if not v or any(n < 2 for n in v):
            raise ValueError("population_sizes must be a non-empty list of integers >= 2")
        if len(set(v)) != len(v):
            raise ValueError("population_sizes contains duplicates")
        return v

    @field_validator("environments")
    @classmethod
    def _envs(cls, v: list[str]) -> list[str]:
        out = []
        for name in v:
            canon = ENVIRONMENT_ALIASES.get(name)
            if canon is None:
                raise ValueError(f"unknown environment {name!r}; known: {sorted(ENVIRONMENT_ALIASES)}")
            if canon in out:
                raise ValueError(f"environment {name!r} listed twice")
            out.append(canon)
        if not out:
            raise ValueError("at least one environment is required")
        return out

    @model_validator(mode="after")
    def _cross_checks(self) -> "ExperimentConfig":
        if self.replicates < 1 or self.rounds < 1:
            raise ValueError("replicates and rounds must be >= 1")
        if "erdos_renyi" in self.environments:
            for n in self.population_sizes:
                if self.network.target_mean_degree > n - 1:
                    raise ValueError(
                        f"network.target_mean_degree={self.network.target_mean_degree} "
                        f"exceeds N-1={n - 1} for population size {n}"
                    )
                if self.network.target_mean_degree <= 0:
                    raise ValueError("network.target_mean_degree must be positive")
        if self.stopping.type == "convergence" and self.measurement.probe_frequency != "every_round":
            raise ValueError("convergence-based stopping requires measurement.probe_frequency=every_round")
        if self.population.model_assignment != "homogeneous" and not self.population.models:
            raise ValueError("heterogeneous model_assignment requires population.models")
        if self.population.model_assignment == "custom" and not self.population.custom_file:
            raise ValueError("population.model_assignment=custom requires population.custom_file")
        for r in self.measurement.selected_rounds:
            if not 0 <= r <= self.rounds:
                raise ValueError(f"measurement.selected_rounds entry {r} outside 0..rounds")
        return self

    # ---- convenience -------------------------------------------------
    @property
    def experiment_id(self) -> str:
        return self.experiment.name

    @property
    def dimension_names(self) -> list[str]:
        return list(self.stance_dimensions)

    @property
    def max_rounds(self) -> int:
        if self.stopping.type == "convergence" and self.stopping.max_rounds:
            return self.stopping.max_rounds
        return self.rounds

    def all_environments(self) -> list[str]:
        envs = list(self.environments)
        if self.controls.no_interaction and "no_interaction" not in envs:
            envs.append("no_interaction")
        return envs

    def probe_rounds(self) -> list[int]:
        last = self.max_rounds
        if self.measurement.probe_frequency == "every_round":
            return list(range(0, last + 1))
        if self.measurement.probe_frequency == "final_only":
            return [0, last]
        return sorted({0, last, *self.measurement.selected_rounds})

    def model_configs(self) -> dict[str, ModelConfig]:
        """All model configurations keyed by assignment key."""
        if self.population.model_assignment == "homogeneous":
            return {"default": self.model}
        return dict(self.population.models)
