"""Column schemas of the tidy tables written for every run.

Explicit Arrow schemas keep column types stable even for empty tables
(e.g. `network_edges` for non-network runs), which keeps concatenation
across runs trivial.
"""

from __future__ import annotations

import pyarrow as pa

RUN_FIELDS = [
    "run_id", "experiment_id", "condition_id", "replicate", "population_id", "seed", "environment_type",
    "population_size", "num_rounds", "rounds_completed", "model", "model_parameters", "topic_id",
    "start_time", "end_time", "status",
]

AGENTS = pa.schema([
    ("run_id", pa.string()),
    ("agent_id", pa.string()),
    ("agent_index", pa.int32()),
    ("public_name", pa.string()),
    ("initial_latent_position", pa.float64()),
    ("initial_stance_vector", pa.string()),  # JSON {dimension: value}
    ("model_assignment", pa.string()),
    ("model", pa.string()),
    ("system_prompt", pa.string()),
])

MESSAGES = pa.schema([
    ("run_id", pa.string()),
    ("message_id", pa.string()),
    ("round", pa.int32()),
    ("sequence", pa.int64()),
    ("sender_id", pa.string()),
    ("recipient_id", pa.string()),
    ("channel", pa.string()),
    ("text", pa.string()),
    ("timestamp", pa.string()),
    ("prompt_tokens", pa.int64()),
    ("completion_tokens", pa.int64()),
    ("word_count", pa.int32()),
    ("over_word_limit", pa.bool_()),
    ("finish_reason", pa.string()),
    ("model", pa.string()),
    ("empty_retries", pa.int32()),
    ("generation_event_id", pa.string()),
])

EXPOSURES = pa.schema([
    ("run_id", pa.string()),
    ("round", pa.int32()),
    ("agent_id", pa.string()),
    ("message_id", pa.string()),
    ("phase", pa.string()),  # "generation" or "probe"
    ("event_id", pa.string()),
])

GENERATION_EVENTS = pa.schema([
    ("run_id", pa.string()),
    ("event_id", pa.string()),
    ("round", pa.int32()),
    ("agent_id", pa.string()),
    ("phase", pa.string()),
    ("recipient_id", pa.string()),
    ("n_available", pa.int32()),
    ("n_visible", pa.int32()),
    ("context_tokens_est", pa.int64()),
    ("prompt_sha256", pa.string()),
    ("model", pa.string()),
    ("notes", pa.string()),
    ("from_cache", pa.bool_()),
    ("timestamp", pa.string()),
])

PROBES = pa.schema([
    ("run_id", pa.string()),
    ("round", pa.int32()),
    ("agent_id", pa.string()),
    ("stance_dimension", pa.string()),
    ("value", pa.float64()),
    ("parse_status", pa.string()),
    ("raw_probe_response", pa.string()),
    ("event_id", pa.string()),
])

NETWORK_EDGES = pa.schema([
    ("run_id", pa.string()),
    ("source", pa.string()),
    ("target", pa.string()),
])

METRICS = pa.schema([
    ("run_id", pa.string()),
    ("round", pa.int32()),
    ("metric_name", pa.string()),
    ("metric_value", pa.float64()),
    ("metric_metadata", pa.string()),
])

TABLES = {
    "agents": AGENTS,
    "messages": MESSAGES,
    "exposures": EXPOSURES,
    "generation_events": GENERATION_EVENTS,
    "probes": PROBES,
    "network_edges": NETWORK_EDGES,
}
