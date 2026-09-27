# LLM Society

A configurable simulation framework for studying consensus, polarisation,
topical convergence and linguistic convergence in small societies of
interacting LLM agents.

The primary experiment is a **3 × 3 factorial design** (communication
environment × population size), with paired initial populations, private
stance probes, a complete exposure log and an optional no-interaction control.

> **Interpretation.** Agents are independent conversational instances of an
> LLM, not simulations of humans. Results answer: *when multiple instances of
> model X are placed in communication environment Y, what collective dynamics
> emerge?*

## Quick start

```bash
cd llm-society
pip install -e ".[dev]"            # + ".[anthropic]" for Claude, ".[stats]" for mixed models

# Whole 3x3 design + no-interaction control on the deterministic mock LLM (no API calls, ~1 min)
llm-society run experiments/climate_consensus_dev.yaml
llm-society report climate_consensus_dev          # analysis + figures + report.md

# The real pilot
llm-society validate      experiments/climate_consensus_v1.yaml
llm-society estimate-cost experiments/climate_consensus_v1.yaml
llm-society run           experiments/climate_consensus_v1.yaml --dry-run   # populations + plan, no calls
llm-society run           experiments/climate_consensus_v1.yaml             # asks before paid calls
```

Output goes to `data/experiments/<experiment name>/` (set by `experiment.output_dir`).

| Command | What it does |
|---|---|
| `validate CONFIG` | Check a config (incl. ER edge probabilities) without API calls |
| `estimate-cost CONFIG` | Calls, input/output tokens and (with `pricing:`) USD per condition |
| `run CONFIG [--dry-run] [--only RUN_ID]` | Run every run; re-running continues incomplete runs and skips complete ones |
| `resume RUN_ID` | Resume one interrupted run from its last checkpoint |
| `status EXPERIMENT` | Complete / incomplete / failed / pending per run |
| `analyse EXPERIMENT` | Post-hoc measures for all completed runs |
| `report EXPERIMENT` | Summary tables, figures and `report/report.md` |
| `estimate-stance EXPERIMENT --provider P --model M` | Independent LLM-classifier stance estimates from agents' messages |
| `transcript RUN_ID [--round t]` | Print a conversation for manual reading |

## Experimental design as implemented

**Environments** (`environments:`): `forum` (public, chronological),
`open_dyadic` (one private message per round to a uniformly random other agent),
`erdos_renyi` (one private message per round to a uniformly random graph
neighbour), plus `no_interaction` (control; add with `controls.no_interaction: true`).

**Rounds.** `round 0 probe → round 1 → probe → … → round T → final probe`.
Every agent writes exactly one message per round, in every environment
(speaking opportunity is held constant). Within a round all agents generate
concurrently from the same snapshot; messages are routed only after the whole
round has been generated, so API completion order never matters. Forum posts
are appended in a seeded random order. `forum.scheduler: asynchronous`
switches the forum to sequential posting in which later posters see earlier
same-round posts.

**Erdős–Rényi network.** `p = k / (N − 1)` with `k = network.target_mean_degree`,
so mean degree does not grow with N; resampled until connected; fixed for the
run; edge list saved per run.

**Paired populations.** For each (N, replicate) one population is generated
from a derived seed and saved to `populations/`; every environment of that
replicate uses exactly the same agents. Positions are generated externally
(`stratified`, `uniform`, `normal`, `bimodal`, `clustered`, `custom_file`):
a latent orientation `z ∈ [-1, 1]` is mapped to each dimension through a
per-dimension `loading`, plus independent noise, so dimensions are correlated
but not perfectly. Positions are then randomly assigned to agent IDs.
Stance dimensions express priorities and trade-offs, never factual claims.

**Seeds.** A master seed derives independent streams for population
(`profile`, N, rep), network (`network`, N, rep), recipient selection and
exposure sampling (`routing`, N, rep, env), forum ordering (`ordering`, …)
and analysis. There is no global random state. Generator states are
checkpointed.

**Private probes.** At baseline and after each probed round every agent is
asked, in a separate call, for its position as JSON (probe temperature from
`measurement.temperature`). The probe sees the agent's current context but is
never written to its memory. Malformed answers are kept verbatim, parsed
deterministically where possible, optionally repaired by a separate call, and
flagged (`parse_status`: `ok`, `extracted`, `regex`, `*+clamped`, `repaired`,
`failed`). `probe_frequency` can be `every_round`, `selected_rounds` or
`final_only` (baseline + final) for the measurement-reactivity check.

**Memory.** `AgentMemory` stores message IDs (own, received, public seen).
`memory.strategy` (`token_budget` by default, also `full` and
`recent_messages`) selects what goes into the prompt, identically across
environments: the agent's `reserve_own_messages` most recent own messages,
then the most recent other messages until `max_context_tokens` (estimated at
`chars_per_token`) is used. Whatever is selected is logged.

**Exposure log.** Every message has a sender, round and global sequence
number; every generation and probe call has an event with the exact list of
message IDs shown (`exposures` table). This reconstructs exactly what every
agent saw before every message it wrote.

## Data

```
data/experiments/<experiment_id>/
    config.yaml  config.resolved.json  manifest.json
    populations/pop_N050_r003.json
    runs/<environment>_N050_r003/
        run.json              # status, seeds, environment description, models, timing
        agents.parquet        # initial latent + stance vector, model, rendered system prompt
        messages.parquet      # sender, recipient (null in forum), channel, text, tokens, word count
        exposures.parquet     # (round, agent, message) shown before each generation/probe call
        generation_events.parquet
        probes.parquet        # long: round, agent, dimension, value, parse_status, raw response
        network_edges.parquet
        metrics.parquet       # written by `analyse`
        checkpoint.json
        raw/round_XXX.jsonl.gz  # system prompt, rendered context, request, raw response per call
    analysis/  report/
```

**Checkpointing.** State is saved after every completed round. Every
completed API call is also appended to `partial_calls.jsonl`, so if a run
crashes mid-round the resumed round (which restores RNG and memory state and
therefore re-issues byte-identical requests) gets the finished calls from the
cache. The test suite checks that an interrupted and resumed run is identical
to an uninterrupted one.

**Frozen configurations.** Once an experiment directory exists, running a
config with a different resolved hash under the same `experiment.name` is
refused. To change a parameter, create `…_v2`.

## Measures (`analyse`)

Metrics are plugins (`llm_society/measurement/metrics.py`,
`register_metric`), each enabled by a flag in `analysis:`. The run is the
unit of replication.

- **Substantive**: mean pairwise stance distance `D_t` (normalised),
  `convergence_ratio = D_t / D_0`, per-dimension SD / MAD / IQR / mean,
  centroid displacement, bimodality coefficient, GMM cluster count by BIC
  (descriptive; k = 1 allowed), within/between cluster distance, minority
  share, persistence of initially extreme agents.
- **Lexical**: NFKC, lower-casing, tokenisation, optional stopword removal and
  lemmatisation (spaCy if installed); raw text is never modified. Mean
  pairwise Jensen–Shannon divergence, Jaccard of top-k words, TF-IDF cosine,
  vocabulary size, TTR, MATTR, entropy, effective vocabulary, share of
  vocabulary used by ≥ 2 agents.
- **Semantic**: agent-round embedding centroids and their mean pairwise cosine
  distance (sentence-transformers if installed, otherwise a TF-IDF + SVD
  embedding; the backend is recorded).
- **Topical**: one topic model (NMF or LDA) fitted on the pooled corpus of
  all runs, then per-agent topic distributions: pairwise topic JSD,
  population topic entropy, topic prevalence, and topic entry/peak/extinction
  (`topic_lifecycle.csv`).
- **Network**: neighbour vs global stance distance (local/global ratio),
  local disagreement, Moran's I, stance assortativity, path-length vs stance
  distance correlation, community-level distance and variance ratio; for
  dyadic runs also the emergent cumulative interaction network (distinct
  pairs, reciprocity, in-degree Gini, repeat share, local/global ratio on it).
- **Transmission**: from the exposure log, `P(adopt word | exposed)` vs
  `P(adopt word | unexposed)` and their ratio, plus adoption events with first
  exposure, exposure count and exposing agents.
- **External stance** (optional, `estimate-stance`): an LLM classifier rates
  what each agent wrote, stored separately from the self-reports.

Summaries: `condition_trajectories.csv` (mean + run-bootstrap CI per
environment × N × round), `run_summary.csv`, `paired_differences.csv`
(within-replicate differences between environments, bootstrap CI over
replicates) and `mixed_models.txt` (`outcome ~ environment * N + round`,
random intercept per run; needs `statsmodels`). The report produces
figures 1–7 of the specification plus semantic and local-vs-global plots.

## Models

`model.provider`: `mock`, `openai`, `anthropic`, `local` (any
OpenAI-compatible server). The engine only uses the `LLMClient` protocol
(`llm/base.py`). Heterogeneous populations use
`population.model_assignment: random | stratified | custom` with named
`population.models`. Transient failures (timeouts, 429, 5xx, overload) are
retried with exponential backoff. Valid but unusual responses are never
regenerated; only empty responses are retried (`runtime.empty_response_retries`),
and refusals are kept as data.

Several current Claude models reject sampling parameters. For these the
adapter does not send `temperature` and records `temperature_not_sent` on
every affected call, which means probe temperature 0 is **not** applied.
They also think by default, so `max_output_tokens` must leave room for
thinking; effort is set through `model.extra_params`. See the comments in
`experiments/climate_consensus_v1.yaml`.

## Decisions where the specification left room

- **What the probe sees.** The probe after round *t* sees the context the
  agent would have at the start of round *t + 1*, so that it includes messages
  delivered at the end of round *t*. Otherwise the final round's messages
  would never be measured.
- **Starting positions are verbal by default** (`prompts.position_format:
  verbal`), e.g. *you lean somewhat toward "prioritise mitigation"*.
  With `numeric`, agents are told the same 0–100 numbers the probe asks for,
  so probes could simply echo the prompt. `numeric` and `both` are available.
- **Agents are told their own public label** ("Other participants know you as
  Participant 12") so they can follow being addressed by name.
- **Exposure matching** delivers each forum post to each other agent with
  probability `posts_per_round / (N − 1)`, so on average each agent receives
  one new post per round, as in the dyadic conditions.
- **The no-interaction control shows agents their own previous
  contributions** (the same memory algorithm), so drift from self-conditioning
  counts as part of the "shared model prior" baseline.
- **Convergence stopping** (off by default) stops once the relative change
  in `D_t` between consecutive rounds has been below `threshold` for
  `consecutive_rounds` rounds.

## Not yet implemented

`memory.strategy: summarised`; recipient rules beyond uniform (the
`RecipientSelector` registry is the extension point); dynamic and non-ER
networks; explicit reply-to fields and attention networks for the forum;
external events (shocks); personas; the influence regression of §71. None
of these needs changes to the simulation engine's structure.

## Layout

```
llm_society/
  config/        schema.py (pydantic, extra keys rejected), loader.py
  simulation/    population.py, agent.py, memory.py, scheduler.py, runner.py, experiment.py, events.py
  environments/  forum.py, dyadic.py, network.py, no_interaction.py, registry.py
  llm/           base.py, mock_client.py, openai_client.py, anthropic_client.py, local_client.py
  prompts/       agent.py, probe.py, templates/*.txt
  measurement/   probes.py, stance.py, lexical.py, semantic.py, topics.py, network.py,
                 transmission.py, external_stance.py, metrics.py (registry)
  analysis/      data.py, pipeline.py, statistics.py, plots.py, report.py
  storage/       schemas.py, parquet.py, checkpoints.py, manifest.py
  cost.py  seeds.py  cli.py
experiments/     climate_consensus_v1.yaml, climate_consensus_dev.yaml, ai_society_example.yaml
tests/           unit tests (spec §63) and end-to-end integration tests (spec §64)
```
