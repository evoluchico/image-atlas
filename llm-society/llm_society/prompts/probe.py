"""Rendering of the private measurement probe.

The probe is appended to the agent's current context for a one-off call.
Neither the probe nor its answer is ever written back to agent memory.
"""

from __future__ import annotations

from ..config.schema import ExperimentConfig
from .agent import ContextItem, PromptBuilder


def probe_instruction(cfg: ExperimentConfig, template: str) -> str:
    dims = cfg.stance_dimensions
    lines = []
    for name, d in dims.items():
        question = d.description or (d.label or name.replace("_", " "))
        lines.append(f'- {name}: {question} ({d.min:g} = "{d.low_anchor}", {d.max:g} = "{d.high_anchor}")')
    json_lines = ",\n".join(f'  "{name}": <{d.min:g}-{d.max:g}>' for name, d in dims.items())
    mins = {d.min for d in dims.values()}
    maxs = {d.max for d in dims.values()}
    return template.format(
        scale_min=f"{min(mins):g}" if len(mins) == 1 else "the minimum",
        scale_max=f"{max(maxs):g}" if len(maxs) == 1 else "the maximum shown",
        dimension_list="\n".join(lines),
        json_template="{\n" + json_lines + "\n}",
    )


def probe_prompt(
    builder: PromptBuilder, items: list[ContextItem], viewer_id: str, framing: str, viewer_name: str = ""
) -> str:
    return (
        builder.context_block(items, viewer_id, framing, viewer_name)
        + "\n\n"
        + probe_instruction(builder.cfg, builder.probe_template)
    )


def repair_prompt(cfg: ExperimentConfig, raw_text: str) -> str:
    keys = ", ".join(cfg.stance_dimensions)
    return (
        "Convert the following text into a JSON object with exactly these numeric keys: "
        f"{keys}. Use only numbers that appear in the text; use null if a value is missing. "
        "Return only JSON.\n\n---\n" + raw_text
    )
