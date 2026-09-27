"""Rendering of agent system prompts and conversational context.

All text an agent sees is produced here, from templates, so that prompt
content is separated from simulation logic and can be replaced by config.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

from ..config.schema import ExperimentConfig
from ..simulation.population import AgentProfile


def load_template(name: str) -> str:
    return resources.files("llm_society.prompts").joinpath("templates", name).read_text(encoding="utf-8").strip()


@dataclass
class ContextItem:
    """A message as it appears in an agent's context."""

    message_id: str
    round: int
    sender_id: str
    sender_name: str
    recipient_id: str | None
    recipient_name: str | None
    channel: str
    text: str


def verbal_level(u: float, low: str, high: str) -> str:
    """Map a normalised position u in [0, 1] to a five-level verbal description."""
    if u < 0.2:
        return f'you lean strongly toward "{low}"'
    if u < 0.4:
        return f'you lean somewhat toward "{low}"'
    if u < 0.6:
        return "you are balanced between the two"
    if u < 0.8:
        return f'you lean somewhat toward "{high}"'
    return f'you lean strongly toward "{high}"'


class PromptBuilder:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.system_template = cfg.prompts.agent_system_template or load_template("agent_system.txt")
        self.probe_template = cfg.prompts.probe_template or load_template("probe.txt")
        self.context_templates = {
            "forum": load_template("context_forum.txt"),
            "dyadic": load_template("context_dyadic.txt"),
            "none": load_template("context_none.txt"),
        }

    # ---- system prompt ------------------------------------------------
    def position_description(self, profile: AgentProfile) -> str:
        fmt = self.cfg.prompts.position_format
        lines = []
        for name, dim in self.cfg.stance_dimensions.items():
            value = profile.stance[name]
            label = dim.label or name.replace("_", " ")
            u = (value - dim.min) / (dim.max - dim.min)
            numeric = f"{value:.0f} on a {dim.min:g}-{dim.max:g} scale"
            if fmt == "numeric":
                lines.append(f'- {label}: {numeric} ({dim.min:g} = "{dim.low_anchor}", {dim.max:g} = "{dim.high_anchor}")')
                continue
            line = f'- {label} (from "{dim.low_anchor}" to "{dim.high_anchor}"): {verbal_level(u, dim.low_anchor, dim.high_anchor)}'
            if fmt == "both":
                line += f" ({numeric})"
            lines.append(line)
        return "\n".join(lines)

    def system_prompt(self, profile: AgentProfile) -> str:
        return self.system_template.format(
            topic=self.cfg.topic.title,
            initial_position_description=self.position_description(profile),
            message_word_limit=self.cfg.generation.message_word_limit,
        )

    # ---- context ------------------------------------------------------
    @staticmethod
    def render_item(item: ContextItem, viewer_id: str, framing: str) -> str:
        own = item.sender_id == viewer_id
        if framing == "forum":
            who = "You" if own else item.sender_name
            return f"[Round {item.round}] {who}: {item.text}"
        if framing == "dyadic":
            if own:
                return f"[Round {item.round}] You -> {item.recipient_name}: {item.text}"
            return f"[Round {item.round}] {item.sender_name} -> you: {item.text}"
        return f"[Round {item.round}] {item.text}"

    def history(self, items: list[ContextItem], viewer_id: str, framing: str) -> str:
        if not items:
            return {
                "forum": "(No posts yet.)",
                "dyadic": "(No messages yet.)",
                "none": "(Nothing yet.)",
            }[framing]
        return "\n\n".join(self.render_item(i, viewer_id, framing) for i in items)

    def context_block(self, items: list[ContextItem], viewer_id: str, framing: str, viewer_name: str = "") -> str:
        return self.context_templates[framing].format(
            opening_question=self.cfg.topic.opening_question,
            self_name=viewer_name,
            history=self.history(items, viewer_id, framing),
        )

    def task_instruction(self, framing: str, round_id: int, recipient_name: str | None, first: bool) -> str:
        limit = self.cfg.generation.message_word_limit
        if framing == "forum":
            what = "your first post" if first else "your next post"
            return f"Write {what} to the public forum (at most {limit} words). Reply with the post text only."
        if framing == "dyadic":
            return (
                f"Write a private message to {recipient_name} (at most {limit} words). "
                f"Only {recipient_name} will read it. Reply with the message text only."
            )
        what = "your first contribution" if first else "your next contribution"
        return f"Write {what} to the discussion (at most {limit} words). Reply with the text only."

    def generation_prompt(
        self,
        items: list[ContextItem],
        viewer_id: str,
        framing: str,
        round_id: int,
        recipient_name: str | None,
        first: bool,
        viewer_name: str = "",
    ) -> str:
        return (
            self.context_block(items, viewer_id, framing, viewer_name)
            + "\n\n"
            + self.task_instruction(framing, round_id, recipient_name, first)
        )

    def estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.cfg.memory.chars_per_token) + 1
