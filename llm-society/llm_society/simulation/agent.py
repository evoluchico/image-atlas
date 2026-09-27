"""Agents: a fixed profile, a system prompt, and an explicit memory.

Agents know nothing about environments. What they can see is decided by the
environment (which fills their memory) and the memory policy (which selects
the context); the framing text comes from the `PromptBuilder`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .memory import AgentMemory
from .population import AgentProfile


@dataclass
class Agent:
    profile: AgentProfile
    system_prompt: str
    memory: AgentMemory

    @property
    def agent_id(self) -> str:
        return self.profile.agent_id

    @property
    def model_key(self) -> str:
        return self.profile.model_key

    @classmethod
    def create(cls, profile: AgentProfile, system_prompt: str) -> "Agent":
        return cls(profile, system_prompt, AgentMemory(profile.agent_id, initial_profile=asdict(profile)))
