"""Abstract interfaces for pluggable simulation components.

Defines the contracts for CognitiveFunction, CompressionFunction,
SpreadFunction, and the Scenario dataclass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .common import Agent, Event


class CognitiveFunction(ABC):
    """How an agent decides what to do each tick."""

    @abstractmethod
    async def decide(
        self,
        agent: Agent,
        visible_events: list[Event],
        world_context: str,
        agents: list[Agent],
    ) -> dict[str, Any]:
        """Return dict with keys: thought, action, mode, new_friend, reputation_deltas."""
        ...

    @abstractmethod
    async def reflect(
        self,
        agent: Agent,
        todays_events: list[Event],
    ) -> str:
        """End-of-day self-improvement. Returns one sentence to append to identity_md."""
        ...

    @abstractmethod
    async def name_child(self, parent_a: Agent, parent_b: Agent) -> str:
        """Generate a name for the child of two parents."""
        ...


class CompressionFunction(ABC):
    """How the shared event log shrinks when it exceeds the token budget."""

    @abstractmethod
    async def compress(self, events: list[Event], token_budget: int) -> tuple[str, list[Event]]:
        """Summarize oldest events. Returns (summary_text, remaining_events)."""
        ...


class SpreadFunction(ABC):
    """How information propagates between agents."""

    @abstractmethod
    def spread(
        self,
        agents: list[Agent],
        events: list[Event],
        degradation: float,
    ) -> list[Event]:
        """Generate rumor events from agent-to-agent information transfer."""
        ...


@dataclass
class Scenario:
    """JSON world definition — the primary user surface."""

    name: str
    description: str
    theme: str
    harshness: float
    tick_unit: str = "4h"
    locations: list[str] = field(default_factory=list)
    agents: list[dict[str, Any]] = field(default_factory=list)
    opening_events: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    location_coords: dict[str, dict[str, Any]] = field(default_factory=dict)
    base_drain: float = 1.0
    base_gain: float = 4.0
    reputation_decay: float = 0.03
    starting_resources: float = 15.0
