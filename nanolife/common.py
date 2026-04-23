"""Shared data types used across the simulation.

Defines the Agent, Event, and TickResult structures that every other
module depends on.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, TypedDict


class Event(TypedDict, total=False):
    tick: int
    type: str
    agent: str
    content: str
    parents: list[str]
    inherited_identity: str
    inherited_traits: dict[str, Any]
    inherited_goal: str
    mode: str
    delta: float
    source: str  # rumor origin
    reason: str
    cause: str
    age_days: int
    goal: str
    witnesses: list[str]
    resources_changed: float


@dataclass
class Agent:
    id: str
    name: str
    alive: bool
    traits: dict[str, float]
    memory: list[Event]
    friendships: list[str]
    parents: list[str]
    reputation: float
    goal: str
    identity_md: str
    birth_tick: int
    death_tick: int | None
    resources: float = 10.0
    location: str | None = None
    # Optional (x, y) grid position used only by the renderer. The engine
    # does not read this field; it exists so scenarios with a map can place
    # agents on that map. None = not placed on a grid.
    position: tuple[int, int] | None = None

    lifespan: int = 365  # ticks before natural death

    def age(self, current_tick: int) -> int:
        return current_tick - self.birth_tick


@dataclass
class TickResult:
    tick: int
    events: list[Event] = field(default_factory=list)
    births: int = 0
    deaths: int = 0
    population: int = 0


def make_id() -> str:
    return uuid.uuid4().hex[:8]
