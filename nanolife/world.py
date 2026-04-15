"""World state, clock, and event log.

Holds the shared mutable state for one simulation run: agents, locations,
the append-only event log, and the calendar clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import Agent, Event
from .logger import RunLogger


TICKS_PER_DAY: dict[str, int] = {
    "minute": 1440,
    "hour": 24,
    "4h": 6,
    "day": 1,
    "week": 1,  # 1 tick = 1 week
}

MEMORY_DEPTH: dict[str, int] = {
    "minute": 30,
    "hour": 48,
    "4h": 28,
    "day": 14,
    "week": 8,
}


class Clock:
    def __init__(self, tick_unit: str = "4h"):
        self.tick: int = 0
        self.tick_unit = tick_unit
        self._tpd = TICKS_PER_DAY.get(tick_unit, 6)

    def advance(self) -> int:
        self.tick += 1
        return self.tick

    def calendar(self) -> dict[str, int]:
        if self.tick_unit == "week":
            total_days = self.tick * 7
        else:
            total_days = self.tick // self._tpd

        year = total_days // 365
        day_of_year = total_days % 365
        month = day_of_year // 30
        day = day_of_year % 30
        return {"day": day + 1, "month": month + 1, "year": year }

    def label(self) -> str:
        c = self.calendar()
        return f"day {c['day']}, month {c['month']}, year {c['year']}"


class EventLog:
    """Append-only shared event log with per-agent local views."""

    def __init__(self, logger: RunLogger | None = None):
        self._events: list[Event] = []
        self._logger = logger

    def append(self, event: Event) -> None:
        self._events.append(event)
        if self._logger:
            self._logger.log(event)

    def all(self) -> list[Event]:
        return list(self._events)

    def since(self, tick: int) -> list[Event]:
        return [e for e in self._events if e.get("tick", 0) >= tick]

    def for_agent(self, agent_id: str) -> list[Event]:
        """Return events this agent witnessed (was in witnesses list, or is the agent)."""
        out: list[Event] = []
        for e in self._events:
            witnesses = e.get("witnesses", [])
            if agent_id in witnesses or e.get("agent") == agent_id:
                out.append(e)
        return out

    def replace(self, new_events: list[Event]) -> None:
        """Used by compression — replace log contents."""
        self._events = new_events

    @property
    def size(self) -> int:
        return len(self._events)

    def token_estimate(self) -> int:
        import json
        return sum(len(json.dumps(e)) // 4 for e in self._events)




@dataclass
class WorldState:
    clock: Clock
    event_log: EventLog
    harshness: float
    agents: list[Agent] = field(default_factory=list)
    locations: list[str] = field(default_factory=lambda: ["Central Hub", "The Wilds", "Outskirts"])
    logger: RunLogger | None = None
    scenario_name: str = "default"
    tick_unit: str = "4h"
    token_threshold: int = 4000
    total_births: int = 0
    total_deaths: int = 0
    location_coords: dict[str, dict[str, Any]] = field(default_factory=dict)
    base_drain: float = 1.0
    base_gain: float = 4.0
    reputation_decay: float = 0.03
    starting_resources: float = 15.0

    @classmethod
    def create(
        cls,
        harshness: float = 0.5,
        tick_unit: str = "4h",
        run_dir: str | None = None,
        scenario_name: str = "default",
        token_threshold: int = 4000,
        base_drain: float = 1.0,
        base_gain: float = 4.0,
        reputation_decay: float = 0.03,
        starting_resources: float = 15.0,
    ) -> WorldState:
        logger = RunLogger(run_dir) if run_dir else None
        clock = Clock(tick_unit)
        event_log = EventLog(logger)
        return cls(
            clock=clock,
            event_log=event_log,
            harshness=harshness,
            logger=logger,
            scenario_name=scenario_name,
            tick_unit=tick_unit,
            token_threshold=token_threshold,
            base_drain=base_drain,
            base_gain=base_gain,
            reputation_decay=reputation_decay,
            starting_resources=starting_resources,
        )

    @property
    def alive_agents(self) -> list[Agent]:
        return [a for a in self.agents if a.alive]

    @property
    def dead_agents(self) -> list[Agent]:
        return [a for a in self.agents if not a.alive]

    @property
    def population(self) -> int:
        return len(self.alive_agents)

    @property
    def memory_depth(self) -> int:
        return MEMORY_DEPTH.get(self.tick_unit, 28)
