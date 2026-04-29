"""Random rumor spread (default implementation).

Each tick a random subset of agents share one memory with a neighbor,
with optional content degradation to model information loss.
"""
from __future__ import annotations

import random

from ..common import Agent, Event
from ..interfaces import SpreadFunction


class RandomSpread(SpreadFunction):
    """Each tick, a random subset of agents share one memory with a neighbor.
    Content degrades by the degradation float (information loss)."""

    def spread(
        self,
        agents: list[Agent],
        events: list[Event],
        degradation: float = 0.2,
    ) -> list[Event]:
        if len(agents) < 2:
            return []

        rumors: list[Event] = []
        spreaders = random.sample(agents, min(len(agents) // 2, len(agents)))

        for agent in spreaders:
            if not agent.memory:
                continue

            shareable = [
                e for e in agent.memory
                if e.get("type") in ("action", "friendship", "death", "birth", "reputation")
            ]
            if not shareable:
                continue

            item = random.choice(shareable)
            others = [a for a in agents if a.id != agent.id]
            if not others:
                continue
            target = random.choice(others)

            content = item.get("content", "")
            if random.random() < degradation:
                words = content.split()
                if len(words) > 3:
                    drop = random.randint(0, len(words) - 1)
                    words[drop] = "..."
                    content = " ".join(words)
                content += " (rumor)"

            rumor_event: Event = {
                "tick": 0,  # set by engine
                "type": "rumor",
                "agent": target.id,
                "content": f"{agent.name} told {target.name}: {content}",
                "source": agent.id,
                "witnesses": [target.id],
            }
            rumors.append(rumor_event)

        return rumors
