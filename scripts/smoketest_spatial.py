"""Offline smoke test for spatial movement (no LLM, deterministic).

Swaps in a stub CognitiveFunction that emits a fixed delta each tick, then
asserts agent positions update and a 'step' event appears in the log.

Run: python -m scripts.smoketest_spatial
"""
from __future__ import annotations

import asyncio

from nanolife.common import Agent, Event
from nanolife.engine import Engine
from nanolife.interfaces import CognitiveFunction
from nanolife.world import WorldState
from nanolife.worldmap import Tile, WorldMap


class ScriptedCognitive(CognitiveFunction):
    """Returns pre-programmed decisions keyed by agent name -> list of deltas."""

    def __init__(self, deltas_by_agent: dict[str, list[tuple[int, int]]]) -> None:
        self.deltas = {n: list(d) for n, d in deltas_by_agent.items()}
        self.total_cost = 0.0

    async def decide(self, agent: Agent, recent: list[Event], ctx: str,
                     alive: list[Agent]) -> dict:
        queue = self.deltas.get(agent.name, [])
        delta = list(queue.pop(0)) if queue else [0, 0]
        return {
            "thought": f"scripted delta={delta}",
            "mode": "productive",
            "action": "walk",
            "reputation_deltas": {},
            "new_friend": None,
            "new_location": None,
            "delta": delta,
        }

    async def reflect(self, agent: Agent, recent: list[Event]) -> str:
        return ""

    async def name_child(self, parent: Agent, other: Agent) -> str:
        return "Child"


def main() -> None:
    # 5x3 map, wall blocks (2,1).
    legend = {
        ".": Tile(".", "grass"),
        "#": Tile("#", "wall"),
    }
    wmap = WorldMap.from_ascii(".....\n..#..\n.....", legend)

    world = WorldState.create(harshness=0.0)
    world.world_map = wmap

    cognitive = ScriptedCognitive({
        "Alice": [(1, 0), (1, 0), (1, 0), (1, 0)],  # 4 steps right; tick 3 hits wall at (2,1)? no Alice is row 0
        "Bob": [(1, 0), (1, 0), (1, 0)],            # row 1: tick 1 (1,1)->(2,1) blocked, should not move
    })

    engine = Engine(world=world, cognitive=cognitive)

    alice, _ = engine.spawn_agent(name="Alice", location="field")
    alice.position = (0, 0)
    bob, _ = engine.spawn_agent(name="Bob", location="field")
    bob.position = (1, 1)

    async def go():
        for _ in range(3):
            await engine.tick()

    asyncio.run(go())

    steps = [e for e in world.event_log.all() if e.get("type") == "step"]
    print(f"steps emitted: {len(steps)}")
    for e in steps:
        print("  ", e["content"])
    print(f"Alice pos: {alice.position}  (expected (3, 0))")
    print(f"Bob pos: {bob.position}  (expected (1, 1) — wall blocks east)")

    assert alice.position == (3, 0), f"Alice moved wrong: {alice.position}"
    assert bob.position == (1, 1), f"Bob should be blocked by wall: {bob.position}"
    assert len(steps) == 3, f"expected 3 step events (3 for Alice, 0 for Bob), got {len(steps)}"
    print("OK")


if __name__ == "__main__":
    main()
