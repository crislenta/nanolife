"""Offline smoke test for spatial movement (no LLM, deterministic).

Swaps in a stub CognitiveFunction that emits a fixed (mode, delta) each tick,
then asserts agent positions update only when mode=="walk", and that a 'step'
event appears in the log.

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
    """Returns pre-programmed (mode, delta) decisions by agent name."""

    def __init__(self, script: dict[str, list[tuple[str, tuple[int, int]]]]) -> None:
        self.script = {n: list(d) for n, d in script.items()}
        self.total_cost = 0.0

    async def decide(self, agent: Agent, recent: list[Event], ctx: str,
                     alive: list[Agent]) -> dict:
        q = self.script.get(agent.name, [])
        mode, delta = q.pop(0) if q else ("productive", (0, 0))
        return {
            "thought": f"scripted mode={mode} delta={list(delta)}",
            "mode": mode,
            "action": mode,
            "reputation_deltas": {},
            "new_friend": None,
            "new_location": None,
            "delta": list(delta),
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
        # Alice (row 0): 3 walks east. Should reach (3, 0).
        "Alice": [("walk", (1, 0)), ("walk", (1, 0)), ("walk", (1, 0))],
        # Bob (row 1): 1 walk east blocked by wall at (2,1); stays at (1,1).
        "Bob":   [("walk", (1, 0)), ("walk", (1, 0)), ("walk", (1, 0))],
        # Carol (row 2): productive action with delta=(1,0) — MUST NOT step.
        "Carol": [("productive", (1, 0)), ("productive", (1, 0)), ("productive", (1, 0))],
    })

    engine = Engine(world=world, cognitive=cognitive)

    alice, _ = engine.spawn_agent(name="Alice", location="field")
    alice.position = (0, 0)
    bob, _ = engine.spawn_agent(name="Bob", location="field")
    bob.position = (1, 1)
    carol, _ = engine.spawn_agent(name="Carol", location="field")
    carol.position = (0, 2)

    async def go():
        for _ in range(3):
            await engine.tick()

    asyncio.run(go())

    steps = [e for e in world.event_log.all() if e.get("type") == "step"]
    step_agents = {e["agent"] for e in steps}

    print(f"steps emitted: {len(steps)}")
    for e in steps:
        print("  ", e["content"])
    print(f"Alice pos: {alice.position}  (expected (3, 0))")
    print(f"Bob pos:   {bob.position}  (expected (1, 1) — wall blocks east)")
    print(f"Carol pos: {carol.position}  (expected (0, 2) — productive, no step)")

    # Original assertions
    assert alice.position == (3, 0), f"Alice moved wrong: {alice.position}"
    assert bob.position == (1, 1), f"Bob should be blocked by wall: {bob.position}"
    assert len(steps) == 3, f"expected 3 step events (3 Alice, 0 Bob, 0 Carol), got {len(steps)}"

    # New assertions (fixup PR #2)
    assert alice.id in step_agents, "walk+delta(1,0) agent should have emitted a step event"
    assert carol.id not in step_agents, \
        "productive+delta(1,0) agent must NOT step — delta is ignored off the walk verb"
    assert carol.position == (0, 2), \
        f"Carol (productive+delta) should not move: {carol.position}"
    print("OK")


if __name__ == "__main__":
    main()
