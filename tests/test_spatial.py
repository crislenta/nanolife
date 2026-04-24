"""Contract tests for the spatial behavior introduced in PR #2.

Background
----------
PR #2 (feat/spatial-movement-local-view, merged to master 2026-04-23) added
grid-based movement and local ASCII perception to the engine. A handful of
invariants keep that surface trustworthy:

WorldMap data model
~~~~~~~~~~~~~~~~~~~
  1. ``WorldMap.from_ascii`` right-pads every row to the max width so the
     grid is always rectangular. An empty input is a 0x0 map, not a crash.
  2. ``passable(x, y)`` returns False for out-of-bounds AND for any tile
     whose terrain is in ``BLOCKED_TERRAINS`` (mountain, water, wall).
     Grass, path, road, floor, etc. are walkable.
  3. ``local_view`` renders the caller as ``@``, other alive agents by
     their name's first letter, map tiles by glyph, and off-map cells as
     a single space — preserving rectangular shape so the LLM sees a patch.
  4. ``local_view`` ignores dead agents and agents whose position is None.

Engine movement (``Engine._try_step``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  5. No world_map -> no movement. An agent with position=None never steps.
  6. A (0, 0) delta is a no-op — no event, no mutation.
  7. Impassable tiles (wall/water/mountain) block; the agent stays put,
     no ``step`` event is emitted.
  8. Occupied tiles block; two agents never collide at the same (x, y).
  9. On a successful step, a ``step`` event appears in the world log with
     type="step", the moving agent's id, and witnesses set to the agent
     list passed in.

Cognitive delta parsing (``LLMCognitive._parse_response``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 10. ``action == "walk"`` promotes mode to ``"walk"`` and carries the
     delta through. Non-walk modes never reach ``_try_step`` — the engine
     gates movement on ``mode == "walk"`` — so any delta the parser may
     pass through for a non-walk decision is harmless.
 11. Delta components are clamped to {-1, 0, 1}.
 12. ``walk`` with a missing/zero/malformed delta falls back to
     ``mode == "productive"`` and records ``parse_error``. The delta key
     is dropped so the engine cannot try to step.

These are pure unit tests — no LLM, no sockets. We bypass the LLMCognitive
API-key check via ``__new__`` exactly like ``test_cognitive_freewill.py``.
"""
from __future__ import annotations

import asyncio

import pytest

from nanolife.common import Agent
from nanolife.defaults.cognitive import LLMCognitive
from nanolife.engine import Engine
from nanolife.interfaces import CognitiveFunction
from nanolife.world import WorldState
from nanolife.worldmap import BLOCKED_TERRAINS, Tile, WorldMap


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _make_agent(name: str, pos: tuple[int, int] | None, alive: bool = True) -> Agent:
    return Agent(
        id=f"{name.lower()}_id",
        name=name,
        alive=alive,
        traits={},
        memory=[],
        friendships=[],
        parents=[],
        reputation=0.0,
        goal="",
        identity_md="",
        birth_tick=0,
        death_tick=None,
        resources=10.0,
        location=None,
        position=pos,
    )


LEGEND = {
    ".": Tile(".", "grass"),
    "#": Tile("#", "wall"),
    "~": Tile("~", "water"),
    "^": Tile("^", "mountain"),
}


def _map(text: str) -> WorldMap:
    return WorldMap.from_ascii(text, LEGEND)


class _NullCognitive(CognitiveFunction):
    """Never called — Engine only needs an attribute presence for _try_step."""

    async def decide(self, agent, recent, ctx, alive):  # pragma: no cover - defensive
        raise AssertionError("not used")

    async def reflect(self, agent, recent):  # pragma: no cover
        return ""

    async def name_child(self, parent, other):  # pragma: no cover
        return "Child"


@pytest.fixture
def parser() -> LLMCognitive:
    """LLMCognitive with the API client skipped — parser methods only."""
    return LLMCognitive.__new__(LLMCognitive)


@pytest.fixture
def engine() -> Engine:
    world = WorldState.create(harshness=0.0)
    return Engine(world=world, cognitive=_NullCognitive())


@pytest.fixture
def wmap() -> WorldMap:
    # 5x3 grid: wall at (2, 1) blocks the middle row.
    #   . . . . .
    #   . . # . .
    #   . . . . .
    return _map(".....\n..#..\n.....")


# ---------------------------------------------------------------------------
# 1-4: WorldMap data model
# ---------------------------------------------------------------------------


class TestWorldMap:
    def test_from_ascii_pads_to_rectangle(self):
        # Ragged input -> rectangular grid at max width.
        m = _map("..\n.\n...")
        assert m.width == 3
        assert m.height == 3
        # Short rows are padded with the fallback glyph (' ').
        assert m.tiles[1][2].glyph == " "

    def test_from_ascii_empty_is_zero_by_zero(self):
        m = _map("")
        assert m.width == 0
        assert m.height == 0
        assert m.tiles == []

    def test_in_bounds_rejects_negative_and_overflow(self, wmap):
        assert wmap.in_bounds(0, 0)
        assert wmap.in_bounds(4, 2)
        assert not wmap.in_bounds(-1, 0)
        assert not wmap.in_bounds(0, -1)
        assert not wmap.in_bounds(5, 0)
        assert not wmap.in_bounds(0, 3)

    def test_blocked_terrains_contract(self):
        # Public constant: anything the engine treats as impassable.
        assert "wall" in BLOCKED_TERRAINS
        assert "water" in BLOCKED_TERRAINS
        assert "mountain" in BLOCKED_TERRAINS
        # Grass MUST stay walkable — it's every scenario's default floor.
        assert "grass" not in BLOCKED_TERRAINS

    def test_passable_rejects_oob_and_blocked_tiles(self, wmap):
        assert wmap.passable(0, 0)        # grass
        assert wmap.passable(4, 2)        # grass
        assert not wmap.passable(2, 1)    # wall
        assert not wmap.passable(-1, 0)   # out of bounds
        assert not wmap.passable(5, 0)    # out of bounds

    def test_passable_rejects_water_and_mountain(self):
        m = _map(".~.\n.^.")
        assert not m.passable(1, 0)  # water
        assert not m.passable(1, 1)  # mountain
        assert m.passable(0, 0)
        assert m.passable(2, 1)

    def test_local_view_renders_self_and_others(self, wmap):
        me = _make_agent("Me", (2, 1))      # on the wall tile (renderer still centers here)
        a = _make_agent("Alice", (3, 1))
        b = _make_agent("Bob", (2, 0))
        view = wmap.local_view((2, 1), radius=1, agents=[me, a, b])
        rows = view.splitlines()
        assert len(rows) == 3
        # Each row width = 2*radius + 1 = 3.
        assert all(len(r) == 3 for r in rows)
        # Center is '@' (self).
        assert rows[1][1] == "@"
        # East neighbor is 'A' (Alice's first letter).
        assert rows[1][2] == "A"
        # North neighbor is 'B' (Bob).
        assert rows[0][1] == "B"

    def test_local_view_fills_off_map_with_space(self, wmap):
        me = _make_agent("Me", (0, 0))
        view = wmap.local_view((0, 0), radius=1, agents=[me])
        rows = view.splitlines()
        # Top-left 2x2 of the view is off the map: all spaces.
        assert rows[0] == "   "
        assert rows[1][0] == " "
        # Self at center.
        assert rows[1][1] == "@"

    def test_local_view_ignores_dead_and_unplaced_agents(self, wmap):
        me = _make_agent("Me", (2, 1))
        ghost = _make_agent("Ghost", (3, 1), alive=False)
        floater = _make_agent("Floater", None)
        view = wmap.local_view((2, 1), radius=1, agents=[me, ghost, floater])
        rows = view.splitlines()
        # East cell should show the map glyph, NOT 'G' for Ghost.
        assert rows[1][2] == "."


# ---------------------------------------------------------------------------
# 5-9: Engine._try_step movement contract
# ---------------------------------------------------------------------------


class TestEngineTryStep:
    def test_no_world_map_means_no_step(self, engine):
        agent = _make_agent("Alice", (0, 0))
        ev = engine._try_step(agent, [1, 0], wmap=None, alive=[agent], tick=0, witnesses=[])
        assert ev is None
        assert agent.position == (0, 0)

    def test_agent_without_position_never_moves(self, engine, wmap):
        agent = _make_agent("Alice", None)
        ev = engine._try_step(agent, [1, 0], wmap=wmap, alive=[agent], tick=0, witnesses=[])
        assert ev is None
        assert agent.position is None

    def test_zero_delta_is_noop(self, engine, wmap):
        agent = _make_agent("Alice", (1, 1))
        ev = engine._try_step(agent, [0, 0], wmap=wmap, alive=[agent], tick=0, witnesses=[])
        assert ev is None
        assert agent.position == (1, 1)

    def test_malformed_delta_is_noop(self, engine, wmap):
        agent = _make_agent("Alice", (1, 1))
        for bad in (None, "east", [1], [1, 0, 0], (1,), 7):
            ev = engine._try_step(agent, bad, wmap=wmap, alive=[agent], tick=0, witnesses=[])
            assert ev is None, f"delta={bad!r} should be ignored"
        assert agent.position == (1, 1)

    def test_wall_blocks_east_step(self, engine, wmap):
        # Bob at (1, 1) tries east -> (2, 1) which is a wall.
        bob = _make_agent("Bob", (1, 1))
        ev = engine._try_step(bob, [1, 0], wmap=wmap, alive=[bob], tick=0, witnesses=[])
        assert ev is None
        assert bob.position == (1, 1)

    def test_occupancy_blocks_step(self, engine, wmap):
        alice = _make_agent("Alice", (0, 0))
        blocker = _make_agent("Blocker", (1, 0))
        ev = engine._try_step(alice, [1, 0], wmap=wmap, alive=[alice, blocker],
                              tick=0, witnesses=[])
        assert ev is None
        assert alice.position == (0, 0)
        assert blocker.position == (1, 0)

    def test_dead_agents_do_not_block_step(self, engine, wmap):
        alice = _make_agent("Alice", (0, 0))
        corpse = _make_agent("Corpse", (1, 0), alive=False)
        ev = engine._try_step(alice, [1, 0], wmap=wmap, alive=[alice, corpse],
                              tick=0, witnesses=[])
        assert ev is not None
        assert alice.position == (1, 0)

    def test_successful_step_emits_event(self, engine, wmap):
        alice = _make_agent("Alice", (0, 0))
        ev = engine._try_step(alice, [1, 0], wmap=wmap, alive=[alice],
                              tick=7, witnesses=["alice_id"])
        assert ev is not None
        assert ev["type"] == "step"
        assert ev["agent"] == alice.id
        assert ev["tick"] == 7
        assert ev["witnesses"] == ["alice_id"]
        assert "stepped (0, 0) -> (1, 0)" in ev["content"]
        assert alice.position == (1, 0)

    def test_oob_step_blocked_at_map_edge(self, engine, wmap):
        # Alice at (4, 0) tries east -> (5, 0): off the map.
        alice = _make_agent("Alice", (4, 0))
        ev = engine._try_step(alice, [1, 0], wmap=wmap, alive=[alice], tick=0, witnesses=[])
        assert ev is None
        assert alice.position == (4, 0)


# ---------------------------------------------------------------------------
# 10-12: Cognitive delta-parse contract
# ---------------------------------------------------------------------------


class TestCognitiveDeltaParse:
    def test_walk_with_valid_delta_carries_through(self, parser):
        raw = """{
            "thought": "heading east",
            "action": "walk",
            "description": "walking east to the river",
            "delta": [1, 0]
        }"""
        out = parser._parse_response(raw, agents=[], agent_name="Alice")
        assert out["mode"] == "walk"
        assert out["delta"] == [1, 0]
        assert "parse_error" not in out

    def test_delta_components_clamped_to_unit_range(self, parser):
        raw = """{
            "thought": "dashing",
            "action": "walk",
            "description": "sprint",
            "delta": [5, -3]
        }"""
        out = parser._parse_response(raw, agents=[], agent_name="Alice")
        # Both components must be clamped into {-1, 0, 1}.
        assert out["delta"] == [1, -1]

    def test_non_walk_mode_preserves_decision_but_engine_ignores_delta(self, parser):
        # The parser may pass delta through in any mode — it's the engine's
        # job to gate movement on mode=="walk". The engine-integration test
        # ``test_productive_with_delta_does_not_move`` exercises that gate.
        # This test only pins the parser's responsibility: when the model
        # chose "productive" (or any non-walk verb), the returned mode must
        # stay non-walk, so the engine's gate keeps the agent stationary.
        raw = """{
            "thought": "farming",
            "mode": "productive",
            "action": "tend the barley",
            "delta": [1, 0]
        }"""
        out = parser._parse_response(raw, agents=[], agent_name="Alice")
        assert out["mode"] == "productive"
        # Parser may or may not strip delta — engine is the gate. Pin behavior:
        # whatever the parser chooses, mode stays "productive" so _try_step
        # never runs on this decision.
        assert out["mode"] != "walk"

    def test_walk_with_zero_delta_falls_back_to_productive(self, parser):
        raw = """{
            "thought": "stuck",
            "action": "walk",
            "description": "walk nowhere",
            "delta": [0, 0]
        }"""
        out = parser._parse_response(raw, agents=[], agent_name="Alice")
        assert out["mode"] == "productive"
        assert "delta" not in out
        assert "parse_error" in out
        assert "walk chosen but delta invalid" in out["parse_error"]

    def test_walk_with_missing_delta_falls_back_to_productive(self, parser):
        raw = """{
            "thought": "no vector",
            "action": "walk",
            "description": "walking"
        }"""
        out = parser._parse_response(raw, agents=[], agent_name="Alice")
        assert out["mode"] == "productive"
        assert "delta" not in out
        assert "parse_error" in out

    def test_walk_with_malformed_delta_falls_back(self, parser):
        raw = """{
            "thought": "??",
            "action": "walk",
            "description": "walking",
            "delta": "east"
        }"""
        out = parser._parse_response(raw, agents=[], agent_name="Alice")
        assert out["mode"] == "productive"
        assert "delta" not in out
        assert "parse_error" in out


# ---------------------------------------------------------------------------
# Integration: scripted tick via Engine.tick (exercises _try_step path end-to-end)
# ---------------------------------------------------------------------------


class _ScriptedCognitive(CognitiveFunction):
    """Returns a pre-programmed decision for a named agent on every tick."""

    def __init__(self, decision_by_name: dict[str, dict]) -> None:
        self._by_name = decision_by_name

    async def decide(self, agent, recent, ctx, alive):
        return self._by_name[agent.name]

    async def reflect(self, agent, recent):
        return ""

    async def name_child(self, parent, other):
        return "Child"


class TestEngineTickIntegration:
    def test_walk_moves_agent_and_logs_step(self, wmap):
        world = WorldState.create(harshness=0.0)
        world.world_map = wmap
        cog = _ScriptedCognitive({
            "Alice": {
                "thought": "east",
                "mode": "walk",
                "action": "walk east",
                "reputation_deltas": {},
                "new_friend": None,
                "new_location": None,
                "delta": [1, 0],
            },
        })
        engine = Engine(world=world, cognitive=cog)
        alice, _ = engine.spawn_agent(name="Alice", location="field")
        alice.position = (0, 0)

        asyncio.run(engine.tick())

        assert alice.position == (1, 0)
        steps = [e for e in world.event_log.all() if e.get("type") == "step"]
        assert len(steps) == 1
        assert steps[0]["agent"] == alice.id

    def test_productive_with_delta_does_not_move(self, wmap):
        world = WorldState.create(harshness=0.0)
        world.world_map = wmap
        cog = _ScriptedCognitive({
            "Alice": {
                "thought": "work",
                "mode": "productive",
                "action": "tend the field",
                "reputation_deltas": {},
                "new_friend": None,
                "new_location": None,
                "delta": [1, 0],  # should be IGNORED because mode != walk
            },
        })
        engine = Engine(world=world, cognitive=cog)
        alice, _ = engine.spawn_agent(name="Alice", location="field")
        alice.position = (0, 0)

        asyncio.run(engine.tick())

        assert alice.position == (0, 0)
        steps = [e for e in world.event_log.all() if e.get("type") == "step"]
        assert steps == []
