"""Smoke tests for nanosim.render — the map-centered renderer.

Locks in the layout contract from the refactor that made render.py the
sole renderer (terminal.py deleted): map dominates the center, roster
on the left, status on the right, event log along the bottom.
"""
from __future__ import annotations

from nanosim.common import Agent
from nanosim.render import render
from nanosim.worldmap import Tile, WorldMap


def _make_agent(name: str, position: tuple[int, int]) -> Agent:
    return Agent(
        id=f"agent_{name.lower()}",
        name=name,
        alive=True,
        traits={},
        memory=[],
        friendships=[],
        parents=[],
        reputation=0.0,
        goal="survive",
        identity_md="",
        birth_tick=0,
        death_tick=None,
        resources=12.0,
        position=position,
    )


def _make_world() -> WorldMap:
    legend = {
        ".": Tile(glyph=".", terrain="grass", color="green"),
        "#": Tile(glyph="#", terrain="wall", color="grey50"),
        "~": Tile(glyph="~", terrain="water", color="blue"),
    }
    ascii_map = "\n".join([
        "#########",
        "#.......#",
        "#..~~...#",
        "#..~~...#",
        "#.......#",
        "#########",
    ])
    return WorldMap.from_ascii(ascii_map, legend)


def test_render_returns_non_empty_frame():
    world = _make_world()
    agents = [_make_agent("Ada", (2, 1)), _make_agent("Bjorn", (5, 4))]
    frame = render(world, agents, ["t   1 [action] Ada: explores east"], tick=1)
    assert isinstance(frame, str)
    assert frame.strip() != ""


def test_render_includes_map_tiles_and_agent_glyphs():
    world = _make_world()
    agents = [_make_agent("Ada", (2, 1)), _make_agent("Bjorn", (5, 4))]
    frame = render(world, agents, ["t   1 [action] Ada: explores east"], tick=1)
    # Tile glyphs from the legend must appear (map is being drawn).
    assert "#" in frame
    assert "." in frame
    assert "~" in frame
    # At least one agent glyph (first letter of name, uppercased) must appear.
    assert "A" in frame or "B" in frame


def test_render_has_event_log_panel_title():
    world = _make_world()
    agents = [_make_agent("Ada", (2, 1))]
    frame = render(world, agents, ["t   1 [action] Ada: explores east"], tick=1)
    # Panel titles use lowercase per render.py: 'agents', 'world', 'status', 'event log'.
    lower = frame.lower()
    assert "event log" in lower
    assert "world" in lower
    assert "agents" in lower
    assert "status" in lower


def test_render_map_region_dominates_center():
    """The map panel should get the largest horizontal slice when a WorldMap is present.

    Heuristic: count occurrences of map-only tile glyphs ('#', '~') vs the
    side-panel labels. The map panel renders the full map every frame, so
    the number of '#' chars should comfortably exceed the number of agent
    roster rows (which is bounded by len(agents)).
    """
    world = _make_world()
    agents = [_make_agent("Ada", (2, 1)), _make_agent("Bjorn", (5, 4))]
    frame = render(world, agents, ["t   1 [action] Ada: explores east"], tick=1)
    # The 6x9 map has 30 '#' tiles on the border alone.
    assert frame.count("#") >= 20, (
        f"expected map tiles to dominate, got only {frame.count('#')} '#' chars"
    )


def test_render_works_without_worldmap():
    """Legacy scenarios with no WorldMap should still render (roster + status only)."""
    agents = [_make_agent("Ada", (0, 0))]
    frame = render(None, agents, ["t   1 [action] Ada: thinks"], tick=1)
    assert isinstance(frame, str)
    assert frame.strip() != ""
    assert "agents" in frame.lower()
