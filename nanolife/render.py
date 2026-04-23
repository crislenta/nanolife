"""Caves-of-Qud-style terminal renderer.

Pure function: takes read-only state, returns a rendered string frame.
The engine never calls this; it's driven by the simulate script when
``--render`` is passed. Works with or without a WorldMap.
"""
from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .common import Agent
from .worldmap import WorldMap


def _map_panel(world: WorldMap, agents: list[Agent]) -> Panel:
    # Agents overlay tiles; later agents in the list win on collision.
    overlay: dict[tuple[int, int], tuple[str, str]] = {}
    for a in agents:
        if a.position is None or not a.alive:
            continue
        x, y = a.position
        if 0 <= x < world.width and 0 <= y < world.height:
            glyph = a.name[0].upper() if a.name else "?"
            overlay[(x, y)] = (glyph, "bold yellow")

    text = Text()
    for y in range(world.height):
        for x in range(world.width):
            if (x, y) in overlay:
                g, style = overlay[(x, y)]
                text.append(g, style=style)
            else:
                tile = world.tiles[y][x]
                text.append(tile.glyph, style=tile.color or "white")
        if y < world.height - 1:
            text.append("\n")
    return Panel(text, title="map", border_style="dim")


def _roster_panel(agents: list[Agent]) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold", no_wrap=True)
    table.add_column(style="dim")
    for a in agents:
        if not a.alive:
            continue
        rep = getattr(a, "reputation", None)
        res = getattr(a, "resources", None)
        bits = []
        if rep is not None:
            bits.append(f"rep {rep:+.1f}")
        if res is not None:
            bits.append(f"res {res:.0f}")
        table.add_row(a.name, " ".join(bits))
    return Panel(table, title="agents", border_style="dim")


def _log_panel(event_log: list[str]) -> Panel:
    tail = event_log[-15:]
    text = Text("\n".join(tail) if tail else "(no events)", style="white")
    return Panel(text, title="event log", border_style="dim")


def render(
    world: Optional[WorldMap],
    agents: list[Agent],
    event_log: list[str],
    tick: int,
    width: int = 120,
    height: int = 32,
) -> str:
    """Render one frame; return a styled string (ANSI via rich export)."""
    console = Console(record=True, width=width, height=height, force_terminal=True)
    layout = Layout()
    layout.split_column(
        Layout(Panel(Text(f"nanolife — tick {tick}", style="bold cyan"), border_style="cyan"),
               name="header", size=3),
        Layout(name="body"),
    )
    if world is not None:
        layout["body"].split_row(
            Layout(_map_panel(world, agents), name="map", ratio=2),
            Layout(name="side", ratio=1),
        )
        layout["body"]["side"].split_column(
            Layout(_roster_panel(agents), name="roster"),
            Layout(_log_panel(event_log), name="log"),
        )
    else:
        layout["body"].split_row(
            Layout(_roster_panel(agents), name="roster", ratio=1),
            Layout(_log_panel(event_log), name="log", ratio=2),
        )
    console.print(layout)
    return console.export_text(styles=True)
