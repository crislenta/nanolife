"""Caves-of-Qud / Teleport-style terminal renderer.

Pure function: takes read-only state, returns a rendered string frame.
The engine never calls this; it's driven by the simulate script when
``--render`` is passed. Works with or without a WorldMap.

Layout (map-centric, Teleport aesthetic):

    +-----------------------------------------------------------+
    |  header: title . tick . pop . cost                        |
    +-----------+-----------------------------------+-----------+
    |           |                                   |           |
    |  roster   |               MAP                 |  status   |
    |  (left)   |           (center, big)           |  (right)  |
    |           |                                   |           |
    +-----------+-----------------------------------+-----------+
    |  event log (wide, bottom)                                 |
    +-----------------------------------------------------------+
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


# Palette — sparse, monochrome-leaning, amber/cyan accents.
_BORDER = "grey39"
_BORDER_ACCENT = "cyan"
_TITLE = "bold cyan"
_AGENT_STYLE = "bold yellow"


def _map_panel(world: WorldMap, agents: list[Agent]) -> Panel:
    """Render the map with agent glyphs overlaid on tiles."""
    overlay: dict[tuple[int, int], tuple[str, str]] = {}
    for a in agents:
        if a.position is None or not a.alive:
            continue
        x, y = a.position
        if 0 <= x < world.width and 0 <= y < world.height:
            glyph = a.name[0].upper() if a.name else "?"
            overlay[(x, y)] = (glyph, _AGENT_STYLE)

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

    return Panel(
        text,
        title="[bold]world[/bold]",
        border_style=_BORDER_ACCENT,
        padding=(0, 1),
    )


def _header_panel(tick: int, agents: list[Agent], cost: float | None) -> Panel:
    """Top status strip: title . tick . alive count . accumulated cost."""
    alive = sum(1 for a in agents if a.alive)
    total = len(agents)
    bits = [
        Text("nanolife", style=_TITLE),
        Text(f"tick {tick:04d}", style="bold white"),
        Text(f"pop {alive}/{total}", style="white"),
    ]
    if cost is not None:
        bits.append(Text(f"${cost:.3f}", style="green"))
    line = Text("   .   ").join(bits)
    return Panel(line, border_style=_BORDER_ACCENT, padding=(0, 1))


def _roster_panel(agents: list[Agent]) -> Panel:
    """Left side: who's alive, reputation, resources, short thought."""
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(style=_AGENT_STYLE, no_wrap=True, width=3)
    table.add_column(style="white", no_wrap=True)
    table.add_column(style="dim", no_wrap=False)

    for a in agents:
        if not a.alive:
            continue
        glyph = a.name[0].upper() if a.name else "?"
        rep = getattr(a, "reputation", None)
        res = getattr(a, "resources", None)
        stats: list[str] = []
        if rep is not None:
            stats.append(f"rep {rep:+.1f}")
        if res is not None:
            stats.append(f"res {res:.0f}")
        stat_line = " ".join(stats) if stats else ""
        thought = getattr(a, "last_thought", "") or ""
        # Trim noisy thoughts — the log shows full lines.
        if len(thought) > 40:
            thought = thought[:37] + "..."
        table.add_row(glyph, a.name, stat_line)
        if thought:
            table.add_row("", "", Text(thought, style="italic dim"))

    dead = [a for a in agents if not a.alive]
    if dead:
        table.add_row("", "", "")
        table.add_row(
            Text("x", style="red"),
            Text(f"dead ({len(dead)})", style="red"),
            Text(", ".join(a.name for a in dead), style="dim red"),
        )

    return Panel(
        table,
        title="[bold]agents[/bold]",
        border_style=_BORDER,
        padding=(1, 1),
    )


def _status_panel(agents: list[Agent], tick: int) -> Panel:
    """Right side: world vitals . recent action tallies."""
    alive = [a for a in agents if a.alive]
    action_counts: dict[str, int] = {}
    for a in alive:
        act = getattr(a, "last_action", None)
        if act:
            action_counts[act] = action_counts.get(act, 0) + 1

    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(style="white", no_wrap=True, justify="right")

    table.add_row("tick", str(tick))
    table.add_row("alive", str(len(alive)))
    if alive:
        avg_rep = sum(getattr(a, "reputation", 0.0) or 0.0 for a in alive) / len(alive)
        avg_res = sum(getattr(a, "resources", 0.0) or 0.0 for a in alive) / len(alive)
        table.add_row("avg rep", f"{avg_rep:+.2f}")
        table.add_row("avg res", f"{avg_res:.1f}")
    if action_counts:
        table.add_row("", "")
        table.add_row(Text("this tick", style="bold white"), "")
        for act, n in sorted(action_counts.items(), key=lambda kv: -kv[1]):
            table.add_row(act, str(n))

    return Panel(
        table,
        title="[bold]status[/bold]",
        border_style=_BORDER,
        padding=(1, 1),
    )


def _log_panel(event_log: list[str], lines: int = 8) -> Panel:
    """Bottom: wide event feed. Newest line last (terminal-native)."""
    tail = event_log[-lines:]
    if not tail:
        body = Text("(no events yet)", style="dim italic")
    else:
        body = Text()
        for i, line in enumerate(tail):
            if i > 0:
                body.append("\n")
            # Subtle fade for older lines.
            age = len(tail) - 1 - i
            style = "white" if age == 0 else ("grey70" if age < 3 else "grey50")
            body.append(line, style=style)
    return Panel(
        body,
        title="[bold]event log[/bold]",
        border_style=_BORDER,
        padding=(0, 1),
    )


def render(
    world: Optional[WorldMap],
    agents: list[Agent],
    event_log: list[str],
    tick: int,
    width: int = 140,
    height: int = 40,
    cost: float | None = None,
) -> str:
    """Render one frame; return ANSI-styled string.

    The layout puts the map dead center with panels framing it. When
    there's no world (legacy scenarios), we fall back to a roster+log
    split with no map.
    """
    console = Console(record=True, width=width, height=height, force_terminal=True)
    layout = Layout()

    # Three rows: header . body . log
    layout.split_column(
        Layout(_header_panel(tick, agents, cost), name="header", size=3),
        Layout(name="body"),
        Layout(_log_panel(event_log), name="log", size=10),
    )

    if world is not None:
        # Map dominates the center. Cap side panels so the map gets the
        # visual weight, Teleport-style. For small maps the panel size is
        # driven by available space; for large maps the panels stay thin.
        side_w = min(32, max(22, (width - (world.width + 4)) // 2))
        layout["body"].split_row(
            Layout(_roster_panel(agents), name="roster", size=side_w),
            Layout(_map_panel(world, agents), name="map"),
            Layout(_status_panel(agents, tick), name="status", size=side_w),
        )
    else:
        # No map — give roster/status equal billing, no center panel.
        layout["body"].split_row(
            Layout(_roster_panel(agents), name="roster", ratio=1),
            Layout(_status_panel(agents, tick), name="status", ratio=1),
        )

    console.print(layout)
    return console.export_text(styles=True)
