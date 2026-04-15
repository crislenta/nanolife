"""Rich terminal dashboard for live simulation display.

Provides a full-screen TUI with a world overview, agent roster, event feed,
procedural minimap, and keyboard-navigable detail views.
"""
from __future__ import annotations

import math
import random
import sys
import termios
import threading
import time
import tty
from typing import Any

from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .common import Agent, Event, TickResult
from .world import WorldState


_GREEN = "#00ff88"
_RED = "#ff3344"
_CYAN = "#00ccff"
_GOLD = "#c9a84c"
_MAGENTA = "#cc66ff"
_DIM = "#555555"
_ORANGE = "#ff8800"
_BG_PANEL = "#111118"


# ── View modes ────────────────────────────────────────────────
# "world"  → default overview
# "births" → birth log
# "deaths" → death log
# "alive"  → all living agents list
# ("agent", idx) → specific NPC detail view

VIEW_CYCLE = ["world", "births", "deaths", "alive"]


class Dashboard:
    """Cinematic Rich terminal dashboard for nanolife."""

    MAX_FEED = 40

    def __init__(self, world: WorldState, scenario_name: str = "nanolife"):
        self.world = world
        self.scenario_name = scenario_name
        self.console = Console()
        self._all_events: list[Event] = []
        self._tick_events: list[Event] = []
        self._ei_score = 0
        self._ei_detected: list[str] = []
        self._cost: float = 0.0
        self._last_death: Event | None = None
        self._total_transfers: int = 0
        self._total_transfer_volume: float = 0.0
        self._work_ticks: int = 0
        self._action_ticks: int = 0

        # Minimap state
        self._map_terrain_cache: list[list[tuple[str, str]]] | None = None
        self._map_flash_locs: dict[str, tuple[str, int]] = {}  # loc -> (type, ttl)

        # View state: managed by keyboard input
        self._view: str = "world"       # current view key
        self._agent_idx: int = 0        # which agent when view == "agent"
        self._view_pos: int = -1        # -1 = world, 0..2 = births/deaths/alive, 3+ = agents

    @property
    def view_label(self) -> str:
        if self._view == "world":
            return "OVERVIEW"
        if self._view == "births":
            return "BIRTHS"
        if self._view == "deaths":
            return "DEATHS"
        if self._view == "alive":
            return "ALIVE"
        if self._view == "agent":
            agents = sorted(self.world.agents, key=lambda a: (not a.alive, a.name))
            if 0 <= self._agent_idx < len(agents):
                return agents[self._agent_idx].name.upper()
            return "AGENT"
        return "OVERVIEW"

    def navigate(self, direction: int) -> None:
        """Move view left (-1) or right (+1)."""
        all_agents = sorted(self.world.agents, key=lambda a: (not a.alive, a.name))
        total_agent_views = len(all_agents)
        max_pos = len(VIEW_CYCLE) - 1 + total_agent_views  # world(-1), births(0), deaths(1), alive(2), agent0(3)...

        self._view_pos += direction
        if self._view_pos < -1:
            self._view_pos = max_pos
        elif self._view_pos > max_pos:
            self._view_pos = -1

        if self._view_pos == -1:
            self._view = "world"
        elif self._view_pos < len(VIEW_CYCLE):
            self._view = VIEW_CYCLE[self._view_pos]
        else:
            self._view = "agent"
            self._agent_idx = self._view_pos - len(VIEW_CYCLE)

    def reset_view(self) -> None:
        """Back to world overview."""
        self._view = "world"
        self._view_pos = -1

    # ── Layout ────────────────────────────────────────────────

    def make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )

        if self._view == "world":
            layout["body"].split_row(
                Layout(name="left", size=18),
                Layout(name="center", ratio=2),
                Layout(name="right", size=32),
            )
            layout["center"].split_column(
                Layout(name="feed", ratio=1),
                Layout(name="minimap", size=16),
            )
            layout["right"].split_column(
                Layout(name="agents", ratio=1),
                Layout(name="spotlight", size=10),
            )
        else:
            layout["body"].split_row(
                Layout(name="left", size=18),
                Layout(name="detail", ratio=3),
            )

        return layout

    # ── Header ────────────────────────────────────────────────

    def header_panel(self) -> Panel:
        cal = self.world.clock.calendar()
        tick = self.world.clock.tick

        left = Text()
        left.append("  NANOLIFE", style="bold white")
        left.append(f" · {self.scenario_name}", style=f"bold {_GOLD}")

        mid = Text()
        mid.append(f"  day {cal['day']}", style="white")
        mid.append(f"  month {cal['month']}", style="white")
        mid.append(f"  year {cal['year']}", style="white")
        mid.append(f"  T{tick:04d}", style="dim")

        right = Text()
        alive = len(self.world.alive_agents)
        right.append(f"  pop ", style="dim")
        right.append(f"{alive}", style=f"bold {_GREEN}")
        right.append(f"  born ", style="dim")
        right.append(f"{self.world.total_births}", style=_CYAN)
        right.append(f"  dead ", style="dim")
        right.append(f"{self.world.total_deaths}", style=_RED)
        right.append(f"  ${self._cost:.2f}  ", style=_ORANGE)

        row = Table.grid(expand=True)
        row.add_column(ratio=1)
        row.add_column(ratio=1, justify="center")
        row.add_column(ratio=1, justify="right")
        row.add_row(left, mid, right)

        border = _RED if self._last_death else _GREEN
        return Panel(row, style=border, height=3)

    # ── World Stats (left sidebar) ────────────────────────────

    def stats_panel(self) -> Panel:
        alive = len(self.world.alive_agents)
        dead = self.world.total_deaths
        born = self.world.total_births
        tick = self.world.clock.tick
        h = self.world.harshness

        t = Table.grid(padding=(0, 1))
        t.add_column(width=10)
        t.add_column(justify="right", width=5)

        total_res = sum(a.resources for a in self.world.alive_agents)
        work_rate = self._work_ticks / self._action_ticks if self._action_ticks else 0

        t.add_row(Text("population", style="dim"), Text(str(alive), style=f"bold {_GREEN}"))
        t.add_row(Text("births", style="dim"), Text(str(born), style=_CYAN))
        t.add_row(Text("deaths", style="dim"), Text(str(dead), style=_RED))
        t.add_row(Text("tick", style="dim"), Text(str(tick), style="dim"))
        t.add_row(Text(""), Text(""))

        res_style = _GREEN if total_res > alive * 5 else (_ORANGE if total_res > alive * 2 else _RED)
        t.add_row(Text("resources ($)", style="dim"), Text(f"{total_res:.0f}", style=res_style))
        t.add_row(Text("transfers", style="dim"), Text(str(self._total_transfers), style=_GOLD))
        t.add_row(Text("work rate", style="dim"), Text(f"{work_rate:.0%}", style="dim"))
        t.add_row(Text(""), Text(""))

        bar_len = 10
        filled = int(h * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        bar_style = _GREEN if h < 0.4 else (_ORANGE if h < 0.7 else _RED)
        t.add_row(Text("harshness", style="dim"), Text(f"{h:.1f}", style=bar_style))
        t.add_row(Text(""), Text(bar, style=bar_style))
        t.add_row(Text(""), Text(""))

        loc_counts: dict[str, int] = {}
        for a in self.world.alive_agents:
            loc = a.location or "?"
            loc_counts[loc] = loc_counts.get(loc, 0) + 1

        for loc in sorted(loc_counts, key=lambda l: -loc_counts[l]):
            name = loc[:12] if len(loc) > 12 else loc
            t.add_row(Text(name, style=_GOLD), Text(str(loc_counts[loc]), style="white"))

        return Panel(t, title="[bold]WORLD[/bold]", border_style=_GREEN, style=f"on {_BG_PANEL}")

    # ── Feed (center, world view only) ────────────────────────

    def feed_panel(self) -> Panel:
        lines: list[Text] = []

        for e in self._all_events[-(self.MAX_FEED):]:
            line = self._format_event(e)
            lines.append(line)

        if not lines:
            lines.append(Text("  (awaiting signal...)", style="dim"))

        return Panel(Group(*lines), title="[bold]FEED[/bold]", border_style="yellow", style=f"on {_BG_PANEL}")

    # ── Agents list (right top, world view only) ──────────────

    def agents_panel(self) -> Panel:
        alive = sorted(self.world.alive_agents, key=lambda a: -a.reputation)
        dead = sorted(self.world.dead_agents, key=lambda a: -(a.death_tick or 0))

        t = Table.grid(padding=(0, 1))
        t.add_column(width=1)
        t.add_column(width=10)
        t.add_column(width=5, justify="right")
        t.add_column(width=5, justify="right")
        t.add_column(width=6)

        for a in alive:
            loc = (a.location or "?")[:6]
            rep_style = _GREEN if a.reputation > 0.5 else (_RED if a.reputation < 0.3 else "yellow")
            res_style = _GREEN if a.resources > 10 else (_ORANGE if a.resources > 3 else _RED)
            t.add_row(
                Text("●", style=_GREEN),
                Text(a.name[:10], style="white"),
                Text(f"{a.reputation:.2f}", style=rep_style),
                Text(f"${a.resources:.0f}", style=res_style),
                Text(loc, style="dim"),
            )

        for a in dead[:5]:
            t.add_row(
                Text("✕", style=_RED),
                Text(a.name[:10], style="dim strike"),
                Text("dead", style="dim"),
                Text("", style="dim"),
                Text("", style="dim"),
            )

        if alive:
            leader = alive[0]
            pariah = alive[-1]
            t.add_row(Text(""), Text(""), Text(""), Text(""))
            t.add_row(
                Text("♛", style=_GOLD),
                Text(leader.name[:10], style=f"bold {_GOLD}"),
                Text(f"{leader.reputation:.2f}", style=_GREEN),
                Text("leader", style="dim"),
            )
            if len(alive) > 1 and pariah.reputation < 0.4:
                t.add_row(
                    Text("▼", style=_RED),
                    Text(pariah.name[:10], style=_RED),
                    Text(f"{pariah.reputation:.2f}", style=_RED),
                    Text("pariah", style="dim"),
                )

        return Panel(t, title="[bold]AGENTS[/bold]", border_style=_MAGENTA, style=f"on {_BG_PANEL}")

    # ── Spotlight (right bottom, world view only) ─────────────

    def spotlight_panel(self) -> Panel:
        if self._last_death:
            return self._death_spotlight(self._last_death)

        alive = self.world.alive_agents
        if not alive:
            return Panel(Text("  All have perished.", style=f"dim {_RED}"), title="SPOTLIGHT", border_style="dim", style=f"on {_BG_PANEL}")

        hint = Text()
        hint.append("  ← → ", style=f"bold {_CYAN}")
        hint.append("cycle views", style="dim")
        hint.append("   Esc ", style=f"bold {_CYAN}")
        hint.append("overview", style="dim")
        return Panel(hint, title="[bold]NAVIGATE[/bold]", border_style="dim", style=f"on {_BG_PANEL}")

    def _death_spotlight(self, e: Event) -> Panel:
        name = "?"
        for a in self.world.agents:
            if a.id == e.get("agent"):
                name = a.name
                break
        cause = e.get("cause", "unknown")
        age = e.get("age_days", "?")
        lines = [
            Text(f"  ☠  {name.upper()}", style=f"bold {_RED}"),
            Text(f"  cause: {cause}", style=_RED),
            Text(f"  age: {age} ticks", style="dim"),
            Text(f'  "{e.get("content", "")[:60]}"', style="dim italic"),
        ]
        return Panel(Group(*lines), title="[bold]DEATH[/bold]", border_style=_RED, style=f"on {_BG_PANEL}")

    # ── Minimap ───────────────────────────────────────────────

    _BIOME_CHARS: dict[str, list[tuple[str, str]]] = {
        "ice":      [('▓', '#4466aa'), ('▒', '#3355aa'), ('░', '#334488'), ('*', '#6688cc')],
        "forest":   [('♠', '#1a4a1a'), ('♣', '#1a3a1a'), ('▒', '#1e3e1e'), ('░', '#1a2a1a')],
        "mountain": [('^', '#5a5a6a'), ('▲', '#4a4a5a'), ('▒', '#444455'), ('░', '#3a3a4a')],
        "plains":   [('·', '#2a3a1a'), ('░', '#1e2e1a'), (',', '#2a3a1a'), (' ', '')],
        "urban":    [('▪', '#4a3a28'), ('░', '#3a2a1a'), ('·', '#44382a'), ('#', '#3a3020')],
        "coastal":  [('≈', '#1a4466'), ('~', '#1a3a55'), ('░', '#1a2a3a'), ('·', '#2a3a3a')],
    }
    _DEFAULT_BIOME = [('░', '#1e2e1e'), ('·', '#2a2a2a'), (' ', ''), (' ', '')]

    @staticmethod
    def _value_noise_2d(x: float, y: float, seed: int) -> float:
        """Simple value noise: smooth random field in [0, 1]."""
        ix, iy = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - ix, y - iy
        fx = fx * fx * (3 - 2 * fx)
        fy = fy * fy * (3 - 2 * fy)

        def _hash(px: int, py: int) -> float:
            n = (px * 374761393 + py * 668265263 + seed) & 0xFFFFFFFF
            n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
            return (n & 0xFFFF) / 0xFFFF

        v00 = _hash(ix, iy)
        v10 = _hash(ix + 1, iy)
        v01 = _hash(ix, iy + 1)
        v11 = _hash(ix + 1, iy + 1)
        return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy

    def _gen_terrain(self, w: int, h: int) -> list[list[tuple[str, str]]]:
        """Generate coherent terrain with biome influence from location coords."""
        seed = hash(self.scenario_name) & 0xFFFFFFFF
        rng = random.Random(seed)
        coords = self.world.location_coords

        canvas: list[list[tuple[str, str]]] = [[(' ', '')] * w for _ in range(h)]

        for y in range(h):
            for x in range(w):
                nx, ny = x / w, y / h

                # Multi-octave value noise for organic shapes
                n = (
                    self._value_noise_2d(nx * 6, ny * 6, seed) * 0.5
                    + self._value_noise_2d(nx * 12, ny * 12, seed + 1) * 0.3
                    + self._value_noise_2d(nx * 24, ny * 24, seed + 2) * 0.2
                )

                # Coastline: fade to water near edges
                edge_dist = min(nx, 1 - nx, ny * 1.5, (1 - ny) * 1.2)
                coast_fade = min(1.0, edge_dist * 5)

                if coast_fade < 0.35 + n * 0.2:
                    water_chars = [('≈', '#1a3a66'), ('~', '#1a2a55'), ('≈', '#223a5a'), ('·', '#1a2a44')]
                    canvas[y][x] = rng.choice(water_chars)
                    continue

                # Find nearest location for biome blending
                best_biome = self._DEFAULT_BIOME
                best_dist = 999.0
                if coords:
                    for loc, info in coords.items():
                        lx, ly = info.get("x", 0.5), info.get("y", 0.5)
                        d = math.sqrt((nx - lx) ** 2 + (ny - ly) ** 2)
                        if d < best_dist:
                            best_dist = d
                            biome_key = info.get("biome", "")
                            best_biome = self._BIOME_CHARS.get(biome_key, self._DEFAULT_BIOME)

                # Mix in elevation-based defaults: north = colder, south = warmer
                if not coords:
                    if ny < 0.2:
                        best_biome = self._BIOME_CHARS["ice"]
                    elif ny < 0.4:
                        best_biome = self._BIOME_CHARS["forest"]
                    elif ny > 0.85:
                        best_biome = self._BIOME_CHARS["coastal"]
                    elif n > 0.65:
                        best_biome = self._BIOME_CHARS["mountain"]
                    else:
                        best_biome = self._DEFAULT_BIOME

                # Use noise to pick char from biome palette (avoids uniform static)
                idx = int(n * len(best_biome)) % len(best_biome)
                canvas[y][x] = best_biome[idx]

        return canvas

    def _compute_map_positions(self, w: int, h: int) -> dict[str, tuple[int, int]]:
        """Position locations using scenario coords if available, else auto-layout."""
        locations = self.world.locations
        coords = self.world.location_coords
        n = len(locations)
        if n == 0:
            return {}

        margin_x, margin_y = 2, 1
        usable_w = w - margin_x * 2 - 14
        usable_h = h - margin_y * 2 - 1

        if coords:
            positions: dict[str, tuple[int, int]] = {}
            for loc in locations:
                info = coords.get(loc)
                if info:
                    x = margin_x + int(info["x"] * usable_w)
                    y = margin_y + int(info["y"] * usable_h)
                else:
                    h_val = hash(loc) & 0xFFFFFFFF
                    x = margin_x + (h_val % usable_w)
                    y = margin_y + ((h_val >> 12) % usable_h)
                positions[loc] = (x, y)
            return positions

        # Auto-layout: grid with organic jitter
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        cell_w = max(16, (w - 2) // cols)
        cell_h = max(3, (h - 1) // max(rows, 1))
        rng = random.Random(hash(tuple(sorted(locations))) ^ 0xBEEF)

        positions = {}
        for i, loc in enumerate(locations):
            col = i % cols
            row = i // cols
            max_dx = max(0, cell_w - min(len(loc), 14) - 4)
            max_dy = max(0, cell_h - 3)
            x = 1 + col * cell_w + rng.randint(0, max_dx)
            y = row * cell_h + rng.randint(0, max_dy)
            positions[loc] = (x, y)
        return positions

    def _draw_road(
        self,
        canvas: list[list[tuple[str, str]]],
        x1: int, y1: int, x2: int, y2: int,
        w: int, h: int,
    ) -> None:
        """L-shaped road that routes horizontally then vertically."""
        passable = {' ', '░', '·', ',', '^', '~', '≈', '▒', '▓', '*', '♠', '♣'}
        mid_x = x2
        # Horizontal leg
        sx, ex = (x1, mid_x) if x1 < mid_x else (mid_x, x1)
        for x in range(sx, ex + 1):
            if 0 <= y1 < h and 0 <= x < w and canvas[y1][x][0] in passable:
                canvas[y1][x] = ('─', '#554433')
        # Vertical leg
        sy, ey = (y1, y2) if y1 < y2 else (y2, y1)
        for y in range(sy, ey + 1):
            if 0 <= y < h and 0 <= mid_x < w and canvas[y][mid_x][0] in passable:
                canvas[y][mid_x] = ('│', '#554433')
        # Corner
        if 0 <= y1 < h and 0 <= mid_x < w:
            if canvas[y1][mid_x][0] in passable | {'─', '│'}:
                canvas[y1][mid_x] = ('┘' if y2 < y1 and mid_x < x1
                                     else '└' if y2 < y1
                                     else '┐' if mid_x < x1
                                     else '┌', '#554433')

    def minimap_panel(self) -> Panel:
        W, H = 52, 12

        # Cache terrain (regenerated only once or when locations change)
        if self._map_terrain_cache is None or len(self._map_terrain_cache) != H or len(self._map_terrain_cache[0]) != W:
            self._map_terrain_cache = self._gen_terrain(W, H)

        # Deep copy terrain so we don't mutate the cache
        canvas: list[list[tuple[str, str]]] = [row[:] for row in self._map_terrain_cache]

        positions = self._compute_map_positions(W, H)

        # Agents per location
        loc_agents: dict[str, list[Agent]] = {}
        for a in self.world.alive_agents:
            loc = a.location or "?"
            loc_agents.setdefault(loc, []).append(a)
        loc_dead: dict[str, int] = {}
        for a in self.world.dead_agents:
            loc = a.location or "?"
            loc_dead[loc] = loc_dead.get(loc, 0) + 1

        # Decay flash timers
        expired = [k for k, (_, ttl) in self._map_flash_locs.items() if ttl <= 0]
        for k in expired:
            del self._map_flash_locs[k]
        for k in list(self._map_flash_locs):
            ft, ttl = self._map_flash_locs[k]
            self._map_flash_locs[k] = (ft, ttl - 1)

        # Register new flashes from this tick
        for e in self._tick_events:
            loc_id = None
            if e.get("type") == "death":
                agent_id = e.get("agent", "")
                for a in self.world.agents:
                    if a.id == agent_id:
                        loc_id = a.location
                        break
                if loc_id:
                    self._map_flash_locs[loc_id] = ("death", 3)
            elif e.get("type") == "birth":
                agent_id = e.get("agent", "")
                for a in self.world.agents:
                    if a.id == agent_id:
                        loc_id = a.location
                        break
                if loc_id:
                    self._map_flash_locs[loc_id] = ("birth", 2)

        # Population glow: draw halos around busy locations
        for loc, agents in loc_agents.items():
            if loc not in positions:
                continue
            cx, cy = positions[loc]
            count = len(agents)
            if count >= 3:
                glow_chars = [('░', '#2a2a18'), ('·', '#222218')]
                radius = min(3, 1 + count // 3)
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius * 2, radius * 2 + 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            dist = abs(dy) + abs(dx) // 2
                            if dist >= radius - 1 and canvas[ny][nx][0] in {' ', '░', '·', ',', '~', '≈', '▒'}:
                                canvas[ny][nx] = glow_chars[dist % 2]

        # Roads: connect each location to its 2 nearest neighbours
        loc_list = list(positions.keys())
        for i, loc_a in enumerate(loc_list):
            ax, ay = positions[loc_a]
            dists = []
            for j, loc_b in enumerate(loc_list):
                if i == j:
                    continue
                bx, by = positions[loc_b]
                dists.append((abs(ax - bx) + abs(ay - by), loc_b))
            dists.sort()
            for _, loc_b in dists[:2]:
                bx, by = positions[loc_b]
                self._draw_road(canvas, ax + 1, ay, bx + 1, by, W, H)

        # Crown: mark leader's location
        leader_loc: str | None = None
        alive = self.world.alive_agents
        if alive:
            leader = max(alive, key=lambda a: a.reputation)
            if leader.reputation > 0.4:
                leader_loc = leader.location

        # Pass 1: clear areas and draw location markers/names
        for loc, (cx, cy) in positions.items():
            name = loc[:14]
            agents_here = loc_agents.get(loc, [])

            clear_w = max(len(name) + 4, len(agents_here) * 2 + 4)
            for dy in range(-1, 2):
                for dx in range(-1, clear_w + 1):
                    ny, nx = cy + dy, cx - 1 + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        canvas[ny][nx] = (' ', '')

            is_flash = loc in self._map_flash_locs
            flash_type = self._map_flash_locs.get(loc, (None, 0))[0]

            if is_flash and flash_type == "death":
                marker_ch, marker_style = '☠', f'bold {_RED}'
                name_style = _RED
            elif is_flash and flash_type == "birth":
                marker_ch, marker_style = '✦', f'bold {_GREEN}'
                name_style = _GREEN
            elif loc == leader_loc:
                marker_ch, marker_style = '♛', f'bold {_GOLD}'
                name_style = f'bold {_GOLD}'
            else:
                marker_ch, marker_style = '◆', f'bold {_GOLD}'
                name_style = _GOLD

            if 0 <= cy < H and 0 <= cx < W:
                canvas[cy][cx] = (marker_ch, marker_style)
            for k, ch in enumerate(name):
                nx = cx + 2 + k
                if 0 <= nx < W and 0 <= cy < H:
                    canvas[cy][nx] = (ch, name_style)

        # Pass 2: draw agents AFTER all clearing is done
        for loc, (cx, cy) in positions.items():
            agents_here = loc_agents.get(loc, [])
            dead_here = loc_dead.get(loc, 0)

            if cy + 1 < H:
                for k, a in enumerate(agents_here[:10]):
                    nx = cx + 2 + k * 2
                    if 0 <= nx < W:
                        if a.reputation > 0.6:
                            col = _GREEN
                        elif a.reputation > 0.35:
                            col = _GOLD
                        elif a.reputation > 0.1:
                            col = _ORANGE
                        else:
                            col = _RED
                        canvas[cy + 1][nx] = (a.name[0].upper(), f'bold {col}')

                if dead_here:
                    offset = max(0, len(agents_here[:10])) * 2 + 2
                    nx = cx + 2 + offset
                    skull = f'✕{dead_here}'
                    for k, sch in enumerate(skull):
                        if 0 <= nx + k < W:
                            canvas[cy + 1][nx + k] = (sch, _RED)

        # Compass rose (top-right corner)
        compass = [('N', f'bold {_GOLD}')]
        rose_x, rose_y = W - 2, 0
        if 0 <= rose_y < H and 0 <= rose_x < W:
            canvas[rose_y][rose_x] = compass[0]
        if rose_y + 1 < H and rose_x < W:
            canvas[rose_y + 1][rose_x] = ('│', '#555544')
        if rose_y + 1 < H and rose_x - 1 >= 0:
            canvas[rose_y + 1][rose_x - 1] = ('W', '#444433')
        if rose_y + 1 < H and rose_x + 1 < W:
            canvas[rose_y + 1][rose_x + 1] = ('E', '#444433')
        if rose_y + 2 < H and rose_x < W:
            canvas[rose_y + 2][rose_x] = ('S', '#444433')

        # Render canvas to Rich Text
        lines: list[Text] = []
        for y in range(H):
            line = Text()
            for x in range(W):
                ch, style = canvas[y][x]
                line.append(ch, style=style or 'dim')
            lines.append(line)

        # Legend line
        legend = Text()
        legend.append("  ◆", style=_GOLD)
        legend.append(" loc  ", style="dim")
        legend.append("A", style=f"bold {_GREEN}")
        legend.append("/", style="dim")
        legend.append("A", style=f"bold {_RED}")
        legend.append(" rep  ", style="dim")
        legend.append("♛", style=f"bold {_GOLD}")
        legend.append(" leader  ", style="dim")
        legend.append("☠", style=_RED)
        legend.append(" death  ", style="dim")
        legend.append("─", style="#554433")
        legend.append(" road", style="dim")
        lines.append(legend)

        return Panel(
            Group(*lines),
            title="[bold]MAP[/bold]",
            border_style='#335533',
            style=f"on {_BG_PANEL}",
        )

    # ── Detail views (births, deaths, alive, agent) ───────────

    def detail_panel(self) -> Panel:
        if self._view == "births":
            return self._births_detail()
        elif self._view == "deaths":
            return self._deaths_detail()
        elif self._view == "alive":
            return self._alive_detail()
        elif self._view == "agent":
            return self._agent_detail()
        return Panel(Text(""), border_style="dim")

    def _births_detail(self) -> Panel:
        births = [e for e in self._all_events if e.get("type") == "birth"]
        lines: list[Text] = []
        for e in births[-30:]:
            tick = e.get("tick", 0)
            content = e.get("content", "")
            line = Text()
            line.append(f"  {tick:04d}  ", style="dim")
            line.append(f"+ {content}", style=_GREEN)
            lines.append(line)
        if not lines:
            lines.append(Text("  No births yet.", style="dim"))
        return Panel(Group(*lines), title=f"[bold]BIRTHS[/bold]  ({len(births)} total)", border_style=_GREEN, style=f"on {_BG_PANEL}")

    def _deaths_detail(self) -> Panel:
        deaths = [e for e in self._all_events if e.get("type") == "death"]
        lines: list[Text] = []
        for e in deaths[-30:]:
            tick = e.get("tick", 0)
            content = e.get("content", "")
            cause = e.get("cause", "?")
            age = e.get("age_days", "?")
            line = Text()
            line.append(f"  {tick:04d}  ", style="dim")
            line.append(f"☠ {content}", style=_RED)
            line.append(f"  ({cause}, age {age})", style="dim")
            lines.append(line)
        if not lines:
            lines.append(Text("  No deaths yet.", style="dim"))
        return Panel(Group(*lines), title=f"[bold]DEATHS[/bold]  ({len(deaths)} total)", border_style=_RED, style=f"on {_BG_PANEL}")

    def _alive_detail(self) -> Panel:
        alive = sorted(self.world.alive_agents, key=lambda a: -a.resources)
        lines: list[Text] = []
        for a in alive:
            line = Text()
            rep_style = _GREEN if a.reputation > 0.5 else (_RED if a.reputation < 0.3 else "yellow")
            res_style = _GREEN if a.resources > 10 else (_ORANGE if a.resources > 3 else _RED)
            line.append(f"  ● ", style=_GREEN)
            line.append(f"{a.name:<14s}", style="bold white")
            line.append(f"${a.resources:<6.0f}", style=res_style)
            line.append(f"rep {a.reputation:+.2f}  ", style=rep_style)
            line.append(f"loc {(a.location or '?')[:12]:<12s}  ", style="dim")
            line.append(f"goal: {a.goal[:25]}", style="dim")
            lines.append(line)

            # Friendships
            friend_names = []
            for fid in a.friendships[:4]:
                for ag in self.world.agents:
                    if ag.id == fid:
                        friend_names.append(ag.name)
                        break
            if friend_names:
                lines.append(Text(f"       friends: {', '.join(friend_names)}", style=_MAGENTA))

        if not lines:
            lines.append(Text("  No one alive.", style=f"dim {_RED}"))
        return Panel(Group(*lines), title=f"[bold]ALIVE[/bold]  ({len(alive)} agents)", border_style=_CYAN, style=f"on {_BG_PANEL}")

    def _agent_detail(self) -> Panel:
        agents = sorted(self.world.agents, key=lambda a: (not a.alive, a.name))
        if not agents or self._agent_idx >= len(agents):
            return Panel(Text("  No agent selected.", style="dim"), border_style="dim")

        a = agents[self._agent_idx]
        alive_str = "ALIVE" if a.alive else "DEAD"
        border = _GREEN if a.alive else _RED
        rep_style = _GREEN if a.reputation > 0.5 else (_RED if a.reputation < 0.3 else "yellow")

        lines: list[Text] = []
        lines.append(Text(f"  {a.name.upper()}", style=f"bold {border}"))
        lines.append(Text(f"  status: {alive_str}   rep: {a.reputation:.2f}   loc: {a.location or '?'}", style=rep_style))
        lines.append(Text(f"  goal: {a.goal}", style="dim"))
        lines.append(Text(""))

        # Resource & economics
        res_style = _GREEN if a.resources > 10 else (_ORANGE if a.resources > 3 else _RED)
        res_line = Text()
        res_line.append(f"  resource: ", style="dim")
        res_line.append(f"${a.resources:.1f}", style=f"bold {res_style}")

        agent_actions = [e for e in self._all_events if e.get("type") == "action" and e.get("agent") == a.id]
        agent_work = sum(1 for e in agent_actions if e.get("working"))
        work_rate = agent_work / len(agent_actions) if agent_actions else 0
        res_line.append(f"   work: {work_rate:.0%}", style="dim")

        agent_transfers_out = [e for e in self._all_events if e.get("type") == "transfer" and e.get("agent") == a.id]
        agent_transfers_in = [e for e in self._all_events if e.get("type") == "transfer" and e.get("target") == a.id]
        total_out = sum(e.get("amount", 0) for e in agent_transfers_out)
        total_in = sum(e.get("amount", 0) for e in agent_transfers_in)
        res_line.append(f"   sent: {total_out:.0f}  recv: {total_in:.0f}", style="dim")
        lines.append(res_line)
        lines.append(Text(""))

        # Traits
        if a.traits:
            trait_str = "  ".join(f"{k}: {v:.2f}" for k, v in a.traits.items())
            lines.append(Text(f"  traits: {trait_str}", style="dim"))

        # Friendships
        friend_names = []
        for fid in a.friendships:
            for ag in self.world.agents:
                if ag.id == fid:
                    friend_names.append(ag.name)
                    break
        if friend_names:
            lines.append(Text(f"  friends: {', '.join(friend_names)}", style=_MAGENTA))

        # Recent transfers
        recent_transfers = (agent_transfers_out + agent_transfers_in)
        recent_transfers.sort(key=lambda e: e.get("tick", 0))
        if recent_transfers:
            lines.append(Text(""))
            lines.append(Text("  TRANSFERS", style=f"bold {_GOLD}"))
            for te in recent_transfers[-5:]:
                sender_id = te.get("agent", "")
                target_id = te.get("target", "")
                amount = te.get("amount", 0)
                other_id = target_id if sender_id == a.id else sender_id
                other_name = "?"
                for ag in self.world.agents:
                    if ag.id == other_id:
                        other_name = ag.name
                        break
                direction = "→" if sender_id == a.id else "←"
                amt_style = _RED if sender_id == a.id else _GREEN
                tl = Text()
                tl.append(f"  T{te.get('tick', 0):04d} ", style="dim")
                tl.append(f"{direction} {other_name[:10]} ", style="white")
                tl.append(f"${abs(amount):.1f}", style=amt_style)
                lines.append(tl)

        # Identity
        if a.identity_md:
            sentences = [s.strip() for s in a.identity_md.split(".") if s.strip()]
            lines.append(Text(""))
            lines.append(Text("  IDENTITY", style=f"bold {_GOLD}"))
            for s in sentences[-5:]:
                lines.append(Text(f"  · {s[:70]}", style="dim"))

        # Death info
        if not a.alive and a.death_tick is not None:
            lines.append(Text(""))
            lines.append(Text(f"  ☠ died at tick {a.death_tick}, age {a.age(a.death_tick)} ticks", style=_RED))

        # Recent events for this agent
        agent_events = [e for e in self._all_events if e.get("agent") == a.id]
        if agent_events:
            lines.append(Text(""))
            lines.append(Text("  RECENT EVENTS", style=f"bold {_GOLD}"))
            for e in agent_events[-8:]:
                line = self._format_event(e, compact=True)
                lines.append(line)

        nav_idx = self._agent_idx + 1
        nav_total = len(agents)
        return Panel(
            Group(*lines),
            title=f"[bold]{a.name.upper()}[/bold]  ({nav_idx}/{nav_total})",
            border_style=border,
            style=f"on {_BG_PANEL}",
        )

    # ── Event formatter ───────────────────────────────────────

    def _format_event(self, e: Event, compact: bool = False) -> Text:
        t = e.get("type", "?")
        tick = e.get("tick", 0)
        tick_str = f"{tick:04d}"
        content = e.get("content", "")[:55]

        agent_id = e.get("agent", "")
        name = ""
        for a in self.world.agents:
            if a.id == agent_id:
                name = a.name
                break

        prefix = f" {tick_str} " if not compact else f"  {tick_str} "
        name_str = name.upper()[:10].ljust(10) if name and not compact else ""

        line = Text()
        line.append(prefix, style="#2a2a2a")

        if t == "death":
            line.append(f"{name_str} ", style=f"bold {_RED}")
            line.append(f"☠ {content}", style=_RED)
        elif t == "birth":
            line.append(f"{name_str} ", style=f"bold {_GREEN}")
            line.append(f"+ {content}", style=_GREEN)
        elif t == "transfer":
            amount = e.get("amount", 0)
            target_id = e.get("target", "")
            target_name = ""
            for a in self.world.agents:
                if a.id == target_id:
                    target_name = a.name
                    break
            amt_style = _GREEN if amount > 0 else _RED
            line.append(f"{name_str} ", style=_GOLD)
            line.append(f"$ {amount:+.1f}", style=amt_style)
            line.append(f" → {target_name[:10]}", style="white")
            if content:
                line.append(f"  {content[:30]}", style="dim")
        elif t == "friendship":
            line.append(f"{name_str} ", style=_MAGENTA)
            line.append(f"~ {content}", style="#2a8855")
        elif t == "reputation":
            delta = e.get("delta", 0)
            style = _GREEN if delta > 0 else _RED
            line.append(f"{name_str} ", style="white")
            line.append(f"★ {content} ({delta:+.2f})", style=style)
        elif t == "action":
            working = e.get("working", False)
            work_icon = "⚒" if working else "○"
            line.append(f"{name_str} ", style="white")
            line.append(f"{work_icon} {content}", style="#888888")
        elif t == "thought":
            line.append(f"{name_str} ", style="dim")
            line.append(f'"{content}"', style="dim italic")
        elif t == "move":
            line.append(f"{name_str} ", style="dim")
            line.append(f"→ {content}", style="dim")
        elif t == "rumor":
            line.append(f"{name_str} ", style="dim")
            line.append(f"≈ {content[:45]}...", style="dim italic")
        elif t == "compression":
            line.append("           ", style="dim")
            line.append("⟳ History compressed", style="dim yellow")
        elif t == "improvement":
            line.append(f"{name_str} ", style="dim")
            line.append(f"◆ {content[:45]}", style=f"dim {_CYAN}")
        else:
            line.append(f"           {content}", style="dim")

        return line

    # ── Emergence (footer) ────────────────────────────────────

    def footer_panel(self) -> Panel:
        text = Text()

        # View indicator
        text.append("  ◀ ", style=f"bold {_CYAN}")
        text.append(self.view_label, style="bold white")
        text.append(" ▶  ", style=f"bold {_CYAN}")
        text.append("│ ", style="dim")

        # Emergence
        max_ei = 11
        score = min(self._ei_score, max_ei)
        bar_len = 11
        filled = "█" * score
        empty = "░" * (bar_len - score)
        bar_style = _GREEN if score >= 5 else (_ORANGE if score >= 3 else "dim")
        text.append(filled, style=bar_style)
        text.append(empty, style="dim")
        text.append(f" {score}/{max_ei} ", style="bold white")

        _TAG_STYLES = {
            "death": _RED,
            "birth": _GREEN,
            "betrayal": _ORANGE,
            "ostracism": _ORANGE,
            "alliance": _CYAN,
            "leadership": _CYAN,
            "faction_split": _MAGENTA,
            "wealth_concentration": _GOLD,
            "economic_dependency": _GOLD,
            "resource_warfare": _RED,
            "free_riding": _ORANGE,
            "generational_transmission": _CYAN,
            "cultural_drift": _MAGENTA,
        }

        for tag in self._ei_detected:
            style = _TAG_STYLES.get(tag, "dim")
            text.append(f" {tag}", style=style)

        return Panel(text, style="dim", height=3)

    # ── Update ────────────────────────────────────────────────

    def update(self, result: TickResult, cost: float = 0.0) -> Layout:
        self._tick_events = result.events
        self._all_events.extend(result.events)
        self._cost = cost

        for e in result.events:
            if e.get("type") == "transfer":
                self._total_transfers += 1
                self._total_transfer_volume += abs(e.get("amount", 0))
            if e.get("type") == "action":
                self._action_ticks += 1
                if e.get("working"):
                    self._work_ticks += 1

        self._update_emergence(result)

        self._last_death = None
        for e in result.events:
            if e.get("type") == "death":
                self._last_death = e
                break

        layout = self.make_layout()
        layout["header"].update(self.header_panel())
        layout["left"].update(self.stats_panel())
        layout["footer"].update(self.footer_panel())

        if self._view == "world":
            layout["feed"].update(self.feed_panel())
            layout["minimap"].update(self.minimap_panel())
            layout["agents"].update(self.agents_panel())
            layout["spotlight"].update(self.spotlight_panel())
        else:
            layout["detail"].update(self.detail_panel())

        return layout

    # ── Emergence detection ───────────────────────────────────

    def _update_emergence(self, result: TickResult) -> None:
        detected = set(self._ei_detected)
        alive = self.world.alive_agents

        for a in alive:
            mutual = [f for f in a.friendships if any(fa for fa in alive if fa.id == f and a.id in fa.friendships)]
            if len(mutual) >= 2:
                detected.add("alliance")
                break

        if alive:
            leader = max(alive, key=lambda a: a.reputation)
            if leader.reputation > 0.7:
                detected.add("leadership")

        for e in result.events:
            if e.get("type") == "reputation" and e.get("delta", 0) < -0.2:
                src = e.get("source", "")
                for a in alive:
                    if a.id == src and any(src in fa.friendships for fa in alive):
                        detected.add("betrayal")
                        break

        for a in alive:
            if a.reputation < -0.5:
                detected.add("ostracism")

        if len(alive) >= 6:
            pos = [a for a in alive if a.reputation > 0.5]
            neg = [a for a in alive if a.reputation < 0.3]
            if len(pos) >= 2 and len(neg) >= 2:
                detected.add("faction_split")

        for e in result.events:
            if e.get("type") == "birth" and e.get("inherited_identity"):
                detected.add("generational_transmission")
                break

        if result.deaths > 0:
            detected.add("death")
        if result.births > 0:
            detected.add("birth")

        # Economic emergence (live)
        if alive and self._total_transfer_volume > 0:
            net_flow: dict[str, float] = {}
            for e in self._all_events:
                if e.get("type") == "transfer":
                    sender = e.get("agent", "")
                    target = e.get("target", "")
                    amount = e.get("amount", 0)
                    net_flow[sender] = net_flow.get(sender, 0) - amount
                    net_flow[target] = net_flow.get(target, 0) + amount

            if net_flow:
                max_gain = max(net_flow.values())
                if max_gain > self._total_transfer_volume * 0.3:
                    detected.add("wealth_concentration")

            # One-way flows >= 10 total
            pair_flows: dict[tuple[str, str], float] = {}
            for e in self._all_events:
                if e.get("type") == "transfer" and e.get("amount", 0) > 0:
                    key = (e.get("agent", ""), e.get("target", ""))
                    pair_flows[key] = pair_flows.get(key, 0) + e["amount"]
            if any(v >= 10.0 for v in pair_flows.values()):
                detected.add("economic_dependency")

        # Free riding: agent with low work rate surviving on transfers
        if self._action_ticks > 20:
            for a in alive:
                a_actions = [e for e in self._all_events if e.get("type") == "action" and e.get("agent") == a.id]
                a_work = sum(1 for e in a_actions if e.get("working"))
                if len(a_actions) >= 5:
                    rate = a_work / len(a_actions)
                    a_recv = sum(e.get("amount", 0) for e in self._all_events if e.get("type") == "transfer" and e.get("target") == a.id and e.get("amount", 0) > 0)
                    if rate < 0.3 and a_recv >= 5.0:
                        detected.add("free_riding")
                        break

        self._ei_detected = sorted(detected)
        self._ei_score = len(detected)


# ── Keyboard listener ─────────────────────────────────────────

class _KeyListener:
    """Non-blocking keyboard reader running in a background thread."""

    def __init__(self, dashboard: Dashboard):
        self.dashboard = dashboard
        self._running = False
        self._thread: threading.Thread | None = None
        self._old_settings: Any = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            return
        self._running = True
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass

    def _run(self) -> None:
        while self._running:
            try:
                ch = sys.stdin.read(1)
                if not ch:
                    continue

                if ch == "\x1b":
                    seq = sys.stdin.read(2)
                    if seq == "[C":       # right arrow
                        self.dashboard.navigate(1)
                    elif seq == "[D":     # left arrow
                        self.dashboard.navigate(-1)
                    elif seq == "" or seq[0] != "[":
                        # bare Esc
                        self.dashboard.reset_view()
                elif ch == "q" or ch == "Q":
                    self.dashboard.reset_view()
            except Exception:
                break


# ── Renderer ──────────────────────────────────────────────────

class TerminalRenderer:
    """Wraps Dashboard with rich.Live for fullscreen in-place rendering."""

    def __init__(self, world: WorldState, scenario_name: str = "nanolife"):
        self.dashboard = Dashboard(world, scenario_name)
        self.live: Live | None = None
        self._keys: _KeyListener | None = None

    def start(self) -> None:
        self.live = Live(
            self.dashboard.make_layout(),
            console=self.dashboard.console,
            refresh_per_second=4,
            screen=True,
        )
        self.live.start()
        self._keys = _KeyListener(self.dashboard)
        self._keys.start()

    def update(self, result: TickResult, cost: float = 0.0) -> None:
        layout = self.dashboard.update(result, cost)
        if self.live:
            self.live.update(layout)

        if result.deaths > 0:
            time.sleep(1.5)
        elif result.births > 0:
            time.sleep(0.6)
        else:
            time.sleep(0.3)

    def stop(self) -> None:
        if self._keys:
            self._keys.stop()
        if self.live:
            self.live.stop()
