"""Grid world state for rendering and spatial reasoning.

Pure data: a grid of Tiles parsed from ASCII, plus two small helpers
that enable optional spatial behavior (movement + local views).

The engine only *reads* this module when a scenario attaches a ``WorldMap``;
legacy scenarios (no ``world`` block) stay byte-identical.

Note: this module is named ``worldmap`` (not ``world``) because
``nanosim.world`` is already taken by the simulation's shared WorldState.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .common import Agent


# Terrains an agent cannot step onto. Kept as a module constant (not a
# scenario knob) to stay minimal — extend via Tile.terrain strings.
BLOCKED_TERRAINS: frozenset[str] = frozenset({"mountain", "water", "wall"})


@dataclass
class Tile:
    """A single map cell."""
    glyph: str
    terrain: str
    color: Optional[str] = None


@dataclass
class WorldMap:
    """Row-major grid of Tiles."""
    width: int
    height: int
    tiles: list[list[Tile]]

    @classmethod
    def from_ascii(cls, text: str, legend: dict[str, Tile]) -> "WorldMap":
        """Parse a multiline ASCII map using a glyph -> Tile legend.

        Missing glyphs fall back to a plain tile using the glyph itself.
        Rows are right-padded with spaces so the grid is rectangular.
        """
        lines = text.rstrip("\n").splitlines()
        if not lines:
            return cls(width=0, height=0, tiles=[])
        width = max(len(line) for line in lines)
        fallback = legend.get(" ", Tile(glyph=" ", terrain="empty"))
        rows: list[list[Tile]] = []
        for line in lines:
            padded = line.ljust(width)
            row: list[Tile] = []
            for ch in padded:
                proto = legend.get(ch)
                if proto is None:
                    row.append(Tile(glyph=ch, terrain=fallback.terrain, color=fallback.color))
                else:
                    # copy so mutations to a Tile don't ripple through the legend
                    row.append(Tile(glyph=proto.glyph, terrain=proto.terrain, color=proto.color))
            rows.append(row)
        return cls(width=width, height=len(rows), tiles=rows)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, x: int, y: int) -> bool:
        """True if (x, y) is in bounds and walkable."""
        if not self.in_bounds(x, y):
            return False
        return self.tiles[y][x].terrain not in BLOCKED_TERRAINS

    def local_view(
        self,
        position: tuple[int, int],
        radius: int,
        agents: list[Agent],
    ) -> str:
        """Return a small ASCII patch centered on ``position``.

        - Self is rendered as '@'.
        - Other alive agents are rendered using the first letter of their name.
        - Tiles beyond the map edge render as ' '.
        - Newlines separate rows; trailing whitespace is preserved so the
          LLM sees the patch is square.
        """
        cx, cy = position
        others: dict[tuple[int, int], str] = {}
        for a in agents:
            if not a.alive or a.position is None:
                continue
            if a.position == position:
                continue
            g = (a.name[0].upper() if a.name else "?")
            others[a.position] = g
        rows: list[str] = []
        for dy in range(-radius, radius + 1):
            y = cy + dy
            row_chars: list[str] = []
            for dx in range(-radius, radius + 1):
                x = cx + dx
                if dx == 0 and dy == 0:
                    row_chars.append("@")
                elif (x, y) in others:
                    row_chars.append(others[(x, y)])
                elif self.in_bounds(x, y):
                    row_chars.append(self.tiles[y][x].glyph)
                else:
                    row_chars.append(" ")
            rows.append("".join(row_chars))
        return "\n".join(rows)
