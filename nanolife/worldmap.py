"""Grid world state for terminal rendering.

Pure data: a grid of Tiles parsed from ASCII. Used only by the renderer.
The engine does not read this module; spatial behavior is a future concern.

Note: this module is named ``worldmap`` (not ``world``) because
``nanolife.world`` is already taken by the simulation's shared WorldState.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
