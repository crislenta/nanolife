"""Scenario loader — finds and parses scenario JSON files.

Searches the scenarios/ directory tree for a named scenario and
returns a hydrated Scenario dataclass.
"""
from __future__ import annotations

import json
from pathlib import Path

from .interfaces import Scenario


SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def find_scenario(name: str) -> Path:
    """Search for a scenario JSON by name across all subdirectories."""
    # Check base scenarios
    base = SCENARIOS_DIR / "base" / f"{name}.json"
    if base.exists():
        return base

    # Check named scenario directories
    named = SCENARIOS_DIR / name / "scenario.json"
    if named.exists():
        return named

    # Exhaustive search
    for p in SCENARIOS_DIR.rglob("*.json"):
        data = json.loads(p.read_text())
        if data.get("name") == name:
            return p

    raise FileNotFoundError(f"Scenario '{name}' not found in {SCENARIOS_DIR}")


def load_scenario(name: str) -> Scenario:
    """Load a scenario from JSON file."""
    path = find_scenario(name)
    data = json.loads(path.read_text())

    return Scenario(
        name=data["name"],
        description=data.get("description", ""),
        theme=data.get("theme", ""),
        harshness=data.get("harshness", 0.5),
        tick_unit=data.get("tick_unit", "4h"),
        locations=data.get("locations", ["Central Hub", "The Wilds", "Outskirts"]),
        agents=data.get("agents", []),
        opening_events=data.get("opening_events", []),
        metadata=data.get("metadata", {}),
        location_coords=data.get("location_coords", {}),
        base_drain=data.get("base_drain", 1.0),
        base_gain=data.get("base_gain", 2.5),
        reputation_decay=data.get("reputation_decay", 0.02),
        starting_resources=data.get("starting_resources", 15.0),
        world=data.get("world"),
    )
