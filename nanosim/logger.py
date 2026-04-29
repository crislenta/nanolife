"""JSONL run logger — persists events to disk.

Writes one world log and per-agent logs as append-only JSONL files
under the run directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .common import Event


class RunLogger:
    """Append-only JSONL writer. One file per agent + one world log."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.agents_dir = self.run_dir / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self._world_path = self.run_dir / "world.jsonl"

    def log_world(self, event: Event | dict[str, Any]) -> None:
        self._append(self._world_path, event)

    def log_agent(self, agent_id: str, event: Event | dict[str, Any]) -> None:
        path = self.agents_dir / f"{agent_id}.jsonl"
        self._append(path, event)

    def log(self, event: Event | dict[str, Any], agent_id: str | None = None) -> None:
        self.log_world(event)
        if agent_id:
            self.log_agent(agent_id, event)
        elif "agent" in event:
            self.log_agent(event["agent"], event)

    @staticmethod
    def _append(path: Path, data: dict[str, Any]) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def read_world(self) -> list[dict[str, Any]]:
        return self._read(self._world_path)

    def read_agent(self, agent_id: str) -> list[dict[str, Any]]:
        return self._read(self.agents_dir / f"{agent_id}.jsonl")

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        lines: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
        return lines
