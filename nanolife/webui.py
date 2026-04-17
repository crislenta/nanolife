"""Real-time browser dashboard for a running simulation.

Streams world state to connected browsers over Server-Sent Events (SSE) after
every tick. Serves a single static page from `nanolife/static/index.html`.
The page renders a live feed, agent roster, social graph, and drama ticker.

Usage:
    from nanolife.webui import WebUI
    ui = WebUI(world, scenario_name="nanothrones")
    runner = await ui.start(port=8765)
    # in your tick callback: ui.broadcast(tick_result, cost=cognitive.total_cost)
    ...
    await runner.cleanup()
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from aiohttp import web

from .common import Agent, Event, TickResult
from .world import WorldState


_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _agent_dict(a: Agent) -> dict[str, Any]:
    return {
        "id": a.id,
        "name": a.name,
        "alive": a.alive,
        "reputation": round(a.reputation, 3),
        "resources": round(a.resources, 2),
        "location": a.location,
        "goal": a.goal,
        "traits": {k: round(v, 2) for k, v in a.traits.items()},
        "friendships": list(a.friendships),
        "pacts": list(a.pacts),
        "rivals": list(a.rivals),
        "birth_tick": a.birth_tick,
        "death_tick": a.death_tick,
    }


class WebUI:
    """SSE-based live dashboard. One instance per simulation."""

    def __init__(self, world: WorldState, scenario_name: str = "nanolife"):
        self.world = world
        self.scenario_name = scenario_name
        self._subscribers: set[asyncio.Queue] = set()
        self._cost: float = 0.0
        self._recent_events: list[Event] = []
        self._max_recent = 200

    # ── Snapshot / broadcast ──────────────────────────────────

    def _snapshot(self, tick_events: list[Event] | None = None) -> dict[str, Any]:
        cal = self.world.clock.calendar()
        return {
            "scenario": self.scenario_name,
            "tick": self.world.clock.tick,
            "calendar": cal,
            "population": len(self.world.alive_agents),
            "total_births": self.world.total_births,
            "total_deaths": self.world.total_deaths,
            "harshness": self.world.harshness,
            "cost": round(self._cost, 4),
            "agents": [_agent_dict(a) for a in self.world.agents],
            "locations": list(self.world.locations),
            "location_coords": dict(self.world.location_coords),
            "tick_events": [dict(e) for e in (tick_events or [])],
            "recent_events": [dict(e) for e in self._recent_events[-self._max_recent:]],
        }

    def broadcast(self, result: TickResult | None, cost: float = 0.0) -> None:
        """Push a new tick snapshot to every connected browser."""
        self._cost = cost
        if result:
            self._recent_events.extend(result.events)
            if len(self._recent_events) > self._max_recent:
                self._recent_events = self._recent_events[-self._max_recent:]
        payload = json.dumps(self._snapshot(result.events if result else []))
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.discard(q)

    # ── Routes ────────────────────────────────────────────────

    async def _index(self, request: web.Request) -> web.Response:
        path = _STATIC_DIR / "index.html"
        if not path.exists():
            return web.Response(status=500, text="static/index.html missing")
        return web.Response(text=path.read_text(), content_type="text/html")

    async def _snapshot_http(self, request: web.Request) -> web.Response:
        return web.json_response(self._snapshot([]))

    async def _events_stream(self, request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
        })
        await resp.prepare(request)

        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        self._subscribers.add(q)

        # Send initial state immediately so the page paints without waiting.
        initial = json.dumps(self._snapshot([]))
        await resp.write(f"data: {initial}\n\n".encode())

        try:
            while True:
                # Heartbeat every 15s so the connection stays alive.
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    await resp.write(f"data: {payload}\n\n".encode())
                except asyncio.TimeoutError:
                    await resp.write(b": keepalive\n\n")
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            self._subscribers.discard(q)
        return resp

    def _make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/snapshot", self._snapshot_http)
        app.router.add_get("/events", self._events_stream)
        return app

    # ── Lifecycle ─────────────────────────────────────────────

    async def start(self, port: int = 8765, host: str = "0.0.0.0") -> web.AppRunner:
        runner = web.AppRunner(self._make_app(), access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        print(f"[webui] live at http://localhost:{port}", file=sys.stderr)
        return runner
