"""Main entry point — run a nanolife simulation from the command line.

Parses CLI args, loads a scenario, wires up the engine with LLM providers,
and renders output via the Rich dashboard or plain text.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            if value and key.strip() not in os.environ:
                os.environ[key.strip()] = value.strip().strip("\"'")

from nanolife.common import TickResult
from nanolife.defaults.cognitive import LLMCognitive
from nanolife.engine import Engine
from nanolife.world import WorldState

NAMES = [
    "Ada", "Bjorn", "Cora", "Dmitri", "Elena", "Farid", "Gaia", "Hiro",
    "Iris", "Jin", "Kai", "Luna", "Milo", "Nadia", "Omar", "Petra",
    "Quinn", "Ravi", "Sable", "Tao", "Uma", "Vera", "Wyatt", "Xena",
    "Yuki", "Zara", "Arlo", "Bree", "Cyan", "Dex",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nanolife — artificial life simulation")
    p.add_argument("--agents", type=int, default=None, help="Number of starting agents (default: all for scenario, 10 otherwise)")
    p.add_argument("--ticks", type=int, default=50, help="Number of ticks to simulate")
    p.add_argument("--harshness", type=float, default=0.5, help="World harshness 0.0-1.0")
    p.add_argument("--tick-unit", type=str, default="4h", help="Tick unit (minute/hour/4h/day/week)")
    p.add_argument("--scenario", type=str, default="nanothrones", help="Scenario name to load")
    p.add_argument("--model", type=str, default=None, help="Model for cognition (default depends on provider)")
    p.add_argument("--report-model", type=str, default=None, help="Model for postmortem report (default depends on provider)")
    p.add_argument("--no-report", action="store_true", help="Skip postmortem report generation")
    p.add_argument("--open-router", action="store_true", help="Use OpenRouter with Gemini instead of Groq")
    return p.parse_args()


def _resolve_provider(args: argparse.Namespace) -> dict[str, str]:
    """Return api_key, base_url, model, and report_model based on CLI flags."""
    if args.open_router:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("[FATAL] OPENROUTER_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
            raise RuntimeError("OPENROUTER_API_KEY not set")
        default_model = "google/gemini-2.5-flash"
        return {
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "model": args.model or default_model,
            "report_model": args.report_model or default_model,
            "provider_label": "OpenRouter",
        }
    else:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            print("[FATAL] GROQ_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
            raise RuntimeError("GROQ_API_KEY not set")
        default_model = "openai/gpt-oss-120b"
        return {
            "api_key": api_key,
            "base_url": "https://api.groq.com/openai/v1",
            "model": args.model or default_model,
            "report_model": args.report_model or default_model,
            "provider_label": "Groq",
        }


async def main() -> None:
    args = parse_args()
    provider = _resolve_provider(args)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("logs/runs") / run_id

    scenario = None
    if args.scenario:
        from nanolife.scenario_loader import load_scenario
        scenario = load_scenario(args.scenario)

    harshness = scenario.harshness if scenario else args.harshness
    tick_unit = scenario.tick_unit if scenario else args.tick_unit
    scenario_name = scenario.name if scenario else "nanolife"

    world = WorldState.create(
        harshness=harshness,
        tick_unit=tick_unit,
        run_dir=str(run_dir),
        scenario_name=scenario_name,
        base_drain=scenario.base_drain if scenario else 1.0,
        base_gain=scenario.base_gain if scenario else 2.5,
        reputation_decay=scenario.reputation_decay if scenario else 0.02,
        starting_resources=scenario.starting_resources if scenario else 15.0,
    )
    if scenario and scenario.locations:
        world.locations = scenario.locations
    if scenario and scenario.location_coords:
        world.location_coords = scenario.location_coords

    cognitive = LLMCognitive(
        model=provider["model"],
        api_key=provider["api_key"],
        base_url=provider["base_url"],
    )

    from nanolife.defaults.spread import RandomSpread
    from nanolife.defaults.compression import LLMCompression
    spread = RandomSpread()
    compression = LLMCompression(
        model=provider["model"],
        api_key=provider["api_key"],
        base_url=provider["base_url"],
    )

    engine = Engine(world=world, cognitive=cognitive, spread=spread, compression=compression)

    if scenario and scenario.agents:
        num_agents = args.agents if args.agents is not None else len(scenario.agents)
        for agent_def in scenario.agents[:num_agents]:
            engine.spawn_agent(
                name=agent_def["name"],
                traits=agent_def.get("traits"),
                goal=agent_def.get("goal", "survive and thrive"),
                location=agent_def.get("location"),
            )
        for text in scenario.opening_events:
            world.event_log.append({
                "tick": 0,
                "type": "world",
                "content": text,
                "witnesses": [a.id for a in world.agents],
            })
        n = len(world.agents)
    else:
        num_agents = args.agents if args.agents is not None else 10
        n = min(num_agents, len(NAMES))
        for name in NAMES[:n]:
            engine.spawn_agent(name)

    world.event_log.append({
        "tick": 0,
        "type": "world",
        "content": f"SYSTEM_TICK_UNIT={tick_unit}",
        "witnesses": [],
    })

    model = provider["model"]
    est_calls = n * args.ticks * 2
    est_tokens = est_calls * 400
    model_lower = model.lower()
    if any(k in model_lower for k in ("gpt-oss-120b", "4o-mini")):
        est_rate = (0.15 + 0.60) / 2
    elif "gemini" in model_lower:
        est_rate = (0.30 + 2.50) / 2
    elif "haiku" in model_lower:
        est_rate = (1.0 + 5.0) / 2
    elif "sonnet" in model_lower:
        est_rate = (3.0 + 15.0) / 2
    elif "opus" in model_lower:
        est_rate = (5.0 + 25.0) / 2
    elif "4o" in model_lower:
        est_rate = (2.50 + 10.0) / 2
    else:
        est_rate = (0.50 + 1.50) / 2
    est_cost = est_tokens * est_rate / 1_000_000
    if "--scenario" not in sys.argv and "--ticks" not in sys.argv and "--agents" not in sys.argv:
        print(f"\033[92mSimulation defaulted to scenario {args.scenario}, ticks {args.ticks}, and {n} agents from the scenario.\033[0m")
    print(f"Calculated {args.ticks} turns for {n} agents · {tick_unit} ticks · {provider['provider_label']} · estimated cost ${est_cost:.2f}")

    use_dashboard = os.isatty(1) and not os.environ.get("NANOLIFE_PLAIN")
    scenario_label = args.scenario or "nanolife"
    renderer = None
    t0 = time.time()
    elapsed = 0.0

    try:
        if use_dashboard:
            from nanolife.terminal import TerminalRenderer
            renderer = TerminalRenderer(world, scenario_label)
            renderer.start()

            def on_tick_dashboard(result: TickResult) -> None:
                cost = cognitive.total_cost if hasattr(cognitive, "total_cost") else 0.0
                renderer.update(result, cost)

            t0 = time.time()
            results = await engine.run(args.ticks, on_tick=on_tick_dashboard)
            elapsed = time.time() - t0
        else:
            print(f"nanolife · {scenario_name} | {n} agents | {args.ticks} ticks | harshness {harshness} | {tick_unit} ticks")
            print(f"Logs → {run_dir}")
            print("─" * 60)

            def print_tick(result: TickResult) -> None:
                events_summary = []
                for e in result.events:
                    t = e.get("type", "?")
                    if t == "thought":
                        continue
                    agent_name = e.get("agent", "?")
                    content = e.get("content", "")
                    if t == "death":
                        events_summary.append(f"  ☠ {content} (cause: {e.get('cause', '?')})")
                    elif t == "birth":
                        events_summary.append(f"  + {content}")
                    elif t == "friendship":
                        events_summary.append(f"  ~ {content}")
                    elif t == "reputation":
                        events_summary.append(f"  ★ {content} ({e.get('delta', 0):+.2f})")
                    elif t == "action":
                        events_summary.append(f"  > [{agent_name}] {content}")
                    elif t == "improvement":
                        events_summary.append(f"  ◆ [{agent_name}] {content}")

                print(f"\n── tick {result.tick} | pop {result.population} | born {result.births} | died {result.deaths} ──")
                for line in events_summary[:12]:
                    print(line)
                if len(events_summary) > 12:
                    print(f"  ... and {len(events_summary) - 12} more events")

            t0 = time.time()
            results = await engine.run(args.ticks, on_tick=print_tick)
            elapsed = time.time() - t0

    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print("\n\nSimulation interrupted (Ctrl+C).")

    except Exception:
        elapsed = time.time() - t0
        raise

    finally:
        if renderer:
            try:
                renderer.stop()
            except Exception:
                pass

        alive = world.population
        dead = world.total_deaths
        born = world.total_births
        print("\n" + "═" * 60)
        print(f"Simulation {'interrupted' if not world.alive_agents else 'complete'} in {elapsed:.1f}s")
        print(f"Final: {alive} alive, {dead} dead, {born} born")
        print(f"LLM cost: ${cognitive.total_cost:.4f} ({cognitive.total_tokens} tokens)")
        print(f"Logs: {run_dir}")

        if not args.no_report:
            from nanolife.postmortem import run_postmortem
            try:
                await run_postmortem(
                    run_dir=run_dir,
                    report_model=provider["report_model"],
                    api_key=provider["api_key"],
                    base_url=provider["base_url"],
                    open_browser=True,
                )
            except Exception as e:
                print(f"  [postmortem] Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
