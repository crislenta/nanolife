"""Benchmark: run the same scenario on two providers and compare behavior side-by-side.

Usage:
    python3 -m scripts.benchmark
    python3 -m scripts.benchmark --scenario=nanothrones --ticks=80 --agents=8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            if value and key.strip() not in os.environ:
                os.environ[key.strip()] = value.strip().strip("\"'")

from nanosim.charts import generate_all_charts
from nanosim.common import TickResult
from nanosim.defaults.cognitive import LLMCognitive
from nanosim.defaults.compression import LLMCompression
from nanosim.defaults.spread import RandomSpread
from nanosim.engine import Engine
from nanosim.postmortem import analyze, _load_events
from nanosim.world import WorldState


PROVIDERS = {
    "groq": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "label": "Groq · GPT-OSS-120B",
    },
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemini-2.5-flash",
        "label": "OpenRouter · Gemini 2.5 Flash",
    },
    "vertex": {
        "env_key": "VERTEX_ACCESS_TOKEN",
        "base_url": "",
        "model": "google/gemini-2.5-flash",
        "label": "Vertex AI · Gemini 2.5 Flash",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nanosim benchmark — compare two LLM providers")
    p.add_argument("--scenario", type=str, default="nanothrones", help="Scenario to run")
    p.add_argument("--ticks", type=int, default=80, help="Ticks per run (default 80 for speed)")
    p.add_argument("--agents", type=int, default=None, help="Number of agents (default: all from scenario)")
    p.add_argument("--tick-unit", type=str, default=None, help="Override tick unit")
    p.add_argument(
        "--providers",
        type=str,
        default="groq,openrouter",
        help="Comma-separated providers from: groq,openrouter,vertex",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible benchmark runs")
    return p.parse_args()


def _resolve_vertex_access_token() -> str:
    token = os.environ.get("VERTEX_ACCESS_TOKEN", "").strip() or os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN", "").strip()
    if token:
        return token
    try:
        return subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


async def run_single(
    provider_key: str,
    scenario_name: str,
    ticks: int,
    num_agents: int | None,
    tick_unit_override: str | None,
    bench_dir: Path,
    seed: int,
) -> dict:
    """Run one simulation and return a summary dict."""
    prov = PROVIDERS[provider_key]
    if provider_key == "vertex":
        project = os.environ.get("VERTEX_PROJECT_ID", "").strip() or os.environ.get("GCP_PROJECT_ID", "").strip()
        location = os.environ.get("VERTEX_LOCATION", "").strip() or os.environ.get("GCP_LOCATION", "").strip() or "us-central1"
        api_key = _resolve_vertex_access_token()
        if not project:
            print("  [SKIP] Vertex AI · Gemini 2.5 Flash — VERTEX_PROJECT_ID not set")
            return {"skipped": True, "provider": prov["label"]}
        if not api_key:
            print("  [SKIP] Vertex AI · Gemini 2.5 Flash — no access token (VERTEX_ACCESS_TOKEN or gcloud auth)")
            return {"skipped": True, "provider": prov["label"]}
        prov = dict(prov)
        prov["base_url"] = os.environ.get("VERTEX_OPENAI_BASE_URL", "").strip() or (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi"
        )
    else:
        api_key = os.environ.get(prov["env_key"], "")
        if not api_key:
            print(f"  [SKIP] {prov['label']} — {prov['env_key']} not set")
            return {"skipped": True, "provider": prov["label"]}

    from nanosim.scenario_loader import load_scenario
    import random
    random.seed(seed)
    scenario = load_scenario(scenario_name)

    run_dir = bench_dir / provider_key
    run_dir.mkdir(parents=True, exist_ok=True)

    harshness = scenario.harshness
    tick_unit = tick_unit_override or scenario.tick_unit

    world = WorldState.create(
        harshness=harshness,
        tick_unit=tick_unit,
        run_dir=str(run_dir),
        scenario_name=scenario.name,
        base_drain=scenario.base_drain,
        base_gain=scenario.base_gain,
        reputation_decay=scenario.reputation_decay,
        starting_resources=scenario.starting_resources,
    )
    if scenario.locations:
        world.locations = scenario.locations
    if scenario.location_coords:
        world.location_coords = scenario.location_coords

    cognitive = LLMCognitive(model=prov["model"], api_key=api_key, base_url=prov["base_url"])
    spread = RandomSpread()
    compression = LLMCompression(model=prov["model"], api_key=api_key, base_url=prov["base_url"])
    engine = Engine(world=world, cognitive=cognitive, spread=spread, compression=compression)

    n = num_agents if num_agents is not None else len(scenario.agents)
    for agent_def in scenario.agents[:n]:
        engine.spawn_agent(
            name=agent_def["name"],
            traits=agent_def.get("traits"),
            goal=agent_def.get("goal", "survive and thrive"),
            location=agent_def.get("location"),
        )
    for text in scenario.opening_events:
        world.event_log.append({
            "tick": 0, "type": "world", "content": text,
            "witnesses": [a.id for a in world.agents],
        })
    world.event_log.append({
        "tick": 0, "type": "world",
        "content": f"SYSTEM_TICK_UNIT={tick_unit}",
        "witnesses": [],
    })

    print(f"\n{'━' * 60}")
    print(f"  Running: {prov['label']}")
    print(f"  Model:   {prov['model']}")
    print(f"  Agents:  {n} | Ticks: {ticks}")
    print(f"{'━' * 60}")

    tick_times: list[float] = []

    def on_tick(result: TickResult) -> None:
        d = result.deaths
        b = result.births
        extra = ""
        if d: extra += f" ☠{d}"
        if b: extra += f" +{b}"
        print(f"  tick {result.tick:4d}  pop {result.population:2d}{extra}")

    t0 = time.time()
    try:
        await engine.run(ticks, on_tick=on_tick)
    except KeyboardInterrupt:
        print("  (interrupted)")
    elapsed = time.time() - t0

    extinct = world.population == 0
    if extinct:
        print(f"  ⚠ Population extinct at tick {world.clock.tick} (of {ticks})")


    emergence = analyze(run_dir)
    events = _load_events(run_dir)

    deaths = [e for e in events if e.get("type") == "death"]
    births = [e for e in events if e.get("type") == "birth"]
    actions = [e for e in events if e.get("type") == "action"]
    friendships = [e for e in events if e.get("type") == "friendship"]
    rep_events = [e for e in events if e.get("type") == "reputation"]
    rumors = [e for e in events if e.get("type") == "rumor"]
    improvements = [e for e in events if e.get("type") == "improvement"]
    transfers = [e for e in events if e.get("type") == "transfer"]

    work_actions = sum(1 for e in actions if e.get("mode") == "productive")
    social_actions = sum(1 for e in actions if e.get("mode") == "social")
    rest_actions = sum(1 for e in actions if e.get("mode") == "rest")

    praises = [e for e in rep_events if e.get("delta", 0) > 0]
    criticisms = [e for e in rep_events if e.get("delta", 0) < 0]

    avg_action_len = sum(len(e.get("content", "")) for e in actions) / max(len(actions), 1)
    avg_thought_len = sum(len(e.get("content", "")) for e in events if e.get("type") == "thought") / max(1, sum(1 for e in events if e.get("type") == "thought"))

    death_ticks = [e.get("tick", 0) for e in deaths]
    first_death = min(death_ticks) if death_ticks else None
    last_death = max(death_ticks) if death_ticks else None

    unique_locations = set()
    for e in events:
        if e.get("type") == "move":
            content = e.get("content", "")
            if " to " in content:
                unique_locations.add(content.split(" to ")[-1])

    summary = {
        "provider": prov["label"],
        "model": prov["model"],
        "run_dir": str(run_dir),
        "elapsed_s": round(elapsed, 1),
        "ticks_completed": world.clock.tick,
        "extinct": extinct,
        "final_population": world.population,
        "total_births": world.total_births,
        "total_deaths": world.total_deaths,
        "total_events": len(events),
        "llm_calls": cognitive.llm_calls,
        "total_tokens": cognitive.total_tokens,
        "total_cost": round(cognitive.total_cost, 4),
        "avg_action_length": round(avg_action_len, 1),
        "avg_thought_length": round(avg_thought_len, 1),
        "mode_split": {
            "productive": work_actions,
            "social": social_actions,
            "rest": rest_actions,
        },
        "friendships": len(friendships),
        "praises": len(praises),
        "criticisms": len(criticisms),
        "rumors": len(rumors),
        "transfers": len(transfers),
        "locations_discovered": len(unique_locations),
        "first_death_tick": first_death,
        "last_death_tick": last_death,
        "emergence_index": emergence["emergence_index"],
        "emergence_max": emergence.get("max_emergence_index", 11),
        "phenomena": emergence["phenomena_detected"],
    }
    return summary


def print_comparison(results: list[dict], bench_dir: Path) -> None:
    print("\n\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + "  BENCHMARK COMPARISON".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    valid = [r for r in results if not r.get("skipped")]
    if len(valid) < 2:
        print("\n  Not enough providers ran for side-by-side comparison. Run at least two providers.")
        return

    a, b = valid[0], valid[1]

    def row(label: str, key: str, fmt: str = "{}", higher_better: bool | None = None) -> None:
        va, vb = a.get(key, "—"), b.get(key, "—")
        sa = fmt.format(va) if va != "—" else "—"
        sb = fmt.format(vb) if vb != "—" else "—"

        indicator = "   "
        if higher_better is not None and isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if va > vb:
                indicator = " ◀ " if higher_better else " ▶ "
            elif vb > va:
                indicator = " ▶ " if higher_better else " ◀ "
            else:
                indicator = " = "

        print(f"  {label:<28s} {sa:>18s}{indicator}{sb:>18s}")

    header_a = a["provider"][:20]
    header_b = b["provider"][:20]
    print(f"\n  {'METRIC':<28s} {header_a:>18s}     {header_b:>18s}")
    print(f"  {'─' * 28} {'─' * 18}     {'─' * 18}")

    print(f"  {'PERFORMANCE':<28s}")
    row("  Wall time", "elapsed_s", "{:.1f}s", higher_better=False)
    row("  LLM calls", "llm_calls", "{}", higher_better=False)
    row("  Total tokens", "total_tokens", "{:,}", higher_better=False)
    row("  Total cost", "total_cost", "${:.4f}", higher_better=False)

    print(f"\n  {'SIMULATION OUTCOME':<28s}")
    row("  Ticks completed", "ticks_completed", "{}")
    row("  Final population", "final_population", "{}", higher_better=True)
    row("  Births", "total_births", "{}", higher_better=True)
    row("  Deaths", "total_deaths", "{}", higher_better=False)
    row("  First death tick", "first_death_tick", "{}", higher_better=True)

    print(f"\n  {'BEHAVIOR QUALITY':<28s}")
    row("  Avg action length", "avg_action_length", "{:.0f} chars")
    row("  Avg thought length", "avg_thought_length", "{:.0f} chars")
    row("  Friendships formed", "friendships", "{}", higher_better=True)
    row("  Praises given", "praises", "{}", higher_better=True)
    row("  Criticisms given", "criticisms", "{}")
    row("  Rumors spread", "rumors", "{}")
    row("  Transfers", "transfers", "{}")
    row("  Locations discovered", "locations_discovered", "{}")

    print(f"\n  {'MODE DISTRIBUTION':<28s}")
    for mode in ("productive", "social", "rest"):
        va = a.get("mode_split", {}).get(mode, 0)
        vb = b.get("mode_split", {}).get(mode, 0)
        ta = sum(a.get("mode_split", {}).values()) or 1
        tb = sum(b.get("mode_split", {}).values()) or 1
        sa = f"{va} ({va/ta:.0%})"
        sb = f"{vb} ({vb/tb:.0%})"
        print(f"    {mode:<26s} {sa:>18s}     {sb:>18s}")

    print(f"\n  {'EMERGENCE':<28s}")
    row("  Emergence index", "emergence_index", "{}/11", higher_better=True)
    pa = ", ".join(a.get("phenomena", [])) or "none"
    pb = ", ".join(b.get("phenomena", [])) or "none"
    print(f"  {'  Phenomena detected':<28s}")
    print(f"    A: {pa}")
    print(f"    B: {pb}")

    only_a = set(a.get("phenomena", [])) - set(b.get("phenomena", []))
    only_b = set(b.get("phenomena", [])) - set(a.get("phenomena", []))
    shared = set(a.get("phenomena", [])) & set(b.get("phenomena", []))
    if shared:
        print(f"    Shared: {', '.join(sorted(shared))}")
    if only_a:
        print(f"    Only {a['model']}: {', '.join(sorted(only_a))}")
    if only_b:
        print(f"    Only {b['model']}: {', '.join(sorted(only_b))}")

    # Verdict
    print(f"\n  {'─' * 72}")
    scores = {"a": 0, "b": 0}
    if a["final_population"] > b["final_population"]: scores["a"] += 1
    elif b["final_population"] > a["final_population"]: scores["b"] += 1
    if a["emergence_index"] > b["emergence_index"]: scores["a"] += 1
    elif b["emergence_index"] > a["emergence_index"]: scores["b"] += 1
    if a["friendships"] + a["praises"] > b["friendships"] + b["praises"]: scores["a"] += 1
    elif b["friendships"] + b["praises"] > a["friendships"] + a["praises"]: scores["b"] += 1
    if a["total_cost"] < b["total_cost"]: scores["a"] += 1
    elif b["total_cost"] < a["total_cost"]: scores["b"] += 1
    if a["avg_action_length"] > b["avg_action_length"]: scores["a"] += 1
    elif b["avg_action_length"] > a["avg_action_length"]: scores["b"] += 1

    print(f"\n  VERDICT (5-point heuristic: survival, emergence, sociality, cost, verbosity)")
    print(f"    {a['model']:>30s}:  {scores['a']}/5")
    print(f"    {b['model']:>30s}:  {scores['b']}/5")

    if scores["a"] > scores["b"]:
        print(f"\n    Winner: {a['provider']}")
    elif scores["b"] > scores["a"]:
        print(f"\n    Winner: {b['provider']}")
    else:
        print(f"\n    Result: TIE")

    print()

    comparison_path = bench_dir / "comparison.json"
    comparison_path.write_text(json.dumps({"runs": results, "scores": scores}, indent=2))
    print(f"  Full comparison saved: {comparison_path}")

    print(f"\n{'━' * 60}")
    print("  GENERATING CHARTS")
    print(f"{'━' * 60}")
    for r in valid:
        run_dir = Path(r["run_dir"])
        print(f"\n  [{r['provider']}]")
        generate_all_charts(run_dir)
    print()


async def main() -> None:
    args = parse_args()

    bench_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = Path("logs/benchmarks") / bench_id
    bench_dir.mkdir(parents=True, exist_ok=True)

    print("╔" + "═" * 58 + "╗")
    print("║" + "  NANOSIM BENCHMARK".center(58) + "║")
    print("║" + f"  {args.scenario} · {args.ticks} ticks".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    selected = [p.strip() for p in args.providers.split(",") if p.strip()]
    invalid = [p for p in selected if p not in PROVIDERS]
    if invalid:
        raise SystemExit(f"Unknown providers: {', '.join(invalid)}. Valid: {', '.join(PROVIDERS)}")

    results = []
    for provider_key in selected:
        result = await run_single(
            provider_key=provider_key,
            scenario_name=args.scenario,
            ticks=args.ticks,
            num_agents=args.agents,
            tick_unit_override=args.tick_unit,
            bench_dir=bench_dir,
            seed=args.seed,
        )
        results.append(result)

    print_comparison(results, bench_dir)


if __name__ == "__main__":
    asyncio.run(main())
