"""Create social-ready artifacts from a finished simulation run.

Outputs:
  - x_metric_card.png : single summary image
  - x_replay.gif      : short highlight replay
  - x_thread.md       : ready-to-post thread draft
  - x_summary.json    : machine-readable summary
"""
from __future__ import annotations

import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .postmortem import analyze


def _read_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "world.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _identity_maps(events: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    id_to_name: dict[str, str] = {}
    name_to_id: dict[str, str] = {}
    for e in events:
        if e.get("type") == "birth":
            aid = str(e.get("agent", ""))
            content = str(e.get("content", ""))
            name = content.split(" enters ")[0] if " enters " in content else aid[:8]
            id_to_name[aid] = name
            name_to_id[name] = aid
    return id_to_name, name_to_id


def _population_series(events: list[dict[str, Any]]) -> tuple[list[int], list[int], int]:
    max_tick = max((int(e.get("tick", 0)) for e in events), default=0)
    births = defaultdict(int)
    deaths = defaultdict(int)
    for e in events:
        t = int(e.get("tick", 0))
        if e.get("type") == "birth":
            births[t] += 1
        elif e.get("type") == "death":
            deaths[t] += 1

    initial_agents = births[0]
    pop = initial_agents
    xs: list[int] = []
    ys: list[int] = []
    for tick in range(0, max_tick + 1):
        if tick > 0:
            pop += births[tick]
            pop -= deaths[tick]
        xs.append(tick)
        ys.append(pop)
    return xs, ys, initial_agents


def _candidate_from_event(e: dict[str, Any], id_to_name: dict[str, str]) -> dict[str, Any] | None:
    etype = str(e.get("type", ""))
    tick = int(e.get("tick", 0))
    content = str(e.get("content", "")).strip()
    aid = str(e.get("agent", ""))
    who = id_to_name.get(aid, aid[:8] if aid else "world")

    def mk(kind: str, score: int, text: str) -> dict[str, Any]:
        return {"tick": tick, "kind": kind, "score": score, "text": text}

    if etype == "death":
        return mk("death", 100, content or f"{who} died")
    if etype == "trade":
        return mk("trade", 82, content)
    if etype == "transfer" and abs(float(e.get("amount", 0))) >= 3.0:
        return mk("transfer", 74, content)
    if etype == "reputation" and float(e.get("delta", 0)) <= -0.2:
        return mk("betrayal", 80, content)
    if etype == "friendship":
        return mk("friendship", 60, content)
    if etype == "birth" and tick > 0:
        return mk("birth", 56, content)
    if etype == "improvement":
        return mk("insight", 34, f"{who}: {content}")
    return None


def extract_highlights(events: list[dict[str, Any]], max_moments: int = 12) -> list[dict[str, Any]]:
    id_to_name, _ = _identity_maps(events)
    cands: list[dict[str, Any]] = []
    for e in events:
        c = _candidate_from_event(e, id_to_name)
        if c:
            cands.append(c)

    if not cands:
        return []

    # Keep strongest moments while enforcing diversity in neighboring ticks.
    cands.sort(key=lambda x: (-x["score"], x["tick"]))
    picked: list[dict[str, Any]] = []
    used_ticks: set[int] = set()
    for c in cands:
        t = int(c["tick"])
        if t in used_ticks:
            continue
        picked.append(c)
        used_ticks.add(t)
        if len(picked) >= max_moments:
            break

    picked.sort(key=lambda x: x["tick"])
    return picked


def summarize_run(run_dir: Path, max_moments: int = 12) -> dict[str, Any]:
    events = _read_events(run_dir)
    if not events:
        raise FileNotFoundError(f"No world.jsonl found in {run_dir}")

    event_counts = Counter(str(e.get("type", "unknown")) for e in events)
    x_ticks, y_pop, initial_agents = _population_series(events)
    id_to_name, _ = _identity_maps(events)
    births_total = event_counts.get("birth", 0)
    deaths_total = event_counts.get("death", 0)
    births_after_start = sum(1 for e in events if e.get("type") == "birth" and int(e.get("tick", 0)) > 0)
    final_pop = y_pop[-1] if y_pop else (births_total - deaths_total)
    max_tick = max(x_ticks) if x_ticks else 0

    # Scenario hint from early world events.
    scenario_hint = "simulation"
    for e in events:
        if e.get("type") == "world":
            c = str(e.get("content", "")).strip()
            if c and not c.startswith("SYSTEM_TICK_UNIT="):
                scenario_hint = c[:80]
                break

    emergence = analyze(run_dir)
    highlights = extract_highlights(events, max_moments=max_moments)

    return {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "scenario_hint": scenario_hint,
        "ticks": max_tick,
        "initial_agents": initial_agents,
        "final_population": final_pop,
        "births": births_after_start,
        "deaths": deaths_total,
        "event_counts": dict(event_counts),
        "emergence_index": emergence.get("emergence_index", 0),
        "emergence_max": emergence.get("max_emergence_index", 11),
        "phenomena": emergence.get("phenomena_detected", []),
        "timeline": {"ticks": x_ticks, "population": y_pop},
        "highlights": highlights,
        "id_to_name": id_to_name,
    }


def _draw_population_sparkline(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, ticks: list[int], pop: list[int]) -> None:
    draw.rectangle([x, y, x + w, y + h], outline="#334155", width=2)
    if not ticks or not pop:
        return
    t_min, t_max = min(ticks), max(ticks)
    p_min, p_max = min(pop), max(pop)
    if t_max == t_min:
        t_max += 1
    if p_max == p_min:
        p_max += 1

    pts: list[tuple[float, float]] = []
    for t, p in zip(ticks, pop):
        px = x + ((t - t_min) / (t_max - t_min)) * w
        py = y + h - ((p - p_min) / (p_max - p_min)) * h
        pts.append((px, py))
    if len(pts) >= 2:
        draw.line(pts, fill="#22d3ee", width=4)
    last = pts[-1]
    draw.ellipse([last[0] - 4, last[1] - 4, last[0] + 4, last[1] + 4], fill="#f8fafc")


def render_metric_card(summary: dict[str, Any], out_path: Path, title: str | None = None) -> Path:
    img = Image.new("RGB", (1080, 1350), "#0b1020")
    draw = ImageDraw.Draw(img)
    h1 = ImageFont.load_default()
    h2 = ImageFont.load_default()
    body = ImageFont.load_default()

    draw.text((48, 40), title or "nanosim run summary", fill="#e2e8f0", font=h1)
    draw.text((48, 74), f"run: {summary['run_id']}", fill="#94a3b8", font=body)
    draw.text((48, 98), f"scenario: {summary['scenario_hint']}", fill="#94a3b8", font=body)

    stats = [
        ("ticks", summary["ticks"]),
        ("start agents", summary["initial_agents"]),
        ("final alive", summary["final_population"]),
        ("births", summary["births"]),
        ("deaths", summary["deaths"]),
        ("emergence", f"{summary['emergence_index']}/{summary['emergence_max']}"),
    ]

    y0 = 170
    for i, (k, v) in enumerate(stats):
        col = i % 2
        row = i // 2
        x = 48 + col * 500
        y = y0 + row * 90
        draw.rounded_rectangle([x, y, x + 460, y + 74], radius=12, outline="#334155", width=2, fill="#111827")
        draw.text((x + 18, y + 12), str(k), fill="#94a3b8", font=body)
        draw.text((x + 18, y + 38), str(v), fill="#f8fafc", font=h2)

    draw.text((48, 470), "population over time", fill="#cbd5e1", font=body)
    _draw_population_sparkline(
        draw,
        x=48,
        y=495,
        w=984,
        h=240,
        ticks=summary["timeline"]["ticks"],
        pop=summary["timeline"]["population"],
    )

    draw.text((48, 770), "detected phenomena", fill="#cbd5e1", font=body)
    ph = summary.get("phenomena", [])
    phen_line = ", ".join(ph) if ph else "none"
    for i, line in enumerate(textwrap.wrap(phen_line, width=95)[:3]):
        draw.text((48, 796 + i * 22), line, fill="#e2e8f0", font=body)

    draw.text((48, 890), "top moments", fill="#cbd5e1", font=body)
    for i, m in enumerate(summary.get("highlights", [])[:6]):
        line = f"t{m['tick']:>3} [{m['kind']}] {m['text']}"
        wrapped = textwrap.wrap(line, width=95)
        if not wrapped:
            continue
        draw.text((48, 916 + i * 56), wrapped[0], fill="#f8fafc", font=body)
        if len(wrapped) > 1:
            draw.text((72, 936 + i * 56), wrapped[1], fill="#94a3b8", font=body)

    draw.text((48, 1298), "generated by scripts/x_artifacts.py", fill="#64748b", font=body)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path


def render_replay_gif(summary: dict[str, Any], out_path: Path, title: str | None = None) -> Path:
    frames: list[Image.Image] = []
    draw_font = ImageFont.load_default()
    ticks = summary["timeline"]["ticks"]
    pop = summary["timeline"]["population"]
    pop_at_tick = {t: p for t, p in zip(ticks, pop)}
    highlights = summary.get("highlights", [])
    if not highlights:
        highlights = [{"tick": summary["ticks"], "kind": "summary", "text": "No dramatic moments captured in this run.", "score": 1}]

    for i, moment in enumerate(highlights):
        img = Image.new("RGB", (1280, 720), "#020617")
        d = ImageDraw.Draw(img)
        d.text((40, 30), title or "nanosim replay highlights", fill="#e2e8f0", font=draw_font)
        d.text((40, 56), f"run {summary['run_id']} · moment {i+1}/{len(highlights)}", fill="#94a3b8", font=draw_font)
        d.text((40, 88), f"tick {moment['tick']} · pop {pop_at_tick.get(moment['tick'], summary['final_population'])}", fill="#67e8f9", font=draw_font)
        d.text((40, 118), f"type: {moment['kind']}", fill="#fda4af", font=draw_font)

        box = [40, 160, 1240, 580]
        d.rounded_rectangle(box, radius=16, fill="#0f172a", outline="#334155", width=2)
        lines = textwrap.wrap(moment["text"], width=88)
        y = 196
        for line in lines[:12]:
            d.text((72, y), line, fill="#f8fafc", font=draw_font)
            y += 26

        # Progress bar.
        d.rectangle([40, 640, 1240, 662], fill="#1e293b")
        done = int((i + 1) / len(highlights) * 1200)
        d.rectangle([40, 640, 40 + done, 662], fill="#22d3ee")
        d.text((40, 670), "highlights replay", fill="#64748b", font=draw_font)
        frames.append(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=1200,
        loop=0,
        optimize=False,
    )
    return out_path


def write_thread_template(summary: dict[str, Any], out_path: Path) -> Path:
    phenomena = ", ".join(summary.get("phenomena", [])) or "none"
    highlights = summary.get("highlights", [])[:5]
    lines = [
        f"# nanosim run {summary['run_id']}",
        "",
        "1/ Ran a new nanosim scenario and got non-trivial social dynamics.",
        f"- ticks: {summary['ticks']}",
        f"- pop: {summary['initial_agents']} -> {summary['final_population']}",
        f"- births/deaths: +{summary['births']} / -{summary['deaths']}",
        f"- emergence index: {summary['emergence_index']}/{summary['emergence_max']}",
        f"- phenomena: {phenomena}",
        "",
        "2/ Key moments:",
    ]
    for m in highlights:
        lines.append(f"- t{m['tick']}: [{m['kind']}] {m['text']}")
    lines.extend(
        [
            "",
            "3/ Artifacts:",
            "- x_metric_card.png",
            "- x_replay.gif",
            "",
            "4/ Reproduce command:",
            "python3 -m scripts.simulate --scenario <name> --ticks <n> --seed <seed> --no-report",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def generate_x_artifacts(
    run_dir: Path,
    out_dir: Path | None = None,
    max_moments: int = 12,
    title: str | None = None,
    include_gif: bool = True,
    include_card: bool = True,
) -> dict[str, str]:
    run_dir = Path(run_dir)
    out = out_dir or run_dir
    out.mkdir(parents=True, exist_ok=True)

    summary = summarize_run(run_dir, max_moments=max_moments)
    summary_path = out / "x_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    outputs: dict[str, str] = {"summary_json": str(summary_path)}
    if include_card:
        card = render_metric_card(summary, out / "x_metric_card.png", title=title)
        outputs["metric_card"] = str(card)
    if include_gif:
        gif = render_replay_gif(summary, out / "x_replay.gif", title=title)
        outputs["replay_gif"] = str(gif)
    thread = write_thread_template(summary, out / "x_thread.md")
    outputs["thread_template"] = str(thread)
    return outputs
