"""Generate post-simulation charts from world.jsonl event logs.

Four charts:
  1. Kaplan-Meier survival curve
  2. Agent reputation heatmap (sociometric matrix)
  3. Drama timeline ("Spice-o-Meter")
  4. Character alignment scatter ("Who Are You Really?")
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Shared helpers ──────────────────────────────────────────────

_PALETTE = {
    "bg":        "#ffffff",
    "surface":   "#ffffff",
    "grid":      "#d0d0d0",
    "text":      "#1a1a1a",
    "muted":     "#555555",
    "accent":    "#2c5f8a",
    "green":     "#2a7f3f",
    "red":       "#b82020",
    "orange":    "#c45a00",
    "purple":    "#5b2d8e",
    "teal":      "#1a7a5c",
    "gold":      "#c47f00",
    "pink":      "#a82050",
}

_EVENT_COLORS = {
    "death":      _PALETTE["red"],
    "betrayal":   _PALETTE["orange"],
    "conflict":   _PALETTE["pink"],
    "friendship": _PALETTE["purple"],
    "transfer":   _PALETTE["teal"],
}


def _load_events(run_dir: Path) -> list[dict]:
    world_path = run_dir / "world.jsonl"
    if not world_path.exists():
        return []
    with open(world_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _id_to_name(events: list[dict]) -> dict[str, str]:
    _ROMAN = ["", " II", " III", " IV", " V", " VI", " VII", " VIII"]
    raw: dict[str, str] = {}
    for e in events:
        if e.get("type") == "birth":
            content = e.get("content", "")
            name = content.split(" enters ")[0] if " enters " in content else e.get("agent", "?")[:8]
            raw[e.get("agent", "")] = name

    seen: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for aid, name in raw.items():
        count = seen.get(name, 0)
        seen[name] = count + 1
        suffix = _ROMAN[count] if count < len(_ROMAN) else f" ({count + 1})"
        mapping[aid] = name + suffix
    return mapping


_FONT_FAMILY = "serif"


def _apply_theme(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_facecolor(_PALETTE["surface"])
    ax.figure.set_facecolor(_PALETTE["bg"])
    ax.tick_params(colors=_PALETTE["muted"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#aaaaaa")
    pad = 20 if subtitle else 14
    ax.set_title(title, color=_PALETTE["text"], fontsize=13,
                 fontweight="bold", pad=pad, family=_FONT_FAMILY)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center",
                fontsize=8.5, color=_PALETTE["muted"], style="italic", family=_FONT_FAMILY)


# ── 1. Kaplan-Meier Survival Curve ─────────────────────────────

def chart_survival(run_dir: Path, out_path: Path | None = None) -> Path:
    events = _load_events(run_dir)
    names = _id_to_name(events)
    max_tick = max((e.get("tick", 0) for e in events), default=0)

    births: dict[str, int] = {}
    deaths: dict[str, int] = {}
    for e in events:
        aid = e.get("agent", "")
        if e.get("type") == "birth":
            births[aid] = e.get("tick", 0)
        elif e.get("type") == "death":
            deaths[aid] = e.get("tick", 0)

    total_agents = len(births)
    if total_agents == 0:
        raise ValueError("No agents found in event log")

    ticks = list(range(0, max_tick + 1))
    alive = []
    for t in ticks:
        n = sum(1 for aid, bt in births.items() if bt <= t and deaths.get(aid, max_tick + 1) > t)
        alive.append(n / total_agents)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.step(ticks, alive, where="post", color=_PALETTE["accent"], linewidth=2.0, zorder=3)
    ax.fill_between(ticks, alive, step="post", alpha=0.10, color=_PALETTE["accent"], zorder=2)

    for aid, dtick in deaths.items():
        name = names.get(aid, aid[:6])
        prop = sum(1 for a, bt in births.items() if bt <= dtick and deaths.get(a, max_tick + 1) > dtick) / total_agents
        ax.plot(dtick, prop, "o", color=_PALETTE["red"], markersize=6, zorder=5)
        ax.annotate(name, (dtick, prop), textcoords="offset points", xytext=(6, 6),
                    fontsize=7.5, color=_PALETTE["red"], fontstyle="italic", zorder=5,
                    family=_FONT_FAMILY)

    ax.set_xlabel("Tick", color=_PALETTE["muted"], fontsize=10, family=_FONT_FAMILY)
    ax.set_ylabel("Survival Proportion", color=_PALETTE["muted"], fontsize=10, family=_FONT_FAMILY)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, max_tick)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.4, alpha=0.5)
    _apply_theme(ax, "Kaplan\u2013Meier Survival Curve", f"{total_agents} agents \u00b7 {max_tick} ticks")

    out = out_path or (run_dir / "chart_survival.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ── 2. Reputation Chart (diverging bar) ───────────────────────

def chart_reputation_heatmap(run_dir: Path, out_path: Path | None = None) -> Path:
    events = _load_events(run_dir)
    names = _id_to_name(events)

    agent_ids = list(names.keys())
    if not agent_ids:
        raise ValueError("No agents found")

    praise_received: dict[str, float] = defaultdict(float)
    criticism_received: dict[str, float] = defaultdict(float)

    for e in events:
        if e.get("type") == "reputation":
            tgt = e.get("agent", "")
            delta = e.get("delta", 0)
            if tgt in names:
                if delta > 0:
                    praise_received[tgt] += delta
                elif delta < 0:
                    criticism_received[tgt] += abs(delta)

    active_ids = [aid for aid in agent_ids
                  if praise_received.get(aid, 0) > 0 or criticism_received.get(aid, 0) > 0]
    if not active_ids:
        raise ValueError("No reputation events found")

    net = {aid: praise_received.get(aid, 0) - criticism_received.get(aid, 0) for aid in active_ids}
    active_ids.sort(key=lambda a: net[a])

    agent_labels = [names[aid] for aid in active_ids]
    praises = [praise_received.get(aid, 0) for aid in active_ids]
    criticisms = [-criticism_received.get(aid, 0) for aid in active_ids]
    n = len(active_ids)

    fig, ax = plt.subplots(figsize=(7, 7))
    y_pos = np.arange(n)

    ax.barh(y_pos, praises, height=0.7, color=_PALETTE["green"], alpha=0.85,
            label="Praise received", zorder=3)
    ax.barh(y_pos, criticisms, height=0.7, color=_PALETTE["red"], alpha=0.85,
            label="Criticism received", zorder=3)

    ax.axvline(0, color="#aaaaaa", linewidth=0.8, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(agent_labels, fontsize=8, color=_PALETTE["text"], family=_FONT_FAMILY)
    ax.set_xlabel("Cumulative Reputation Delta", color=_PALETTE["muted"], fontsize=10,
                  family=_FONT_FAMILY)
    ax.grid(True, axis="x", color=_PALETTE["grid"], linewidth=0.4, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, facecolor=_PALETTE["surface"],
              edgecolor=_PALETTE["grid"], labelcolor=_PALETTE["text"],
              prop={"family": _FONT_FAMILY})
    _apply_theme(ax, "Reputation Standing", "Sorted by net reputation (praise \u2212 criticism)")

    out = out_path or (run_dir / "chart_reputation.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ── 3. Drama Timeline ──────────────────────────────────────────

_CONFLICT_KEYWORDS = [
    "belittle", "attack", "betray", "threaten", "warn", "confront",
    "manipulat", "deceiv", "scheme", "plot", "assassin", "poison",
    "challenge", "duel", "seize", "overthrow", "undermine", "accuse",
]


def chart_drama_timeline(run_dir: Path, out_path: Path | None = None) -> Path:
    events = _load_events(run_dir)
    max_tick = max((e.get("tick", 0) for e in events), default=0)
    if max_tick == 0:
        raise ValueError("No ticks in event log")

    buckets: dict[str, list[int]] = {k: [0] * (max_tick + 1) for k in _EVENT_COLORS}

    for e in events:
        t = e.get("tick", 0)
        etype = e.get("type", "")
        if etype == "death":
            buckets["death"][t] += 1
        elif etype == "friendship":
            buckets["friendship"][t] += 1
        elif etype == "reputation" and e.get("delta", 0) <= -0.2:
            buckets["betrayal"][t] += 1
        elif etype == "transfer" and abs(e.get("amount", 0)) >= 3.0:
            buckets["transfer"][t] += 1
        elif etype == "action":
            content = e.get("content", "").lower()
            if any(kw in content for kw in _CONFLICT_KEYWORDS):
                buckets["conflict"][t] += 1

    ticks = np.arange(max_tick + 1)
    fig, ax = plt.subplots(figsize=(7, 7))

    bottom = np.zeros(max_tick + 1)
    labels_order = ["friendship", "transfer", "conflict", "betrayal", "death"]
    label_display = {
        "friendship": "Friendships",
        "transfer": "Big Transfers",
        "conflict": "Conflicts",
        "betrayal": "Betrayals",
        "death": "Deaths",
    }

    for key in labels_order:
        vals = np.array(buckets[key], dtype=float)
        ax.bar(ticks, vals, bottom=bottom, width=1.0, color=_EVENT_COLORS[key],
               label=label_display[key], edgecolor="none", alpha=0.88)
        bottom += vals

    total_drama = bottom
    peak_tick = int(np.argmax(total_drama))
    peak_val = total_drama[peak_tick]
    if peak_val > 0:
        ax.annotate(f"Peak drama\ntick {peak_tick}", xy=(peak_tick, peak_val),
                    xytext=(peak_tick + max_tick * 0.08, peak_val * 0.9),
                    fontsize=8, color=_PALETTE["gold"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=_PALETTE["gold"], lw=1.2))

    ax.set_xlabel("Tick", color=_PALETTE["muted"], fontsize=10, family=_FONT_FAMILY)
    ax.set_ylabel("Dramatic Events", color=_PALETTE["muted"], fontsize=10, family=_FONT_FAMILY)
    ax.set_xlim(-0.5, max_tick + 0.5)
    ax.legend(loc="upper left", fontsize=8, facecolor=_PALETTE["surface"],
              edgecolor=_PALETTE["grid"], labelcolor=_PALETTE["text"],
              prop={"family": _FONT_FAMILY})
    ax.grid(True, axis="y", color=_PALETTE["grid"], linewidth=0.4, alpha=0.5)
    _apply_theme(ax, "Drama Timeline", "Stacked dramatic events per tick")

    out = out_path or (run_dir / "chart_drama.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ── 4. Character Alignment Scatter ─────────────────────────────

def chart_alignment(run_dir: Path, out_path: Path | None = None) -> Path:
    events = _load_events(run_dir)
    names = _id_to_name(events)
    max_tick = max((e.get("tick", 0) for e in events), default=0)

    deaths: dict[str, int] = {}
    work_count: dict[str, int] = defaultdict(int)
    action_count: dict[str, int] = defaultdict(int)
    praise_given: dict[str, int] = defaultdict(int)
    criticism_given: dict[str, int] = defaultdict(int)
    birth_tick: dict[str, int] = {}

    for e in events:
        aid = e.get("agent", "")
        etype = e.get("type", "")
        if etype == "birth":
            birth_tick[aid] = e.get("tick", 0)
        elif etype == "death":
            deaths[aid] = e.get("tick", 0)
        elif etype == "action":
            action_count[aid] += 1
            if e.get("mode") == "productive":
                work_count[aid] += 1
        elif etype == "reputation":
            src = e.get("source", "")
            if src:
                if e.get("delta", 0) > 0:
                    praise_given[src] += 1
                elif e.get("delta", 0) < 0:
                    criticism_given[src] += 1

    agent_ids = [aid for aid in names if action_count.get(aid, 0) > 0]
    if not agent_ids:
        raise ValueError("No agent activity found")

    xs, ys, sizes, colors, labels = [], [], [], [], []
    for aid in agent_ids:
        work_rate = work_count.get(aid, 0) / max(action_count.get(aid, 0), 1)
        disposition = praise_given.get(aid, 0) - criticism_given.get(aid, 0)
        lifespan = (deaths.get(aid, max_tick) - birth_tick.get(aid, 0))
        alive = aid not in deaths

        xs.append(work_rate)
        ys.append(disposition)
        sizes.append(max(40, lifespan * 3.5))
        colors.append(_PALETTE["green"] if alive else _PALETTE["red"])
        labels.append(names[aid])

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.axhline(0, color="#bbbbbb", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(0.5, color="#bbbbbb", linewidth=0.8, linestyle="--", alpha=0.7)

    quad_style = dict(fontsize=9, color=_PALETTE["muted"], alpha=0.30, fontweight="bold",
                      ha="center", va="center", family=_FONT_FAMILY)
    y_range = max(abs(min(ys, default=0)), abs(max(ys, default=0)), 1)
    ax.text(0.25, y_range * 0.75, "LOVABLE\nFREELOADER", **quad_style)
    ax.text(0.75, y_range * 0.75, "HARD-WORKING\nSAINT", **quad_style)

    rng = np.random.default_rng(42)
    jx = [x + rng.uniform(-0.02, 0.02) for x in xs]
    jy = [y + rng.uniform(-0.15, 0.15) * max(y_range * 0.05, 0.1) for y in ys]

    ax.scatter(jx, jy, s=sizes, c=colors, alpha=0.75, edgecolors="#333333", linewidths=0.6, zorder=4)

    placed: list[tuple[float, float]] = []
    for x, y, label in zip(jx, jy, labels):
        dx, dy = 8, 6
        if x > 0.65:
            dx = -8
            ha = "right"
        else:
            ha = "left"
        for px, py in placed:
            if abs(x - px) < 0.08 and abs(y - py) < (y_range * 0.12):
                dy += 12
        placed.append((x, y))
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.5, color=_PALETTE["text"], fontweight="bold", zorder=5,
                    family=_FONT_FAMILY, ha=ha, clip_on=True)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_PALETTE["green"],
               markersize=9, label="Survived"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_PALETTE["red"],
               markersize=9, label="Died"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
              facecolor=_PALETTE["surface"], edgecolor=_PALETTE["grid"], labelcolor=_PALETTE["text"],
              prop={"family": _FONT_FAMILY})

    ax.set_xlabel("Work Rate (industriousness)", color=_PALETTE["muted"], fontsize=10, family=_FONT_FAMILY)
    ax.set_ylabel("Net Social Disposition (praises \u2212 criticisms)", color=_PALETTE["muted"], fontsize=10, family=_FONT_FAMILY)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.4, alpha=0.4)
    _apply_theme(ax, 'Character Alignment: "Who Are You Really?"',
                 "Dot size = lifespan \u00b7 Green = survived \u00b7 Red = died")

    out = out_path or (run_dir / "chart_alignment.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ── Public API ──────────────────────────────────────────────────

def generate_all_charts(run_dir: Path) -> list[Path]:
    """Generate all four charts for a simulation run. Returns paths to saved PNGs."""
    run_dir = Path(run_dir)
    results: list[Path] = []

    chart_fns = [
        ("Survival curve", chart_survival),
        ("Reputation heatmap", chart_reputation_heatmap),
        ("Drama timeline", chart_drama_timeline),
        ("Alignment scatter", chart_alignment),
    ]

    for label, fn in chart_fns:
        try:
            path = fn(run_dir)
            results.append(path)
            print(f"  ✓ {label}: {path}")
        except Exception as exc:
            print(f"  ✗ {label}: {exc}")

    return results
