"""Standardized reproducible metrics for nanolife runs.

This module is pure Python and has no runtime dependencies on LLMs or the
simulation engine — it reads a run's ``world.jsonl`` and returns a flat
dict of scalar metrics. That separation keeps the core engine under its
LOC budget and lets benchmark harnesses be mixed and matched freely.

The four headline metrics that match the README "Benchmark suite" item:

- ``survival_rate``       — population retention vs. starting size
- ``cooperation_index``   — positive social signal density per agent-tick
- ``narrative_coherence`` — how often an agent's follow-up action stays on
                            the same topic as its stated thought
- ``emergence_index``     — re-exported from ``postmortem.analyze`` so
                            benchmark output is a single flat dict

Every metric is deterministic given a run directory — no randomness, no
network calls. That is what makes it a benchmark.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .postmortem import _load_events, analyze


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at",
    "for", "with", "from", "by", "as", "is", "was", "are", "were", "be",
    "been", "being", "it", "its", "this", "that", "these", "those", "i",
    "you", "he", "she", "we", "they", "them", "my", "your", "his", "her",
    "our", "their", "me", "us", "him", "himself", "herself", "myself",
    "ourselves", "themselves", "do", "does", "did", "have", "has", "had",
    "will", "would", "should", "could", "can", "may", "might", "must",
    "not", "no", "so", "if", "then", "than", "just", "into", "over",
    "about", "up", "down", "out", "off", "again", "further", "very",
    "really", "still", "also", "some", "any", "all", "more", "most",
    "other", "such", "only", "own", "same", "too",
}


def _tokens(text: str) -> set[str]:
    return {
        t.lower() for t in _TOKEN_RE.findall(text or "")
        if t.lower() not in _STOPWORDS and len(t) > 2
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ──────────────────────────────────────────────────────────────────────
# Individual metrics
# ──────────────────────────────────────────────────────────────────────

def survival_rate(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Fraction of starting population still alive at the final tick.

    Starting population = distinct agent names born or present at tick 0.
    """
    starters: set[str] = set()
    dead: set[str] = set()
    later_births: set[str] = set()
    last_tick = 0
    for e in events:
        tick = e.get("tick", 0) or 0
        if tick > last_tick:
            last_tick = tick
        t = e.get("type")
        agent = e.get("agent")
        if t == "birth":
            if tick == 0 and agent:
                starters.add(agent)
            elif agent:
                later_births.add(agent)
        elif t == "death" and agent:
            dead.add(agent)
        elif tick == 0 and agent:
            starters.add(agent)

    starting_pop = len(starters)
    survivors = len(starters - dead)
    total_births = len(later_births)
    total_deaths = len(dead)
    rate = (survivors / starting_pop) if starting_pop else 0.0
    return {
        "survival_rate": round(rate, 4),
        "starting_population": starting_pop,
        "survivors": survivors,
        "total_births": total_births,
        "total_deaths": total_deaths,
        "final_tick": last_tick,
    }


def cooperation_index(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Positive social signal density.

    Numerator: praises (reputation delta > 0) + friendships + resource
    transfers. Denominator: total agent-ticks observed (actions). A value
    near 0 means nobody cooperated; values around 0.3+ are high.
    """
    praises = sum(
        1 for e in events
        if e.get("type") == "reputation" and (e.get("delta") or 0) > 0
    )
    criticisms = sum(
        1 for e in events
        if e.get("type") == "reputation" and (e.get("delta") or 0) < 0
    )
    friendships = sum(1 for e in events if e.get("type") == "friendship")
    transfers = sum(1 for e in events if e.get("type") == "transfer")
    actions = sum(1 for e in events if e.get("type") == "action")

    numerator = praises + friendships + transfers
    denom = max(actions, 1)
    index = numerator / denom

    # Ratio of positive to negative social signal (NaN-safe)
    pos_neg = praises / max(criticisms, 1) if (praises or criticisms) else 0.0

    return {
        "cooperation_index": round(index, 4),
        "praises": praises,
        "criticisms": criticisms,
        "friendships": friendships,
        "transfers": transfers,
        "actions": actions,
        "positive_to_negative_ratio": round(pos_neg, 2),
    }


def narrative_coherence(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Jaccard overlap between an agent's thought and their next action.

    For each (agent, tick) pair where both a thought and an action exist,
    we measure the lexical overlap of content tokens. Mean over all such
    pairs is the coherence score in [0, 1]. Higher = actions follow
    through on stated intent.
    """
    by_agent_tick: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for e in events:
        t = e.get("type")
        if t not in ("thought", "action"):
            continue
        agent = e.get("agent")
        tick = e.get("tick")
        if agent is None or tick is None:
            continue
        content = e.get("content", "")
        if t in by_agent_tick[(agent, tick)]:
            continue
        by_agent_tick[(agent, tick)][t] = content

    scores: list[float] = []
    action_texts: list[str] = []
    for (_, _), pair in by_agent_tick.items():
        if "thought" in pair and "action" in pair:
            scores.append(_jaccard(_tokens(pair["thought"]), _tokens(pair["action"])))
        if "action" in pair:
            action_texts.append(pair["action"])

    mean = sum(scores) / len(scores) if scores else 0.0

    # Diversity: distinct normalized action prefixes relative to total actions.
    # Catches the "action='work' monoculture" failure mode that motivated the
    # free-will fix — the old parser would score 1.0 on coherence here but 0.02
    # on diversity.
    normalized = [a.strip().lower()[:60] for a in action_texts if a.strip()]
    distinct = len(set(normalized))
    diversity = distinct / len(normalized) if normalized else 0.0

    # Shannon entropy over action prefixes — a richer diversity measure.
    counts = Counter(normalized)
    total = sum(counts.values())
    entropy = 0.0
    if total:
        for c in counts.values():
            p = c / total
            entropy -= p * math.log2(p)

    return {
        "narrative_coherence": round(mean, 4),
        "coherence_samples": len(scores),
        "action_diversity": round(diversity, 4),
        "action_entropy_bits": round(entropy, 4),
        "distinct_actions": distinct,
        "total_actions": len(normalized),
    }


# ──────────────────────────────────────────────────────────────────────
# Top-level
# ──────────────────────────────────────────────────────────────────────

HEADLINE_METRICS = (
    "survival_rate",
    "cooperation_index",
    "narrative_coherence",
    "action_diversity",
    "emergence_index",
)


def compute_metrics(run_dir: str | Path) -> dict[str, Any]:
    """Run every metric against a single run directory.

    Returns a flat dict. Keys beginning with an underscore are metadata.
    """
    run_dir = Path(run_dir)
    events = _load_events(run_dir)
    if not events:
        return {"_run_dir": str(run_dir), "_empty": True}

    out: dict[str, Any] = {
        "_run_dir": str(run_dir),
        "_event_count": len(events),
    }
    out.update(survival_rate(events))
    out.update(cooperation_index(events))
    out.update(narrative_coherence(events))

    emergence = analyze(run_dir)
    out["emergence_index"] = emergence.get("emergence_index", 0)
    out["emergence_max"] = emergence.get("max_emergence_index", 11)
    out["phenomena_detected"] = emergence.get("phenomena_detected", [])
    return out


def aggregate_runs(run_dirs: list[str | Path]) -> dict[str, Any]:
    """Aggregate metrics across multiple runs of the same scenario.

    Reports mean and stdev of each numeric headline metric so one can
    see whether a change moved the signal or just shuffled noise.
    """
    per_run = [compute_metrics(d) for d in run_dirs]
    valid = [m for m in per_run if not m.get("_empty")]
    agg: dict[str, Any] = {
        "runs": len(valid),
        "per_run": per_run,
    }
    if not valid:
        return agg

    numeric_keys = [
        k for k in valid[0]
        if isinstance(valid[0].get(k), (int, float)) and not k.startswith("_")
    ]
    for key in numeric_keys:
        values = [m.get(key, 0) for m in valid]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        agg[f"{key}_mean"] = round(mean, 4)
        agg[f"{key}_stdev"] = round(math.sqrt(var), 4)
    return agg


def format_table(rows: list[dict[str, Any]], title: str = "") -> str:
    """Render a list of metric dicts as a fixed-width table."""
    if not rows:
        return "(no data)"
    cols = ["_run_dir"] + [k for k in HEADLINE_METRICS if k in rows[0]]
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))
    header = f"{'run':<48s}  " + "  ".join(f"{c:>18s}" for c in cols[1:])
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        run = Path(str(row.get("_run_dir", "?"))).name[:48]
        vals: list[str] = []
        for c in cols[1:]:
            v = row.get(c)
            if isinstance(v, float):
                vals.append(f"{v:>18.4f}")
            elif isinstance(v, int):
                vals.append(f"{v:>18d}")
            else:
                vals.append(f"{str(v)[:18]:>18s}")
        lines.append(f"{run:<48s}  " + "  ".join(vals))
    return "\n".join(lines)
