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


# ──────────────────────────────────────────────────────────────────────
# Statistical significance: Welch's t-test + Cohen's d (stdlib only)
# ──────────────────────────────────────────────────────────────────────

def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-7) -> float:
    """Lentz's continued-fraction expansion for the incomplete beta."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b). Used for the t CDF."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value for Student's t with df degrees of freedom."""
    if df <= 0 or not math.isfinite(t):
        return 1.0
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def welch_t_test(a: list[float], b: list[float]) -> dict[str, float]:
    """Welch's two-sample t-test (unequal variances).

    Returns ``{t, df, p, mean_a, mean_b, mean_diff, cohen_d, n_a, n_b}``.
    Pure stdlib. Falls back to ``p=1, t=0`` when a sample is too small or
    both samples are constant — never raises.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return {
            "n_a": n_a, "n_b": n_b,
            "mean_a": (sum(a) / n_a) if n_a else 0.0,
            "mean_b": (sum(b) / n_b) if n_b else 0.0,
            "mean_diff": 0.0, "t": 0.0, "df": 0.0, "p": 1.0, "cohen_d": 0.0,
        }
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)
    diff = mean_a - mean_b
    se2 = var_a / n_a + var_b / n_b
    if se2 <= 0.0:
        # Both samples constant. p = 1 if equal, p = 0 if not.
        return {
            "n_a": n_a, "n_b": n_b, "mean_a": mean_a, "mean_b": mean_b,
            "mean_diff": diff, "t": 0.0, "df": float(n_a + n_b - 2),
            "p": 0.0 if diff != 0.0 else 1.0,
            "cohen_d": float("inf") if diff != 0.0 else 0.0,
        }
    t = diff / math.sqrt(se2)
    # Welch–Satterthwaite degrees of freedom.
    num = se2 * se2
    den = (var_a * var_a) / (n_a * n_a * (n_a - 1)) + (var_b * var_b) / (n_b * n_b * (n_b - 1))
    df = num / den if den > 0 else float(n_a + n_b - 2)
    p = _t_two_sided_p(t, df)
    # Cohen's d with pooled SD (n-1 weighting).
    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    cohen_d = diff / pooled_sd if pooled_sd > 0 else 0.0
    return {
        "n_a": n_a, "n_b": n_b,
        "mean_a": round(mean_a, 6), "mean_b": round(mean_b, 6),
        "mean_diff": round(diff, 6),
        "t": round(t, 4), "df": round(df, 2), "p": round(p, 6),
        "cohen_d": round(cohen_d, 4),
    }


def _per_run_values(summary: dict, scenario: str | None, metric: str) -> list[float]:
    """Pull one metric's per-run values out of a sweep ``summary.json`` dict.

    If ``scenario`` is given, only runs from that scenario are returned;
    otherwise all per_run entries contribute.
    """
    rows = summary.get("per_run", []) or []
    out: list[float] = []
    for row in rows:
        if scenario is not None and row.get("_scenario") != scenario:
            continue
        v = row.get(metric)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def compare_sweeps(
    summary_a: dict,
    summary_b: dict,
    metrics: tuple[str, ...] = HEADLINE_METRICS,
    label_a: str = "A",
    label_b: str = "B",
) -> dict[str, Any]:
    """Compare two sweep summaries with Welch's t-test per scenario × metric.

    Both summaries must follow the shape produced by ``cmd_sweep`` in
    ``scripts/metrics.py``: a top-level ``per_run`` list where each row has
    ``_scenario`` plus the headline numeric metrics.

    The comparison is pairwise: for every scenario present in BOTH sweeps and
    every metric in ``metrics``, we compute Welch's t-test on the per-seed
    values and report ``p``, ``t``, ``df``, mean_diff, and Cohen's d. Scenarios
    present in only one sweep are listed under ``unmatched_scenarios``.

    Returns a flat dict friendly to JSON dumping.
    """
    scenarios_a = sorted({r.get("_scenario") for r in summary_a.get("per_run", []) if r.get("_scenario")})
    scenarios_b = sorted({r.get("_scenario") for r in summary_b.get("per_run", []) if r.get("_scenario")})
    shared = [s for s in scenarios_a if s in scenarios_b]
    unmatched = [s for s in scenarios_a if s not in scenarios_b] + \
                [s for s in scenarios_b if s not in scenarios_a]

    by_scenario: dict[str, dict[str, dict[str, float]]] = {}
    significant: list[dict[str, Any]] = []
    for scenario in shared:
        by_scenario[scenario] = {}
        for metric in metrics:
            a_vals = _per_run_values(summary_a, scenario, metric)
            b_vals = _per_run_values(summary_b, scenario, metric)
            stats = welch_t_test(a_vals, b_vals)
            by_scenario[scenario][metric] = stats
            if stats["n_a"] >= 2 and stats["n_b"] >= 2 and stats["p"] < 0.05:
                significant.append({
                    "scenario": scenario, "metric": metric,
                    "p": stats["p"], "cohen_d": stats["cohen_d"],
                    "mean_a": stats["mean_a"], "mean_b": stats["mean_b"],
                })

    return {
        "label_a": label_a,
        "label_b": label_b,
        "shared_scenarios": shared,
        "unmatched_scenarios": sorted(set(unmatched)),
        "metrics": list(metrics),
        "by_scenario": by_scenario,
        "significant": significant,
    }


def format_compare(report: dict) -> str:
    """Render a compare_sweeps report as a fixed-width table."""
    lines: list[str] = []
    a, b = report.get("label_a", "A"), report.get("label_b", "B")
    title = f"compare: {a}  vs  {b}"
    lines.append(title)
    lines.append("=" * len(title))
    if not report.get("shared_scenarios"):
        lines.append("(no shared scenarios — nothing to compare)")
        return "\n".join(lines)
    if report.get("unmatched_scenarios"):
        lines.append(f"unmatched scenarios: {', '.join(report['unmatched_scenarios'])}")
        lines.append("")
    header = f"{'scenario':<14s} {'metric':<22s} {'mean_'+a[:6]:>14s} {'mean_'+b[:6]:>14s} {'diff':>10s} {'cohen_d':>9s} {'p':>9s} sig"
    lines.append(header)
    lines.append("-" * len(header))
    for scenario in report["shared_scenarios"]:
        for metric, s in report["by_scenario"][scenario].items():
            sig = "*" if (s["n_a"] >= 2 and s["n_b"] >= 2 and s["p"] < 0.05) else " "
            lines.append(
                f"{scenario:<14s} {metric:<22s} "
                f"{s['mean_a']:>14.4f} {s['mean_b']:>14.4f} "
                f"{s['mean_diff']:>10.4f} {s['cohen_d']:>9.3f} "
                f"{s['p']:>9.4f} {sig}"
            )
    if report.get("significant"):
        lines.append("")
        lines.append(f"significant differences (p<0.05): {len(report['significant'])}")
    return "\n".join(lines)
