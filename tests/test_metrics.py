"""Unit tests for the metrics harness.

Pure-Python, no LLM, no network — safe to run in CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanolife.metrics import (
    aggregate_runs,
    compute_metrics,
    cooperation_index,
    narrative_coherence,
    survival_rate,
    _jaccard,
    _tokens,
)


def test_tokens_strips_stopwords():
    toks = _tokens("I will hunt a deer in the forest")
    assert "hunt" in toks
    assert "deer" in toks
    assert "forest" in toks
    assert "the" not in toks
    assert "will" not in toks


def test_jaccard_edges():
    assert _jaccard(set(), {"a"}) == 0.0
    assert _jaccard({"a"}, set()) == 0.0
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def _synthetic_events():
    return [
        {"tick": 0, "type": "birth", "agent": "A"},
        {"tick": 0, "type": "birth", "agent": "B"},
        {"tick": 0, "type": "birth", "agent": "C"},
        {"tick": 1, "type": "thought", "agent": "A", "content": "I will hunt deer in the forest"},
        {"tick": 1, "type": "action", "agent": "A", "content": "hunt a deer in the forest"},
        {"tick": 1, "type": "thought", "agent": "B", "content": "plant wheat"},
        {"tick": 1, "type": "action", "agent": "B", "content": "sing a lullaby"},
        {"tick": 2, "type": "reputation", "source": "A", "agent": "B", "delta": 0.2},
        {"tick": 2, "type": "reputation", "source": "B", "agent": "A", "delta": 0.1},
        {"tick": 2, "type": "friendship", "agent": "A"},
        {"tick": 3, "type": "transfer", "agent": "A"},
        {"tick": 3, "type": "death", "agent": "C"},
        {"tick": 4, "type": "birth", "agent": "D"},
        {"tick": 5, "type": "action", "agent": "A", "content": "hunt a deer"},
    ]


def test_survival_rate():
    s = survival_rate(_synthetic_events())
    assert s["starting_population"] == 3
    assert s["survivors"] == 2
    assert s["total_births"] == 1
    assert s["total_deaths"] == 1
    assert s["survival_rate"] == round(2 / 3, 4)
    assert s["final_tick"] == 5


def test_cooperation_index():
    c = cooperation_index(_synthetic_events())
    assert c["praises"] == 2
    assert c["criticisms"] == 0
    assert c["friendships"] == 1
    assert c["transfers"] == 1
    assert c["actions"] == 3
    # (praises + friendships + transfers) / actions == 4 / 3
    assert c["cooperation_index"] == round(4 / 3, 4)


def test_narrative_coherence_and_diversity():
    n = narrative_coherence(_synthetic_events())
    # Agent A at tick 1 had both thought and action -> strong overlap.
    # Agent B at tick 1 had both -> zero overlap.
    # Agent A at tick 5 had no matching thought -> skipped.
    assert n["coherence_samples"] == 2
    assert n["narrative_coherence"] > 0.3
    # 3 actions, 3 are all unique after lowercase + prefix -> diversity 1.0
    assert n["distinct_actions"] == 3
    assert n["total_actions"] == 3
    assert n["action_diversity"] == 1.0


def test_diversity_detects_monoculture():
    """Regression test for the free-will fix: if every action is the literal
    string 'work', diversity should be near zero — exactly what the old
    parser fallback produced.
    """
    events = [
        {"tick": i, "type": "action", "agent": f"agent_{i % 10}", "content": "work"}
        for i in range(200)
    ]
    n = narrative_coherence(events)
    assert n["distinct_actions"] == 1
    assert n["action_diversity"] == round(1 / 200, 4)
    assert n["action_entropy_bits"] == 0.0


def test_compute_metrics_handles_empty_run(tmp_path: Path):
    result = compute_metrics(tmp_path)
    assert result.get("_empty") is True


def test_compute_metrics_on_real_jsonl(tmp_path: Path):
    events = _synthetic_events()
    (tmp_path / "world.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events)
    )
    result = compute_metrics(tmp_path)
    assert "_empty" not in result
    assert result["starting_population"] == 3
    assert result["survivors"] == 2
    assert result["cooperation_index"] > 0
    assert "emergence_index" in result


def test_aggregate_runs_computes_mean_and_stdev(tmp_path: Path):
    for i, seed in enumerate([1, 2, 3]):
        d = tmp_path / f"run_{i}"
        d.mkdir()
        events = _synthetic_events()
        (d / "world.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    agg = aggregate_runs([tmp_path / f"run_{i}" for i in range(3)])
    assert agg["runs"] == 3
    assert "survival_rate_mean" in agg
    # Identical runs -> stdev is zero.
    assert agg["survival_rate_stdev"] == 0.0
