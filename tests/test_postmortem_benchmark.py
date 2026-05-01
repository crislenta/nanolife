from __future__ import annotations

import asyncio
import json
from pathlib import Path

from nanosim.postmortem import benchmark_run_quality, run_postmortem


def test_benchmark_run_quality_relevant_case():
    events = [
        {"tick": 0, "type": "birth", "agent": "a1", "content": "Alice enters the world. Goal: survive"},
        {"tick": 0, "type": "birth", "agent": "b1", "content": "Bob enters the world. Goal: survive"},
        {"tick": 0, "type": "birth", "agent": "c1", "content": "Cara enters the world. Goal: survive"},
        {"tick": 0, "type": "birth", "agent": "d1", "content": "Dion enters the world. Goal: survive"},
        {"tick": 1, "type": "action", "agent": "a1", "mode": "productive", "content": "forage near the ridge"},
        {"tick": 1, "type": "action", "agent": "b1", "mode": "social", "content": "broker alliance with Cara"},
        {"tick": 1, "type": "action", "agent": "c1", "mode": "productive", "content": "build fish traps by the river"},
        {"tick": 1, "type": "action", "agent": "d1", "mode": "rest", "content": "recover in shelter"},
        {"tick": 2, "type": "action", "agent": "a1", "mode": "social", "content": "praise Bob's leadership"},
        {"tick": 2, "type": "action", "agent": "b1", "mode": "productive", "content": "organize supply convoy"},
        {"tick": 2, "type": "action", "agent": "c1", "mode": "social", "content": "negotiate food sharing pact"},
        {"tick": 2, "type": "action", "agent": "d1", "mode": "productive", "content": "repair water cistern"},
        {"tick": 2, "type": "friendship", "agent": "a1", "content": "Alice became friends with Bob"},
        {"tick": 2, "type": "reputation", "source": "a1", "agent": "b1", "delta": 0.2, "content": "Alice praised Bob"},
        {"tick": 2, "type": "transfer", "agent": "b1", "target": "c1", "amount": 4.0, "content": "Bob gave 4.0 to Cara"},
        {"tick": 3, "type": "action", "agent": "a1", "mode": "productive", "content": "hunt small game"},
        {"tick": 3, "type": "action", "agent": "b1", "mode": "social", "content": "mediate dispute"},
        {"tick": 3, "type": "action", "agent": "c1", "mode": "productive", "content": "expand trap network"},
        {"tick": 3, "type": "action", "agent": "d1", "mode": "productive", "content": "monitor storm shelters"},
    ]
    data = {"total_ticks": 3}
    emergence = {"emergence_index": 7, "max_emergence_index": 11}

    out = benchmark_run_quality(events, data, emergence)
    assert out["verdict"] == "RELEVANT"
    assert out["score"] >= 70
    assert out["components"]["diversity"] > 0
    assert out["signals"]["survival_ratio"] == 1.0


def test_benchmark_run_quality_slop_case():
    events = [
        {"tick": 0, "type": "birth", "agent": "a1", "content": "Alice enters the world. Goal: survive"},
        {"tick": 0, "type": "birth", "agent": "b1", "content": "Bob enters the world. Goal: survive"},
        {"tick": 1, "type": "action", "agent": "a1", "mode": "rest", "content": "pause and gather my thoughts"},
        {"tick": 1, "type": "action", "agent": "b1", "mode": "rest", "content": "pause and gather my thoughts"},
        {"tick": 1, "type": "parse_error", "agent": "a1", "content": "walk chosen but delta invalid"},
        {"tick": 1, "type": "parse_error", "agent": "b1", "content": "walk chosen but delta invalid"},
        {"tick": 2, "type": "action", "agent": "a1", "mode": "rest", "content": "pause and gather my thoughts"},
        {"tick": 2, "type": "action", "agent": "b1", "mode": "rest", "content": "pause and gather my thoughts"},
        {"tick": 2, "type": "death", "agent": "a1", "content": "Alice has died — cause: starvation."},
        {"tick": 2, "type": "death", "agent": "b1", "content": "Bob has died — cause: starvation."},
    ]
    data = {"total_ticks": 2}
    emergence = {"emergence_index": 0, "max_emergence_index": 11}

    out = benchmark_run_quality(events, data, emergence)
    assert out["verdict"] == "SLOP"
    assert out["score"] < 50
    assert out["signals"]["pause_rate"] >= 0.9
    assert any("parse errors" in r.lower() or "fallback" in r.lower() for r in out["recommendations"])


def test_run_postmortem_writes_executive_summary_json(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events = [
        {"tick": 0, "type": "birth", "agent": "a1", "content": "Alice enters the world. Goal: survive"},
        {"tick": 0, "type": "birth", "agent": "b1", "content": "Bob enters the world. Goal: survive"},
        {"tick": 0, "type": "world", "content": "SYSTEM_TICK_UNIT=day"},
        {"tick": 1, "type": "action", "agent": "a1", "mode": "productive", "content": "forage"},
        {"tick": 1, "type": "action", "agent": "b1", "mode": "social", "content": "praise Alice"},
        {"tick": 1, "type": "reputation", "source": "b1", "agent": "a1", "delta": 0.2, "content": "Bob praised Alice"},
    ]
    (run_dir / "world.jsonl").write_text("\n".join(json.dumps(e) for e in events))

    asyncio.run(run_postmortem(run_dir=run_dir, open_browser=False))

    exec_path = run_dir / "executive_summary.json"
    assert exec_path.exists()
    payload = json.loads(exec_path.read_text())
    assert payload["benchmark_version"] == "nanosim-benchmark-v1"
    assert payload["verdict"] in {"RELEVANT", "PROMISING", "SLOP"}
    assert isinstance(payload["recommendations"], list) and payload["recommendations"]
