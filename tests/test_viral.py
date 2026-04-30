from __future__ import annotations

import json
from pathlib import Path

from nanosim.viral import generate_x_artifacts, summarize_run


def _write_world(path: Path) -> None:
    events = [
        {"tick": 0, "type": "birth", "agent": "a1", "content": "Alice enters the world. Goal: survive"},
        {"tick": 0, "type": "birth", "agent": "b1", "content": "Bob enters the world. Goal: survive"},
        {"tick": 1, "type": "action", "agent": "a1", "content": "work", "mode": "productive"},
        {"tick": 1, "type": "action", "agent": "b1", "content": "work", "mode": "productive"},
        {"tick": 2, "type": "friendship", "agent": "a1", "content": "Alice became friends with Bob"},
        {"tick": 2, "type": "reputation", "agent": "b1", "source": "a1", "delta": 0.3, "content": "Alice praised Bob"},
        {"tick": 3, "type": "transfer", "agent": "a1", "target": "b1", "amount": 4.0, "content": "Alice gave 4.0 to Bob"},
        {"tick": 4, "type": "death", "agent": "b1", "cause": "starvation", "content": "Bob has died — cause: starvation."},
    ]
    path.write_text("\n".join(json.dumps(e) for e in events))


def test_summarize_run_extracts_highlights(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_world(run_dir / "world.jsonl")

    summary = summarize_run(run_dir)
    assert summary["ticks"] == 4
    assert summary["initial_agents"] == 2
    assert summary["deaths"] == 1
    assert summary["births"] == 0
    assert summary["highlights"]
    assert any(h["kind"] == "death" for h in summary["highlights"])


def test_generate_x_artifacts_outputs_files(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_world(run_dir / "world.jsonl")

    outputs = generate_x_artifacts(run_dir=run_dir, max_moments=6)
    for key in ("summary_json", "metric_card", "replay_gif", "thread_template"):
        assert key in outputs
        assert Path(outputs[key]).exists()
