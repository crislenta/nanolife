"""Reproducible metrics harness for nanosim.

Two modes:

  # 1. score an existing run directory (or several — glob supported)
  python -m scripts.metrics score logs/runs/20260423_073139
  python -m scripts.metrics score 'logs/runs/*'

  # 2. sweep a fixed scenario set with multiple seeds, then score all runs
  python -m scripts.metrics sweep \\
      --scenarios nanothrones nanoception \\
      --seeds 0 1 2 \\
      --ticks 30 --agents 8

The sweep mode delegates the actual simulation to ``scripts.simulate`` so
this file never has to know about the engine internals. The score mode is
pure read-only analysis.

Output is a single JSON blob + a pretty table. The JSON is what later
iterations compare against (``jq`` / ``diff`` / checked into PR bodies).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from nanosim.metrics import (
    aggregate_runs,
    compare_sweeps,
    compute_metrics,
    format_compare,
    format_table,
)


def _resolve_run_dirs(patterns: list[str]) -> list[Path]:
    """Expand a list of paths and globs into existing run directories."""
    out: list[Path] = []
    for p in patterns:
        if any(ch in p for ch in "*?["):
            for match in sorted(glob.glob(p)):
                path = Path(match)
                if path.is_dir():
                    out.append(path)
        else:
            path = Path(p)
            if path.is_dir():
                out.append(path)
    return out


def cmd_score(args: argparse.Namespace) -> int:
    run_dirs = _resolve_run_dirs(args.runs)
    if not run_dirs:
        print("no matching run directories", file=sys.stderr)
        return 1

    per_run = [compute_metrics(d) for d in run_dirs]
    print(format_table(per_run, title=f"metrics for {len(per_run)} run(s)"))

    if len(per_run) > 1:
        agg = aggregate_runs(run_dirs)
        print()
        print("aggregate")
        print("---------")
        for key, val in sorted(agg.items()):
            if key in ("runs", "per_run") or not key.endswith(("_mean", "_stdev")):
                continue
            print(f"  {key:<36s} {val}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(
                {"per_run": per_run} if len(per_run) == 1
                else aggregate_runs(run_dirs),
                indent=2,
            )
        )
        print(f"\nwrote {args.output}")
    return 0


def _run_simulate(scenario: str, ticks: int, agents: int | None, seed: int) -> Path | None:
    """Invoke ``scripts.simulate`` in a subprocess with a deterministic seed.

    Returns the run directory, or None on failure. We use a subprocess so
    that every sweep run starts from a clean global-state slate — the
    simulate script already writes to logs/runs/<timestamp>/, so we just
    find the latest dir after the run.
    """
    env = {**os.environ, "PYTHONHASHSEED": str(seed), "NANOLIFE_SEED": str(seed)}
    cmd = [
        sys.executable, "-m", "scripts.simulate",
        f"--scenario={scenario}",
        f"--ticks={ticks}",
        "--no-report",
    ]
    if agents is not None:
        cmd.append(f"--agents={agents}")

    before = _latest_run_dir()
    print(f"    -> {' '.join(cmd)}  (seed={seed})")
    proc = subprocess.run(cmd, env=env, cwd=_REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-20:])
        print(f"    FAIL exit={proc.returncode}\n{tail}")
        return None
    after = _latest_run_dir()
    if after is None or after == before:
        return None
    return after


def _latest_run_dir() -> Path | None:
    base = _REPO / "logs" / "runs"
    if not base.exists():
        return None
    subs = [p for p in base.iterdir() if p.is_dir()]
    if not subs:
        return None
    return max(subs, key=lambda p: p.stat().st_mtime)


def cmd_sweep(args: argparse.Namespace) -> int:
    start = time.time()
    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = _REPO / "logs" / "sweeps" / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"sweep id: {sweep_id}")
    print(f"sweep dir: {sweep_dir}")
    print(f"scenarios: {args.scenarios}  seeds: {args.seeds}  ticks: {args.ticks}  agents: {args.agents}")
    print()

    all_rows: list[dict] = []
    by_scenario: dict[str, list[Path]] = {s: [] for s in args.scenarios}

    for scenario in args.scenarios:
        print(f"scenario: {scenario}")
        for seed in args.seeds:
            random.seed(seed)
            run_dir = _run_simulate(scenario, args.ticks, args.agents, seed)
            if run_dir is None:
                continue
            metrics = compute_metrics(run_dir)
            metrics["_scenario"] = scenario
            metrics["_seed"] = seed
            all_rows.append(metrics)
            by_scenario[scenario].append(run_dir)

    print()
    print(format_table(all_rows, title="per-run metrics"))

    summary: dict = {
        "sweep_id": sweep_id,
        "scenarios": args.scenarios,
        "seeds": args.seeds,
        "ticks": args.ticks,
        "agents": args.agents,
        "elapsed_s": round(time.time() - start, 1),
        "per_run": all_rows,
        "aggregate_by_scenario": {},
    }
    for scenario, runs in by_scenario.items():
        if runs:
            summary["aggregate_by_scenario"][scenario] = aggregate_runs(runs)

    summary_path = sweep_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}")

    print()
    print("aggregates")
    print("----------")
    for scenario, agg in summary["aggregate_by_scenario"].items():
        print(f"  {scenario}:")
        for key in (
            "survival_rate_mean", "survival_rate_stdev",
            "cooperation_index_mean", "cooperation_index_stdev",
            "narrative_coherence_mean", "action_diversity_mean",
            "emergence_index_mean",
        ):
            if key in agg:
                print(f"    {key:<32s} {agg[key]}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two sweep summary.json files with Welch's t-test per metric."""
    path_a, path_b = Path(args.summary_a), Path(args.summary_b)
    if not path_a.is_file():
        print(f"summary not found: {path_a}", file=sys.stderr)
        return 1
    if not path_b.is_file():
        print(f"summary not found: {path_b}", file=sys.stderr)
        return 1
    summary_a = json.loads(path_a.read_text())
    summary_b = json.loads(path_b.read_text())
    label_a = args.label_a or summary_a.get("sweep_id") or path_a.stem
    label_b = args.label_b or summary_b.get("sweep_id") or path_b.stem
    report = compare_sweeps(summary_a, summary_b, label_a=label_a, label_b=label_b)
    print(format_compare(report))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.output}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="nanosim metrics harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    score = sub.add_parser("score", help="score one or more existing run dirs")
    score.add_argument("runs", nargs="+", help="paths or globs of run directories")
    score.add_argument("--output", "-o", default=None, help="write JSON summary here")
    score.set_defaults(func=cmd_score)

    sweep = sub.add_parser("sweep", help="run a scenario x seed sweep, then score")
    sweep.add_argument("--scenarios", nargs="+", required=True)
    sweep.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    sweep.add_argument("--ticks", type=int, default=20)
    sweep.add_argument("--agents", type=int, default=None)
    sweep.set_defaults(func=cmd_sweep)

    compare = sub.add_parser(
        "compare",
        help="compare two sweep summary.json files with Welch's t-test (significance)",
    )
    compare.add_argument("summary_a", help="path to first sweep summary.json")
    compare.add_argument("summary_b", help="path to second sweep summary.json")
    compare.add_argument("--label-a", default=None, help="display label for the first sweep")
    compare.add_argument("--label-b", default=None, help="display label for the second sweep")
    compare.add_argument("--output", "-o", default=None, help="write JSON report here")
    compare.set_defaults(func=cmd_compare)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
