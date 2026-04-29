"""Run a predefined scenario pack for reproducible comparisons.

Usage:
  python3 -m scripts.run_pack --pack drama_pack --provider groq --seed-base 100
  python3 -m scripts.run_pack --pack research_pack --provider vertex --x-artifacts
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a nanosim scenario pack")
    p.add_argument("--pack", type=str, required=True, help="Pack file under scenarios/packs (name or path)")
    p.add_argument("--provider", choices=["groq", "openrouter", "vertex"], default="groq")
    p.add_argument("--seed-base", type=int, default=42, help="Base seed; per-entry seed increments from this")
    p.add_argument("--x-artifacts", action="store_true", help="Generate X artifacts after each run")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return p.parse_args()


def _pack_path(pack: str) -> Path:
    p = Path(pack)
    if p.exists():
        return p
    cand = ROOT / "scenarios" / "packs" / f"{pack}.json"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Pack not found: {pack}")


def _provider_flags(provider: str) -> list[str]:
    if provider == "groq":
        return []
    if provider == "openrouter":
        return ["--open-router"]
    if provider == "vertex":
        return ["--vertex"]
    raise ValueError(provider)


def _expand_runs(data: dict) -> list[dict]:
    """Normalize pack schema into a list of explicit run entries.

    Supported schemas:
      1) {"runs": [...]} or {"entries": [...]} with explicit objects
      2) {"scenarios": [...], "seeds": [...], "ticks": N, "agents": M}
    """
    explicit = data.get("runs") or data.get("entries")
    if explicit:
        return list(explicit)

    scenarios = data.get("scenarios", [])
    if not scenarios:
        return []
    seeds = data.get("seeds", [42])
    ticks = int(data.get("ticks", data.get("default_ticks", 30)))
    agents = data.get("agents", data.get("default_agents"))
    model = data.get("model")

    out: list[dict] = []
    for scenario in scenarios:
        for seed in seeds:
            row = {"scenario": scenario, "seed": int(seed), "ticks": ticks}
            if agents is not None:
                row["agents"] = int(agents)
            if model:
                row["model"] = model
            out.append(row)
    return out


def main() -> None:
    args = parse_args()
    path = _pack_path(args.pack)
    data = json.loads(path.read_text())
    runs = _expand_runs(data)
    if not runs:
        raise SystemExit(f"No runs in pack: {path}")

    print(f"Pack: {data.get('name', path.stem)}")
    print(f"Description: {data.get('description', '')}")
    print(f"Provider: {args.provider}")
    print(f"Entries: {len(runs)}")

    for i, entry in enumerate(runs):
        seed = int(entry.get("seed", args.seed_base + i))
        cmd = [
            sys.executable,
            "-m",
            "scripts.simulate",
            "--scenario",
            str(entry["scenario"]),
            "--ticks",
            str(entry["ticks"]),
            "--seed",
            str(seed),
            "--no-report",
        ]
        if entry.get("agents") is not None:
            cmd.extend(["--agents", str(entry["agents"])])
        if entry.get("model"):
            cmd.extend(["--model", str(entry["model"])])
        cmd.extend(_provider_flags(args.provider))
        if args.x_artifacts:
            cmd.append("--x-artifacts")

        print("\n" + "-" * 72)
        print(" ".join(cmd))
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
