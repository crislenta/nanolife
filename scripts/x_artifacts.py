"""Generate X-ready artifacts from a completed run directory.

Usage:
  python3 -m scripts.x_artifacts --run-dir logs/runs/<run_id>
  python3 -m scripts.x_artifacts --run-dir logs/runs/<run_id> --no-gif --max-moments 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

from nanosim.viral import generate_x_artifacts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate X-ready artifacts for a nanosim run")
    p.add_argument("--run-dir", type=str, required=True, help="Run directory containing world.jsonl")
    p.add_argument("--out-dir", type=str, default=None, help="Optional output directory (default: run dir)")
    p.add_argument("--max-moments", type=int, default=12, help="Max highlighted moments in replay/thread")
    p.add_argument("--title", type=str, default=None, help="Optional title override for card/GIF")
    p.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    p.add_argument("--no-card", action="store_true", help="Skip metric card generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None
    outputs = generate_x_artifacts(
        run_dir=run_dir,
        out_dir=out_dir,
        max_moments=args.max_moments,
        title=args.title,
        include_gif=not args.no_gif,
        include_card=not args.no_card,
    )
    print("Generated:")
    for key, path in outputs.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
