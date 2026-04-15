#!/usr/bin/env bash
# quickstart.sh — One-command setup and demo run for nanolife.
#
# Installs Python dependencies and runs a short plain-text simulation
# so you can verify everything works before configuring an API key.
set -e

echo "╔═══════════════════════════════════════╗"
echo "║  nanolife — quickstart                ║"
echo "╚═══════════════════════════════════════╝"
echo ""

# Install dependencies
echo "→ Installing dependencies..."
pip3 install -q openai rich matplotlib numpy

# Run a simulation
echo "→ Running simulation (10 agents, 30 ticks)..."
NANOLIFE_PLAIN=1 python3 -m scripts.simulate --agents=10 --ticks=30

echo ""
echo "Done! For LLM-powered simulation, set OPENROUTER_API_KEY."
echo "Try: export OPENROUTER_API_KEY=sk-or-..."
echo "     python3 -m scripts.simulate --scenario=nanothrones --agents=15 --ticks=50"
