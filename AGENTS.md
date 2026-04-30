# AGENTS.md

## Cursor Cloud specific instructions

### Overview

nanosim is a single-product Python CLI application — a multi-agent LLM-based Artificial Life simulator. No web framework, no database, no Docker. The entire product runs in the terminal.

### Running tests

```bash
pytest tests/ -v
```

All tests are pure-Python unit tests (no LLM, no network). They run in <1s.

### Running the simulation

The simulation requires an LLM API key. Set at least one of:
- `GROQ_API_KEY` (default provider, free tier at console.groq.com)
- `OPENROUTER_API_KEY` (for Gemini models via `--open-router` flag)

Then run:
```bash
python -m scripts.simulate --scenario=nanothrones --agents=5 --ticks=10 --no-report
```

Without an API key, the CLI exits immediately with `[FATAL] GROQ_API_KEY not set`.

### Key caveats

- **No linter configured in repo**: The project has no `pyproject.toml`, `.flake8`, or linting config. You can use `ruff check .` for basic checks but the existing codebase has ~53 minor lint warnings (unused imports/variables) that are intentional/known.
- **PATH for pip-installed scripts**: User-installed scripts (pytest, ruff, etc.) land in `~/.local/bin`. Ensure `export PATH="$HOME/.local/bin:$PATH"` is active in your shell.
- **`.env` file loading**: `scripts/simulate.py` auto-loads `.env` from the project root. You can copy `.env.example` to `.env` and fill in keys, or export them directly.
- **No build step**: Run Python files directly; there is no `setup.py`, `pyproject.toml`, or package build.
- **Scenario loader**: Scenarios live in `scenarios/` as JSON. The engine loads them via `nanosim.scenario_loader.load_scenario("<name>")`.

### Standard commands reference

See `README.md` for full CLI options, provider table, scenario list, and benchmarking instructions.
