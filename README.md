# nanosim

![nanosim](nanolife.png)

<p align="center">
  <img src="charts/chart_alignment.png" width="32%" />
  <img src="charts/chart_reputation.png" width="32%" />
  <img src="charts/chart_drama.png" width="32%" />
  <br><em>Figure 1 — Gemini 2.5 Flash — nanothrones, 22 agents, 1 tick = 4h, 80 ticks</em>
</p>

---

> **Mission.** nanosim is a minimalist real-time implementation of evolutionary biology and social dynamics with LLMs, where agents have full free-will and mimic the being's instinct to live and to act.

The best Multi-Agent Simulator $1 can buy. Free-will is the design constraint: agents are never given a menu of allowed actions, never scripted, never reward-shaped. On every tick they observe, feel their needs, and choose what to do in natural language. Evolution, economy, and society are what emerges when many such wills run in parallel under scarcity.

nanosim is the smallest multi-agent LLM-based Artificial Life harness — a modular & paralel arhitecture with implementations of Darwinian selection, Maslow's hierarchy of needs, Malthusian economics, Lamarckian inheritance. ~350 lines of code in the main simulation engine `engine.py`. You can theoretically run this simulation ad infinitum. A simulation run costs from $0.1 (eg. gpt-oss-120b) to $5 (eg. Gemini-2.5-flash) and more (eg. Calude-Opus-4.6). The heartbeat is a configurable parameter (eg. '4h', 'day', etc) representing the time passage. No vector databases, no embeddings, just a minimalist implementation with 7 rules: scarcity, harshness, reputation, heredity, compression, local observation, event log.

- **nanochat** by <a href="https://x.com/karpathy">karpathy</a> distilled language model training to its irreducible core.
- **Conway's Game of Life** distilled life to its mathematical essence.
- **nanosim** distills Artificial Life with LLMs to its irreducible core.

For fun, there are multiple scenarios: `nanothrones`, `nanoception`, `nanomatrix`, `nanorings`, `nanozombie`, `nanopoter`

<p align="center">
  <img src="nanolife-terminal.png" width='80%' />
  <br><em>Figure 2 — nanosim terminal</em>
</p>

> **Disclaimer:** The purpose of nanosim is to build a minimal implementation of LLM-based Artificial Life. This build can be further optimized. The simulation needs careful balancing and exposes the main weakness of Large Language Models: LLMs are semantic engines and struggle with understanding numerical values. This holds true especially for smaller models like gpt-oss-120b, where the scenarios often collapse to mass extinction (see Figure 3). This leads to agents who occasionally do nonsensical things.

<p align="center">
  <img src="charts/groq-chart_survival.png" width="49%" />
  <img src="charts/groq_chart_alignment.png" width="49%" />
  <br><em>Figure 3 — Groq · GPT-OSS-120B — nanothrones, 22 agents, 1 tick = 4h, 80 ticks</em>
</p>

## Quickstart

```bash
git clone https://github.com/cris/nanosim.git && cd nanosim

pip install -r requirements.txt

cp .env.example .env   # paste your free Groq key from console.groq.com

python -m scripts.simulate --scenario=nanothrones --agents=15
```

## Setup

1. Get a free API key at [console.groq.com](https://console.groq.com)
2. Copy `.env.example` to `.env` and paste your key:
   ```
   GROQ_API_KEY=gsk_your-key-here
   ```
3. (Optional) Add an [OpenRouter](https://openrouter.ai) key for Gemini support:
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   ```
4. (Optional) Configure Google Vertex AI:
   ```
   VERTEX_PROJECT_ID=your-gcp-project-id
   VERTEX_LOCATION=us-central1
   # Optional if gcloud auth is unavailable in your environment:
   # VERTEX_ACCESS_TOKEN=ya29....
   ```
5. Run a simulation. That's it.
   ```
   python -m scripts.simulate
   ```

## Running a Simulation

```bash
python -m scripts.simulate [OPTIONS]

  --agents N          Number of starting agents (default: 10)
  --ticks N           Number of ticks to simulate (default: 50)
  --harshness F       World harshness 0.0-1.0 (default: 0.5)
  --tick-unit UNIT    minute/hour/4h/day/week (default: 4h)
  --scenario NAME     Load a scenario from scenarios/
  --model MODEL       Model for cognition (default depends on provider)
  --report-model M    Model for postmortem report (default depends on provider)
  --seed N            Random seed for reproducible runs (default: 42)
  --no-report         Skip postmortem report generation
  --x-artifacts       Generate X-ready metric card, replay GIF, and thread draft
  --x-max-moments N   Max highlighted moments in X artifacts (default: 12)
  --vertex            Use Vertex AI OpenAI-compatible endpoint (default)
  --open-router       Use OpenRouter (Gemini 2.5 Flash) instead of Vertex
  --groq              Use Groq (GPT-OSS-120B) instead of Vertex
```

Everything happens in the terminal (Figure 2). A fullscreen Rich dashboard shows the simulation live: world stats, scrolling event feed, agent roster, spotlight, and emergence index.

When the simulation ends — whether by tick limit, extinction, or Ctrl+C — a **postmortem** runs automatically:

1. **Emergence analysis** (instant, no LLM) — detects 11 phenomena (alliances, leadership, factions, betrayals, ostracism, generational transmission, cultural drift, wealth concentration, economic dependency, resource warfare, free riding)
2. **Academic HTML report** (LLM) — a formal evaluation paper with agent case studies, economic analysis, and critical timeline — auto-opens in browser
3. **Charts** (matplotlib) — survival, alignment, reputation, drama curves
4. **Social graph viewer** — interactive HTML visualization of the social network

Examples:

```bash
# Full nanothrones with all 15 characters
python -m scripts.simulate --scenario=nanothrones --agents=15

# Zombie apocalypse, 100 ticks
python -m scripts.simulate --scenario=nanozombie --agents=13 --ticks=100

# Frontier colony with a faster model, no report
python -m scripts.simulate --scenario=colony --agents=20 --model=llama-3.1-8b-instant --no-report

# Use OpenRouter with Gemini 2.5 Flash
python -m scripts.simulate --scenario=nanothrones --open-router

# OpenRouter with a specific model
python -m scripts.simulate --open-router --model google/gemini-2.0-flash-001

# Vertex AI (uses gcloud auth token or VERTEX_ACCESS_TOKEN)
python -m scripts.simulate --scenario=nanothrones --vertex --model google/gemini-2.5-flash

# Deterministic run + social-ready outputs
python -m scripts.simulate --scenario=nanothrones --ticks=30 --seed=7 --x-artifacts
```

### Providers

| Provider   | Flag            | Default Model             | Auth / Key |
| ---------- | --------------- | ------------------------- | ---------- |
| Vertex AI  | _(default)_     | `google/gemini-2.5-flash` | `VERTEX_PROJECT_ID` + gcloud auth token (or `VERTEX_ACCESS_TOKEN`) |
| OpenRouter | `--open-router` | `google/gemini-2.5-flash` | `OPENROUTER_API_KEY` |
| Groq       | `--groq`        | `openai/gpt-oss-120b`     | `GROQ_API_KEY` |

Override the model with `--model` on either provider. Any OpenAI-compatible model string works.

## X-ready Artifacts (card + replay + thread)

To make runs instantly shareable on X, generate artifacts from any completed run:

```bash
# During simulation:
python -m scripts.simulate --scenario=nanothrones --ticks=30 --seed=7 --x-artifacts

# Or from an existing run folder:
python -m scripts.x_artifacts --run-dir logs/runs/<run_id>
```

Outputs:
- `x_metric_card.png` — one-image summary (ticks, pop delta, emergence, top moments)
- `x_replay.gif` — short highlight replay
- `x_thread.md` — draft post/thread with key numbers + moments
- `x_summary.json` — machine-readable summary for tooling/automation

## Reproducible Scenario Packs

Scenario pack files live in `scenarios/packs/`:
- `research_pack.json` — broader cross-scenario evaluation
- `drama_pack.json` — high-drama runs for content + demos

Run a pack:

```bash
python -m scripts.run_pack --pack scenarios/packs/research_pack.json
python -m scripts.run_pack --pack scenarios/packs/drama_pack.json --x-artifacts
```

## Fun Scenarios

| Scenario      | Universe              | Core tension                            |
| ------------- | --------------------- | --------------------------------------- |
| `nanothrones` | Game of Thrones       | Power, betrayal, succession             |
| `nanozombie`  | Walking Dead          | Extreme scarcity, trust collapse        |
| `nanoception` | Inception             | Nested realities, information asymmetry |
| `nanomatrix`  | The Matrix            | Simulated reality, belief divergence    |
| `nanopoter`   | Harry Potter          | House rivalry, dark lord rising         |
| `nanorings`   | Lord of the Rings     | Corruption, fellowship, sacrifice       |
| `colony`      | Frontier settlement   | Cooperation vs. individual survival     |
| `island`      | Shipwreck survival    | Leadership vacuum, scarce resources     |
| `divide`      | Factional coexistence | Historical grievances, shared resources |

Create your own: copy `scenarios/base/custom.json` and fill in agents, resources, and world description.

## How It Works

**The engine enforces 7 primitives. Everything else is emergent.**

> Event Log · Local Observation · Scarcity · Harshness · Reputation · Heredity · Compression

No vector databases. No embeddings. No geographic grounding. Just 7 rules and an LLM.

Each tick, all agents act in parallel:

```
agents observe (local, not global) →
agents act (LLM calls in parallel via asyncio.gather) →
events logged → reputation updated → resources deplete →
births triggered → deaths triggered → log compressed →
next tick
```

### Intellectual Lineage

Every mechanic maps to a specific idea — and a specific line of code.

| Idea                               | Mechanic                                                                                    | Implementation                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Malthus** — Population trap      | Resources deplete each tick; births cost half of both parents' wealth                       | `engine.py`: drain = `base_drain + harshness`, gain = `base_gain * (1 - harshness) * reputation` |
| **Darwin** — Natural selection     | Variation (random traits), selection (starvation + old age), heredity (parent avg ± drift)  | `engine.py`: traits blended with ±0.1 drift; death if `resources <= 0` or `age >= lifespan`      |
| **Lamarck** — Cultural inheritance | Children inherit parents' compressed autobiographies, not just traits                       | `engine.py`: `identity_md` passed from parents to child at birth                                 |
| **Maslow** — Hierarchy of needs    | System prompt shifts with resource level: survival → stability → self-actualization         | `prompts.py`: resource thresholds at 3 and 7 reshape LLM behavior                                |
| **Smith** — Reputation as currency | Income multiplied by reputation; you cannot increase your own — only others can             | `engine.py` + `prompts.py`: praise economy drives cooperation                                    |
| **Emergence**                      | Alliances, factions, betrayals — none coded as behaviors; detected post-hoc from event logs | `postmortem.py`: pattern detectors run over event log after simulation ends                      |

## Emergence Detection

The postmortem detects emergent phenomena from event logs (no LLM needed). Emergence index: N/11.

| Phenomenon                | Detection                                                           |
| ------------------------- | ------------------------------------------------------------------- |
| Alliance                  | 3+ agents with mutual positive reputation                           |
| Leadership                | 1 agent praised by >50% of population                               |
| Faction split             | 3+ negative inter-cluster pairs and 2+ positive intra-cluster pairs |
| Betrayal                  | Friendship followed by strongly negative rep (delta < -0.2)         |
| Ostracism                 | One agent with negative rep from >70% of population                 |
| Generational transmission | Births where child inherits parent identity                         |
| Cultural drift            | 5+ novel terms in compressions not present in original scenario     |
| Wealth concentration      | One agent accumulates >30% of total transfer volume                 |
| Economic dependency       | Repeated transfers from same source totaling >= 10.0                |
| Resource warfare          | Agent drained to death via negative transfers (>= 5.0 stolen)       |
| Free riding               | Work rate < 30% but received >= 5.0 in transfers from others        |

## Benchmarking

Compare LLM providers head-to-head on the same scenario:

```bash
python -m scripts.benchmark [OPTIONS]

  --scenario NAME     Scenario to run (default: nanothrones)
  --ticks N           Ticks per run (default: 80)
  --agents N          Number of agents (default: all from scenario)
  --tick-unit UNIT    Override tick unit
```

Runs Groq and OpenRouter sequentially with identical parameters, then prints a side-by-side comparison. Results are saved to `logs/benchmarks/`.

### Sample Results: `nanothrones` — 22 agents, 80 ticks

| Metric                          |    Groq · GPT-OSS-120B |                         OpenRouter · Gemini 2.5 Flash |
| ------------------------------- | ---------------------: | ----------------------------------------------------: |
| **Performance**                 |                        |                                                       |
| Wall time                       |                 459.8s |                                                885.1s |
| LLM calls                       |                  1,816 |                                                 2,383 |
| Total tokens                    |              2,174,085 |                                             4,914,251 |
| Total cost                      |                  $0.49 |                                                 $2.18 |
| **Simulation Outcome**          |                        |                                                       |
| Final population                |                      1 |                                                    19 |
| Births                          |                      8 |                                                    11 |
| Deaths                          |                     21 |                                                     6 |
| **Behavior Quality**            |                        |                                                       |
| Avg action length               |                5 chars |                                             245 chars |
| Avg thought length              |               33 chars |                                             286 chars |
| Friendships formed              |                      9 |                                                     5 |
| Praises given                   |                     14 |                                                 1,527 |
| Rumors spread                   |                    436 |                                                   565 |
| **Mode Distribution**           |                        |                                                       |
| Productive                      |              893 (99%) |                                             748 (63%) |
| Social                          |                11 (1%) |                                             419 (35%) |
| Rest                            |                 0 (0%) |                                               19 (2%) |
| **Emergence**                   |                        |                                                       |
| Emergence index                 |                   2/11 |                                                  4/11 |
| Phenomena                       | alliance, generational | alliance, cultural drift, faction split, generational |
|                                 |                        |                                                       |
| **Verdict (5-point heuristic)** |                **1/5** |                                               **4/5** |

Requires both `GROQ_API_KEY` and `OPENROUTER_API_KEY` in `.env`.

### Statistical significance: comparing two sweeps

`scripts/metrics.py` also ships a `compare` subcommand for asking "did this
change actually move the metric, or just shuffle noise?". Given two
`summary.json` files produced by `sweep`, it runs Welch's two-sample t-test
(unequal variances) and Cohen's *d* on the per-seed values for every shared
scenario × headline metric. Pure stdlib, zero LLM, deterministic.

```bash
# 1. run a baseline sweep
python -m scripts.metrics sweep --scenarios nanothrones nanoception \
    --seeds 0 1 2 3 4 --ticks 30 --agents 8
# -> writes logs/sweeps/<id>/summary.json

# 2. change something (model, prompt, primitive…) and run again
python -m scripts.metrics sweep --scenarios nanothrones nanoception \
    --seeds 0 1 2 3 4 --ticks 30 --agents 8

# 3. compare
python -m scripts.metrics compare \
    logs/sweeps/<baseline_id>/summary.json \
    logs/sweeps/<experiment_id>/summary.json \
    --label-a baseline --label-b experiment \
    -o logs/sweeps/compare.json
```

Output:

```
compare: baseline  vs  experiment
=================================
scenario       metric                    mean_baseli    mean_experi       diff   cohen_d         p sig
------------------------------------------------------------------------------------------------------
nanothrones    survival_rate                  0.1100         0.9200    -0.8100   -81.000    0.0000 *
nanothrones    cooperation_index              0.0533         0.8500    -0.7967   -97.571    0.0000 *
nanothrones    narrative_coherence            0.3033         0.3033     0.0000     0.000    1.0000
...
```

Rows flagged with `*` are statistically significant at p < 0.05. The Welch
implementation is cross-checked against `scipy.stats.ttest_ind(equal_var=False)`
to ≤5e-5 on p and ≤1e-3 on t (see `tests/test_significance.py`).

## File Structure

```
nanosim/
├── nanosim/
│   ├── engine.py          # THE simulation loop (~350 lines, parallel LLM calls)
│   ├── common.py          # Agent dataclass, Event type, utilities
│   ├── world.py           # Clock, EventLog, WorldState
│   ├── interfaces.py      # CognitiveFunction, CompressionFunction, SpreadFunction, Scenario
│   ├── prompts.py         # System + turn + reflection prompt templates
│   ├── logger.py          # Append-only JSONL writer
│   ├── render.py          # Rich map-centered dashboard renderer
│   ├── postmortem.py      # Emergence analysis + HTML report
│   ├── charts.py          # Matplotlib chart generation
│   ├── scenario_loader.py # JSON scenario loader
│   └── defaults/
│       ├── cognitive.py   # LLM cognitive function (Groq / OpenRouter)
│       ├── compression.py # Log compression (LLM or stub)
│       └── spread.py      # Rumor propagation
├── scenarios/
│   ├── base/              # Starter scenarios (colony, island, divide, custom)
│   ├── nanothrones/       # Game of Thrones
│   ├── nanozombie/        # Walking Dead
│   ├── nanoception/       # Inception
│   ├── nanomatrix/        # The Matrix
│   ├── nanopoter/         # Harry Potter
│   └── nanorings/         # Lord of the Rings
├── scripts/
│   ├── simulate.py        # Run a simulation
│   └── benchmark.py       # Compare providers side-by-side
├── runs/
│   └── quickstart.sh      # One-command demo
├── social_graph_viewer.html  # Social graph visualizer (copied into each run dir)
├── requirements.txt
├── .env.example           # Template for API keys
└── README.md
```

## Extension Points

Swap exactly one interface. The engine doesn't care which implementation you use.

```python
class CognitiveFunction:     # how an agent decides → default: Groq LLM call
class CompressionFunction:   # how history shrinks → default: LLM summary
class SpreadFunction:        # how rumors travel → default: random neighbor + degradation
class Scenario:              # JSON world definition → primary user surface
```

If you have to read `engine.py` to fork nanosim, the abstraction failed.

## Cost

Groq is free for low-volume usage. Estimated costs for larger runs:

| Agents | Provider   | Model                     | Ticks | Estimated Cost |
| ------ | ---------- | ------------------------- | ----- | -------------- |
| 15     | Groq       | `openai/gpt-oss-120b`     | 50    | ~$0.10         |
| 22     | Groq       | `openai/gpt-oss-120b`     | 80    | ~$0.50         |
| 22     | OpenRouter | `google/gemini-2.5-flash` | 80    | ~$2.20         |

## Acknowledgements

- The name and philosophy derive from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) — the idea that you can distill a complex system to its irreducible core and it still works.
- [Groq](https://groq.com) for blazing-fast free-tier inference that makes running 20 parallel agents practical.
- [OpenRouter](https://openrouter.ai) for unified access to Gemini, Claude, and the rest of the frontier.
- [Rich](https://github.com/Textualize/rich) by Will McGugan for the terminal dashboard that makes the simulation watchable.

## Future Work

nanosim is far from complete in this current state, and can be advanced on multiple fronts:

- **PyPI package** — `pip install nanosim` with a CLI entry point so running a simulation is a one-liner.
- **Spatial awareness** — Replace flat location lists with a coordinate graph so agents reason about distance, travel time, and line-of-sight. This unlocks territorial behavior and migration.
- **Inter-agent trade & negotiation** — Agents currently cooperate or compete; a simple barter protocol would let resource scarcity drive alliances and betrayal organically.
- **Benchmark suite** — `scripts/metrics.py` ships a reproducible harness: `score <run_dir>` computes survival rate, cooperation index, narrative coherence, action diversity, and emergence index from any run's `world.jsonl`; `sweep --scenarios ... --seeds ...` runs a grid and aggregates with mean+stdev; `compare a.json b.json` runs Welch's t-test + Cohen's *d* per scenario × metric and flags significant differences at p<0.05. Pure Python, zero LLM. (More scenarios still welcome.)
- **Others:** — Long-horizon memory, Emotional state model, Head-to-head LLMs

## License

MIT
