# Autonomous Worker for nanosim

This system runs 24/7 to continuously develop and improve nanosim through automated experimentation.

## What It Does

Every 4 hours, the autonomous worker:

1. **Runs experiments** — Executes simulations across multiple scenarios (nanothrones, colony, island)
2. **Analyzes results** — Detects patterns, failures, and emergent behaviors
3. **Identifies improvements** — Suggests code enhancements based on analysis
4. **Commits changes** — Pushes experiment logs and results to GitHub
5. **Sends reports** — Emails daily summaries of findings

## Components

- `autonomous_worker.py` — Main 24/7 work loop (runs as a background service)
- `send_report.py` — Email report generator
- `check_and_send_emails.py` — Email outbox processor
- `autonomous_logs/` — Experiment logs and reports
- `autonomous_experiments/` — Raw experiment data (JSON)

## Setup

### 1. API Keys (Required)

Add your API keys in the Nebula agent settings:

- `GROQ_API_KEY` — Get free key at https://console.groq.com
- `OPENROUTER_API_KEY` — (Optional) For Gemini 2.5 Flash at https://openrouter.ai

### 2. GitHub Deploy Key (Required for push access)

The autonomous worker needs write access to push commits. Add this deploy key to your GitHub repo:

**Deploy Key:**
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEOHBaNpGP0wqCwCcPSPQAlO5f6+st1h0KtQ28GH0T41
```

**Steps:**
1. Go to https://github.com/crislenta/nanosim/settings/keys
2. Click "Add deploy key"
3. Title: `nanosim-autonomous-worker`
4. Paste the key above
5. **Check "Allow write access"** ✓
6. Click "Add key"

### 3. Service Status

Check if the worker is running:
```bash
# View service status
supervisorctl status nanosim-worker

# View logs
tail -f autonomous_logs/worker_*.log

# Restart service
supervisorctl restart nanosim-worker
```

## How It Works

### Experiment Cycle (Every 4 Hours)

```
START CYCLE
  ↓
Run 3 simulations (nanothrones, colony, island)
  ↓
Analyze results (extinction events, mass deaths, emergence)
  ↓
Identify top 5 improvements
  ↓
Generate report (markdown summary)
  ↓
Queue email (to cris@thelifesim.com)
  ↓
Commit & push to GitHub
  ↓
Sleep 4 hours
  ↓
REPEAT
```

### Output Structure

```
nanosim/
├── autonomous_logs/
│   ├── worker_YYYYMMDD_HHMMSS.log       # Worker activity log
│   ├── report_cycle1.md                  # Experiment reports
│   ├── report_cycle2.md
│   └── email_outbox.json                 # Queued emails
├── autonomous_experiments/
│   ├── cycle1_nanothrones.json           # Raw experiment data
│   ├── cycle1_colony.json
│   └── cycle1_island.json
```

## Configuration

Edit `autonomous_worker.py` to customize:

- `WORK_CYCLE_HOURS = 4` — How often to run experiments
- Scenarios list — Which scenarios to test
- Agent/tick counts — Simulation parameters
- Model selection — Groq vs OpenRouter

## Monitoring

The worker logs everything to `autonomous_logs/`. Each cycle generates:

1. **Worker log** — Detailed execution trace
2. **Report markdown** — Human-readable summary
3. **Experiment JSONs** — Raw simulation results
4. **Email queue** — Reports to be sent

## Roadmap

Future enhancements the autonomous worker will implement:

- [ ] Dynamic harshness adjustment to prevent extinction
- [ ] Spatial awareness (coordinate graph for agents)
- [ ] Inter-agent trade & negotiation protocols
- [ ] Reproducible benchmark suite
- [ ] LLM-driven code improvements (self-modifying)
- [ ] Multi-model A/B testing

## Maintenance

The worker is designed to run indefinitely. If it stops:

```bash
# Check service status
supervisorctl status nanosim-worker

# View recent errors
tail -100 autonomous_logs/worker_*.log | grep ERROR

# Restart
supervisorctl restart nanosim-worker
```

## Cost Estimates

Using Groq (free tier):
- ~3 simulations per cycle
- ~$0.10-0.30 per cycle
- ~$1.80-7.20 per day (6 cycles)

The worker will automatically throttle if rate limits are hit.

---

**Status:** ✅ Deployed and running 24/7

**Last updated:** 2026-04-20
