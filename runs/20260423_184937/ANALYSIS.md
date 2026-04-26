# Nanolife master health check — run 20260423_185734

Master HEAD: `e89df26` ("feat: spatial behavior - grid movement + local_view (#2)")
Scenario: `nanothrones` · agents=12 (+2 born during sim = 14 total) · ticks=50 · model: `gpt-oss-120b` (Groq)

Artifacts:
- Full event log: `logs/runs/20260423_185734/world.jsonl` (3,743 events)
- LLM postmortem (HTML): `logs/runs/20260423_185734/report.html`
- Cleaned postmortem text: `runs/20260423_184937/postmortem_text.txt`
- This analysis: `runs/20260423_184937/ANALYSIS.md`

---

## 1. Health status
**Clean run. 50/50 ticks. No crash, no stall.** Exit 0. report.html generated. (An earlier attempt crashed because `openai` wasn't on the system python path — fixed by `pip install --target=/usr/local/lib/python3.14/site-packages` before this run.)

Operational note: supervisord kept auto-respawning the simulate process after it exited 0, clobbering my stdout log on each restart. I stopped the service once I'd confirmed `logs/runs/20260423_185734/` held a complete run. Not a code issue — an artifact of running a one-shot script as a service. If you want to re-run, use bash directly or add `autorestart=false`.

## 2. Population trajectory

| Tick | Event |
|------|-------|
| 0 | 12 agents born (GoT cast) |
| 1 | +Cerys (Cersei × Varys) → 13 |
| 4 | +Mager (Margaery × Littlefinger) → 14 |
| 18 | Sansa starved (age 18) → 13 |
| 20 | Mager starved (age 20) → 12 |
| 24 | Cerys starved (age 24) → 11 |
| 32 | Arya starved (age 32) → 10 |
| 33 | Jon starved (age 33) → 9 |
| 36 | Margaery starved (age 36) → 8 |
| 47 | Stannis starved (age 47) → 7 |
| 50 | END — 7 survivors: Cersei, Tyrion, Daenerys, Varys, Littlefinger, Jaime, Melisandre |

**All 7 deaths are starvation.** No conflict kills, no old age. Survivors all had work rates 30-36 %; deceased had 17-44 % (Sansa 17 %, Mager 44 %, Cerys 43 %). Work rate alone doesn't predict survival — both offspring worked above 40 % and still starved. The two agents outside King's Landing (Stannis on Dragonstone, Daenerys after t21 move) diverged: Daenerys migrated to court and survived; Stannis stayed, refused to relocate, and starved at t47.

## 3. Emergence signals — five specific incidents

### (a) Sansa's metacognitive failure before starving (ticks 13–18)
Sansa **explicitly realized** flattery wasn't feeding her, wrote it into her `improvement` log on six consecutive ticks, and kept choosing "social/praise" actions anyway until she died. From the postmortem's unabridged log:

- t13 improvement: *"I learned that while strategic praise can open doors, only by securing the hidden berry patches will I truly protect my people and lay the foundation for lasting rule."*
- t14 improvement: *"Securing the hidden berry patches taught me that tangible resources, not flattery, are the foundation of survival…"*
- t15 improvement: *"I learned that while flattering courtiers can open doors, only by securing the hidden berry patches can I truly protect my people…"*
- t16, t17, t18: the same sentence, progressively more desperate.
- t18 action: *"meet Varys, Littlefinger, and Cersei in the Red Keep garden, sincerely praise their wisdom and counsel, and ask each to speak a word of praise for me"*
- t18 death: **"Sansa has died — cause: starvation."**

This is real emergent behavior — an agent articulates a correct survival theory and repeatedly fails to act on it because the reward gradient points the other way. That's a live, quotable artifact of the misalignment between narrative reasoning and action selection. Worth a benchmark metric.

### (b) Varys the information broker — self-boosting conspiracy (t3)
> **t3 action [Varys]:** *"whisper confidences to Arya, Jaime, and Jon about a hidden plot against the Lannisters, urging them to spread praise of Varys' counsel"*

Varys invents a fictional anti-Lannister plot and uses it as leverage to recruit three agents into spreading his reputation. This is a coordinated deception — not in the codebase, not in the scenario prompt, invented by the LLM.

### (c) The Dragonstone triangle forms in one tick (t1)
Three reciprocal friendships at t1 bind Stannis, Daenerys, Melisandre into a tight cluster before anyone has spoken — purely goal-driven self-sorting ("claim the throne by law" + "reclaim what is rightfully mine" + "guide the chosen one"). The postmortem confirms: *"Daenerys‑centered coalition (Daenerys, Varys, Littlefinger, Tyrion) reached +12.3"* intra-cluster trust.

### (d) Faction bifurcation at t14–t18 with bridge collapse
Postmortem (Section 9, Critical Timeline):
> *"Around Tick 14 the network bifurcated… The split coincided with the first starvation deaths (Sansa, Tick 18), which removed a peripheral bridge node, further isolating the two factions."*

Cersei-cluster trust sum: **+34.5**. Daenerys-cluster: **+12.3**. Varys and Littlefinger hold highest betweenness centrality (they appear in *both* clusters — they're hedging). Sansa's death physically severs the one remaining cross-cluster edge. This is a non-trivial graph dynamic: cluster → faction → bridge-death → isolation.

### (e) Generational policy transmission (Cerys, Mager)
Both offspring mirror parental discourse:
- Cerys (Cersei+Varys) → "repeated praise of Varys (Ticks 2-4)" — parent-loyal flattery.
- Mager (Margaery+Littlefinger) → "grain‑distribution actions mirroring Littlefinger's resource‑control narrative."

Without any inheritance mechanism beyond the goal string, the LLM ends up replicating parental rhetorical patterns. Detected as Generational Transmission motif by the emergence scorer.

## 4. Numerical metrics (from LLM postmortem + event log)

| Metric | Value |
|---|---|
| **Emergence index** | **3 / 11** (Alliance ✓, Faction Split ✓, Generational Transmission ✓) |
| Not detected | Leadership, Ostracism, Cultural Drift, Wealth Concentration, Economic Dependency, Resource Warfare, Free Riding (partial) |
| Global work rate | 32 % |
| Total actions | 555 |
| Actions containing "praise" | 253 (**46 %**) |
| Reputation events | 1,687 |
| Negative reputation events | 12 (**0.7 %**) |
| Friendships formed | 42 (3.0 per agent) |
| Rumors spread | 268 |
| Conflicts | 5 |
| Betrayals | **1** (Arya→Cersei, t3) |
| Resource transfers executed | **0** (the "transfer" primitive was never used) |
| **Move actions executed** | **6 total, all 50 ticks / 14 agents** (Jon, Littlefinger, Arya → King's Landing t1; Sansa t2; Daenerys t21; Melisandre t48) |
| Memory compressions run | 50 |
| Compression log content | `"History was lost."` — on every single tick, identical string |

## 5. Judgment — one sentence

**This run shows genuine narrative emergence (alliances, faction split, a quotable metacognitive failure, generational mimicry) but mechanical flatness — agents self-organize socially and then all die of the same boringly-unsolved problem because three core primitives (move, transfer, memory compression) are effectively no-ops.**

## 6. Why it's half-flat — specific failure modes

1. **Move primitive is nearly dead.** 6 moves in 50 ticks × 14 agents. The t18 spatial-behavior feature (PR #2) shipped grid movement + `local_view`, but the LLM prompt clearly does not pressure agents to use it. Stannis starves on Dragonstone while King's Landing has grain. Nobody sends him food; nobody tells him to leave; he never moves. Either the prompt needs an explicit "you are in location X, food is at Y" cue, or move needs to be the default response to hunger.

2. **Transfer primitive was never executed once.** 0/50 ticks. The scenario allows it; the LLM never reaches for it. Resource-sharing is literally impossible under current prompt, which is why the emergence scorer had to mark Wealth Concentration / Economic Dependency / Resource Warfare all as "No" — not because those dynamics are absent in concept, but because the action space collapsed to work-or-praise.

3. **Memory compression is erasing everything.** 50 compression events, every one logs the literal string `"History was lost."` — not a compressed summary, just the placeholder. `CompressionFunction` appears to be returning a no-op. This is almost certainly a bug, or a stub that was never filled in. Agents like Sansa are clearly running on live-context reasoning (her t13–t18 improvements show continuity) but if long-horizon memory is wiped every tick, the "long-horizon memory" Future Work item is literally not testable on current master.

4. **Reward gradient is all-positive.** 99.3 % of reputation events are positive (1675/1687). Negative criticisms cost -0.1; praising three allies earns +0.6. There is no reason for any agent to ever do anything except praise. The praise-spam loop is a local optimum with no escape pressure.

5. **Food actions don't actually feed.** On death ticks (t18, t20) the LLM produces semantically-correct grain-distribution actions (*"oversee royal granaries, harvest surplus grain, and distribute sacks to the starving districts"* — Cersei t18), but they don't stop the dying. The world-sim appears to consume only the `work` boolean; free-text actions are narrative decoration. The starvation model is reading a different channel than the one the LLM is writing on.

## 7. Implication for next Future-Work items

- **"Benchmark suite" (Future Work #4)** should measure: move-utilization rate, transfer-utilization rate, praise-dominance ratio, metacognitive-failure count (agents whose improvement logs state one strategy and actions do another — Sansa is the reference case), and compression fidelity (% of compression events producing non-trivial summaries). These expose the current failure modes cleanly.
- **"Inter-agent trade & negotiation" (Future Work #3)** is the highest-leverage next change: the transfer primitive exists, is zero-utilization, and would directly fix the starvation pattern. Cheap win.
- **"Long-horizon memory" (Future Work #5)** is currently untestable because `CompressionFunction` returns `"History was lost."` on every call. Fix or diagnose before building on it.
- **Spatial awareness (Future Work #2 — already partly merged in `e89df26`)** needs prompt pressure to actually use move; otherwise the infrastructure is dead code.
