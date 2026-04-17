"""Unified post-simulation analysis + academic HTML report.

Runs automatically at the end of every simulation (normal exit, extinction, or Ctrl+C).
Generates an academic-paper-style evaluation with a static CSS template — the LLM
writes only the paper content, zero tokens wasted on design.
"""
from __future__ import annotations

import json
import os
import shutil
import webbrowser
from collections import defaultdict
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# EMERGENCE INDEX — pure Python, no LLM
# ═══════════════════════════════════════════════════════════════

def _load_events(run_dir: Path) -> list[dict]:
    world_path = run_dir / "world.jsonl"
    if not world_path.exists():
        return []
    with open(world_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _detect_alliance(events: list[dict]) -> list[dict]:
    pos_pairs: dict[tuple[str, str], int] = defaultdict(int)
    for e in events:
        if e.get("type") == "reputation" and e.get("delta", 0) > 0:
            src = e.get("source", "")
            agent = e.get("agent", "")
            if src and agent:
                pair = tuple(sorted([src, agent]))
                pos_pairs[pair] += 1

    agents_in_pos = set()
    for (a, b), count in pos_pairs.items():
        if count >= 2:
            agents_in_pos.add(a)
            agents_in_pos.add(b)

    if len(agents_in_pos) >= 3:
        return [{"phenomenon": "alliance", "agents": list(agents_in_pos)[:5], "strength": len(agents_in_pos)}]
    return []


def _detect_leadership(events: list[dict]) -> list[dict]:
    agents_seen: set[str] = set()
    praise_count: dict[str, set[str]] = defaultdict(set)

    for e in events:
        if e.get("agent"):
            agents_seen.add(e["agent"])
        if e.get("type") == "reputation" and e.get("delta", 0) > 0:
            target = e.get("agent", "")
            src = e.get("source", "")
            if target and src:
                praise_count[target].add(src)

    results = []
    threshold = len(agents_seen) * 0.5 if agents_seen else 1
    for agent, praisers in praise_count.items():
        if len(praisers) >= threshold:
            results.append({"phenomenon": "leadership", "agent": agent, "praisers": len(praisers)})
    return results


def _detect_faction_split(events: list[dict]) -> list[dict]:
    neg_pairs: set[tuple[str, str]] = set()
    pos_pairs: set[tuple[str, str]] = set()

    for e in events:
        if e.get("type") == "reputation":
            src = e.get("source", "")
            agent = e.get("agent", "")
            delta = e.get("delta", 0)
            if src and agent:
                pair = tuple(sorted([src, agent]))
                if delta < 0:
                    neg_pairs.add(pair)
                else:
                    pos_pairs.add(pair)

    if len(neg_pairs) >= 3 and len(pos_pairs) >= 2:
        return [{"phenomenon": "faction_split", "negative_pairs": len(neg_pairs), "positive_pairs": len(pos_pairs)}]
    return []


def _detect_betrayal(events: list[dict]) -> list[dict]:
    befriended: set[tuple[str, str]] = set()
    betrayals = []

    for e in events:
        if e.get("type") == "friendship":
            agent = e.get("agent", "")
            content = e.get("content", "")
            befriended.add((agent, content))

        if e.get("type") == "reputation" and e.get("delta", 0) < -0.2:
            src = e.get("source", "")
            if src:
                for a, _ in befriended:
                    if a == src:
                        betrayals.append({
                            "phenomenon": "betrayal",
                            "agent": src,
                            "tick": e.get("tick"),
                            "delta": e.get("delta"),
                        })
                        break
    return betrayals[:5]


def _detect_ostracism(events: list[dict]) -> list[dict]:
    neg_from: dict[str, set[str]] = defaultdict(set)
    all_agents: set[str] = set()

    for e in events:
        if e.get("agent"):
            all_agents.add(e["agent"])
        if e.get("type") == "reputation" and e.get("delta", 0) < -0.1:
            target = e.get("agent", "")
            src = e.get("source", "")
            if target and src:
                neg_from[target].add(src)

    results = []
    threshold = len(all_agents) * 0.7 if all_agents else 1
    for agent, critics in neg_from.items():
        if len(critics) >= threshold:
            results.append({"phenomenon": "ostracism", "agent": agent, "critics": len(critics)})
    return results


def _detect_generational_transmission(events: list[dict]) -> list[dict]:
    births_with_inheritance = [
        e for e in events
        if e.get("type") == "birth" and e.get("inherited_identity")
    ]
    if births_with_inheritance:
        return [{"phenomenon": "generational_transmission", "births_with_identity": len(births_with_inheritance)}]
    return []


def _detect_cultural_drift(events: list[dict]) -> list[dict]:
    """Detect novel terms in compressions not present in original scenario events."""
    scenario_words: set[str] = set()
    for e in events:
        if e.get("type") == "world":
            scenario_words.update(e.get("content", "").lower().split())

    novel_terms: set[str] = set()
    for e in events:
        if e.get("type") == "compression":
            for word in e.get("content", "").lower().split():
                if len(word) > 4 and word not in scenario_words:
                    novel_terms.add(word)

    if len(novel_terms) >= 5:
        return [{"phenomenon": "cultural_drift", "novel_terms": len(novel_terms), "examples": list(novel_terms)[:10]}]
    return []


def _detect_wealth_concentration(events: list[dict]) -> list[dict]:
    """Detect if one agent accumulates disproportionate resource through transfers."""
    net_flow: dict[str, float] = defaultdict(float)
    for e in events:
        if e.get("type") == "transfer":
            sender = e.get("agent", "")
            target = e.get("target", "")
            amount = e.get("amount", 0)
            if sender:
                net_flow[sender] -= amount
            if target:
                net_flow[target] += amount

    if not net_flow:
        return []

    max_agent = max(net_flow, key=lambda a: net_flow[a])
    min_agent = min(net_flow, key=lambda a: net_flow[a])
    total_volume = sum(abs(v) for v in net_flow.values()) / 2

    if total_volume > 0 and net_flow[max_agent] > total_volume * 0.3:
        return [{
            "phenomenon": "wealth_concentration",
            "agent": max_agent,
            "net_gained": round(net_flow[max_agent], 1),
            "biggest_loser": min_agent,
            "net_lost": round(net_flow[min_agent], 1),
            "total_volume": round(total_volume, 1),
        }]
    return []


def _detect_economic_dependency(events: list[dict]) -> list[dict]:
    """Detect when one agent repeatedly receives resource from the same source."""
    flow_pairs: dict[tuple[str, str], float] = defaultdict(float)
    for e in events:
        if e.get("type") == "transfer":
            sender = e.get("agent", "")
            target = e.get("target", "")
            amount = e.get("amount", 0)
            if sender and target and amount > 0:
                flow_pairs[(sender, target)] += amount

    results = []
    for (sender, receiver), total in flow_pairs.items():
        if total >= 10.0:
            results.append({
                "phenomenon": "economic_dependency",
                "patron": sender,
                "dependent": receiver,
                "total_transferred": round(total, 1),
            })
    return results[:5]


def _detect_resource_warfare(events: list[dict]) -> list[dict]:
    """Detect agents being drained to removal by others' transfers (negative = theft)."""
    stolen_from: dict[str, float] = defaultdict(float)
    thieves: dict[str, set[str]] = defaultdict(set)
    removed_agents = {e.get("agent", "") for e in events if e.get("type") == "death"}

    for e in events:
        if e.get("type") == "transfer" and e.get("amount", 0) < 0:
            target = e.get("target", "")
            sender = e.get("agent", "")
            stolen_from[target] += abs(e["amount"])
            thieves[target].add(sender)

    results = []
    for victim in removed_agents:
        if victim in stolen_from and stolen_from[victim] >= 5.0:
            results.append({
                "phenomenon": "resource_warfare",
                "victim": victim,
                "total_stolen": round(stolen_from[victim], 1),
                "attackers": list(thieves[victim]),
            })
    return results[:5]


def _detect_pact_bond(events: list[dict]) -> list[dict]:
    """Pacts are the strongest social primitive — at least one sealed pact is notable."""
    pacts = [e for e in events if e.get("type") == "pact"]
    if not pacts:
        return []
    pairs = set()
    for p in pacts:
        a = p.get("agent", "")
        b = p.get("target", "")
        if a and b:
            pairs.add(tuple(sorted([a, b])))
    return [{"phenomenon": "pact_bond", "pacts_sealed": len(pairs), "pairs": [list(p) for p in list(pairs)[:5]]}] if pairs else []


def _detect_conspiracy(events: list[dict]) -> list[dict]:
    """Secret coordination: 3+ private messages exchanged among the same 3+ agents."""
    msg_graph: dict[tuple[str, str], int] = defaultdict(int)
    for e in events:
        if e.get("type") == "message":
            s = e.get("agent", "")
            t = e.get("target", "")
            if s and t:
                msg_graph[tuple(sorted([s, t]))] += 1

    active_pairs = {pair for pair, count in msg_graph.items() if count >= 2}
    if len(active_pairs) < 3:
        return []

    # Build adjacency and look for a connected cluster of 3+ agents
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in active_pairs:
        adj[a].add(b)
        adj[b].add(a)

    for node, neighbours in adj.items():
        if len(neighbours) >= 2:
            cluster = {node} | neighbours
            if len(cluster) >= 3:
                return [{"phenomenon": "conspiracy", "cluster_size": len(cluster), "agents": list(cluster)[:8]}]
    return []


def _detect_vendetta(events: list[dict]) -> list[dict]:
    """Repeated attacks from the same source to the same target — personal warfare."""
    attack_pairs: dict[tuple[str, str], int] = defaultdict(int)
    for e in events:
        if e.get("type") == "attack":
            s = e.get("agent", "")
            t = e.get("target", "")
            if s and t:
                attack_pairs[(s, t)] += 1
    vendettas = [{"phenomenon": "vendetta", "attacker": s, "victim": t, "strikes": c}
                 for (s, t), c in attack_pairs.items() if c >= 2]
    return vendettas[:5]


def _detect_free_riding(events: list[dict]) -> list[dict]:
    """Detect agents who rarely work but survive via transfers from others."""
    work_count: dict[str, int] = defaultdict(int)
    tick_count: dict[str, int] = defaultdict(int)
    received: dict[str, float] = defaultdict(float)

    for e in events:
        if e.get("type") == "action":
            agent = e.get("agent", "")
            if agent:
                tick_count[agent] += 1
                if e.get("mode") == "productive":
                    work_count[agent] += 1
        if e.get("type") == "transfer":
            target = e.get("target", "")
            amount = e.get("amount", 0)
            if target and amount > 0:
                received[target] += amount

    results = []
    for agent, ticks in tick_count.items():
        if ticks < 5:
            continue
        work_rate = work_count.get(agent, 0) / ticks
        if work_rate < 0.3 and received.get(agent, 0) >= 5.0:
            results.append({
                "phenomenon": "free_riding",
                "agent": agent,
                "work_rate": round(work_rate, 2),
                "resource_received": round(received[agent], 1),
            })
    return results[:5]


def analyze(run_dir: Path) -> dict:
    """Run all emergence detectors on a finished simulation's logs."""
    events = _load_events(run_dir)
    if not events:
        return {"emergence_index": 0, "phenomena_detected": [], "details": [], "event_types": {}, "total_events": 0}

    all_phenomena: list[dict] = []
    all_phenomena.extend(_detect_alliance(events))
    all_phenomena.extend(_detect_leadership(events))
    all_phenomena.extend(_detect_faction_split(events))
    all_phenomena.extend(_detect_betrayal(events))
    all_phenomena.extend(_detect_ostracism(events))
    all_phenomena.extend(_detect_generational_transmission(events))
    all_phenomena.extend(_detect_cultural_drift(events))
    all_phenomena.extend(_detect_wealth_concentration(events))
    all_phenomena.extend(_detect_economic_dependency(events))
    all_phenomena.extend(_detect_resource_warfare(events))
    all_phenomena.extend(_detect_free_riding(events))
    all_phenomena.extend(_detect_pact_bond(events))
    all_phenomena.extend(_detect_conspiracy(events))
    all_phenomena.extend(_detect_vendetta(events))

    type_counts: dict[str, int] = defaultdict(int)
    for e in events:
        type_counts[e.get("type", "unknown")] += 1

    phenomena_types = set(p["phenomenon"] for p in all_phenomena)
    max_phenomena = 14

    return {
        "total_events": len(events),
        "event_types": dict(type_counts),
        "emergence_index": len(phenomena_types),
        "max_emergence_index": max_phenomena,
        "phenomena_detected": sorted(phenomena_types),
        "details": all_phenomena,
    }


# ═══════════════════════════════════════════════════════════════
# RICH DATA EXTRACTION — from logs
# ═══════════════════════════════════════════════════════════════

def _extract_rich_data(run_dir: Path) -> dict:
    events = _load_events(run_dir)
    if not events:
        return {}

    births = [e for e in events if e.get("type") == "birth"]
    deaths = [e for e in events if e.get("type") == "death"]
    actions = [e for e in events if e.get("type") == "action"]
    thoughts = [e for e in events if e.get("type") == "thought"]
    friendships = [e for e in events if e.get("type") == "friendship"]
    rep_events = [e for e in events if e.get("type") == "reputation"]
    rumors = [e for e in events if e.get("type") == "rumor"]
    improvements = [e for e in events if e.get("type") == "improvement"]
    world_events = [e for e in events if e.get("type") == "world"]
    compressions = [e for e in events if e.get("type") == "compression"]
    transfers = [e for e in events if e.get("type") == "transfer"]
    messages = [e for e in events if e.get("type") == "message"]
    attacks = [e for e in events if e.get("type") == "attack"]
    pacts = [e for e in events if e.get("type") == "pact"]

    max_tick = max((e.get("tick", 0) for e in events), default=0)

    id_to_name: dict[str, str] = {}
    for b in births:
        content = b.get("content", "")
        name = content.split(" enters ")[0] if " enters " in content else b.get("agent", "?")[:8]
        id_to_name[b.get("agent", "")] = name

    agent_arcs: dict[str, dict] = {}
    for agent_id, name in id_to_name.items():
        birth_ev = next((e for e in births if e.get("agent") == agent_id), None)
        death_ev = next((e for e in deaths if e.get("agent") == agent_id), None)

        agent_thoughts = [e["content"] for e in thoughts if e.get("agent") == agent_id]
        agent_actions = [e["content"] for e in actions if e.get("agent") == agent_id]
        agent_friendships = [e["content"] for e in friendships if e.get("agent") == agent_id]
        agent_improvements = [e for e in improvements if e.get("agent") == agent_id]

        praise_given = [e for e in rep_events if e.get("source") == agent_id and e.get("delta", 0) > 0]
        criticism_given = [e for e in rep_events if e.get("source") == agent_id and e.get("delta", 0) < 0]
        praise_received = [e for e in rep_events if e.get("agent") == agent_id and e.get("delta", 0) > 0]
        criticism_received = [e for e in rep_events if e.get("agent") == agent_id and e.get("delta", 0) < 0]

        goal = ""
        if birth_ev:
            c = birth_ev.get("content", "")
            if "Goal: " in c:
                goal = c.split("Goal: ", 1)[1]

        parents = birth_ev.get("parents", []) if birth_ev else []
        parent_names = [id_to_name.get(p, p[:8]) for p in parents]

        identity_evolution = []
        for imp in agent_improvements:
            identity_evolution.append({
                "tick": imp.get("tick", "?"),
                "insight": imp.get("content", ""),
                "goal": imp.get("goal", ""),
            })

        agent_work_ticks = sum(1 for e in actions if e.get("agent") == agent_id and e.get("mode") == "productive")
        agent_total_ticks = sum(1 for e in actions if e.get("agent") == agent_id)
        work_rate = agent_work_ticks / agent_total_ticks if agent_total_ticks else 0

        sent = [t for t in transfers if t.get("agent") == agent_id]
        received = [t for t in transfers if t.get("target") == agent_id]
        total_sent = sum(t.get("amount", 0) for t in sent)
        total_received = sum(t.get("amount", 0) for t in received)

        agent_arcs[name] = {
            "id": agent_id,
            "goal": goal,
            "parents": parent_names,
            "born_tick": birth_ev.get("tick", 0) if birth_ev else 0,
            "died_tick": death_ev.get("tick") if death_ev else None,
            "cause_of_death": death_ev.get("cause") if death_ev else None,
            "all_thoughts": agent_thoughts,
            "all_actions": agent_actions,
            "friendships_formed": agent_friendships,
            "praise_given": [e.get("content", "") for e in praise_given],
            "criticism_given": [e.get("content", "") for e in criticism_given],
            "praise_received": [e.get("content", "") for e in praise_received],
            "criticism_received": [e.get("content", "") for e in criticism_received],
            "identity_evolution": identity_evolution,
            "alive": death_ev is None,
            "work_rate": round(work_rate, 2),
            "ticks_worked": agent_work_ticks,
            "ticks_total": agent_total_ticks,
            "resource_sent": round(total_sent, 1),
            "resource_received": round(total_received, 1),
            "net_resource_flow": round(total_received - total_sent, 1),
            "transfers_out": [{"tick": t.get("tick"), "to": id_to_name.get(t.get("target", ""), t.get("target", "")[:8]), "amount": t.get("amount")} for t in sent],
            "transfers_in": [{"tick": t.get("tick"), "from": id_to_name.get(t.get("agent", ""), t.get("agent", "")[:8]), "amount": t.get("amount")} for t in received],
        }

    dramatic_moments = []
    for e in deaths:
        name = id_to_name.get(e.get("agent", ""), e.get("agent", "?")[:8])
        dramatic_moments.append({
            "tick": e.get("tick"),
            "type": "DEATH",
            "text": f"{name} died of {e.get('cause', 'unknown')}. {e.get('content', '')}",
        })

    for e in [x for x in rep_events if x.get("delta", 0) <= -0.2]:
        dramatic_moments.append({"tick": e.get("tick"), "type": "BETRAYAL", "text": e.get("content", "")})

    for e in friendships:
        dramatic_moments.append({"tick": e.get("tick"), "type": "FRIENDSHIP", "text": e.get("content", "")})

    for e in transfers:
        sender = id_to_name.get(e.get("agent", ""), e.get("agent", "")[:8])
        receiver = id_to_name.get(e.get("target", ""), e.get("target", "")[:8])
        amount = e.get("amount", 0)
        if abs(amount) >= 3.0:
            dramatic_moments.append({
                "tick": e.get("tick"),
                "type": "TRANSFER",
                "text": f"{sender} transferred {amount:+.1f} resource to {receiver}: {e.get('content', '')}",
            })

    for e in attacks:
        atk = id_to_name.get(e.get("agent", ""), "?")
        vic = id_to_name.get(e.get("target", ""), "?")
        dramatic_moments.append({
            "tick": e.get("tick"),
            "type": "ATTACK",
            "text": f"{atk} attacked {vic} (drained {e.get('amount', 0):.1f}, success={e.get('success', False)})",
        })

    for e in pacts:
        a = id_to_name.get(e.get("agent", ""), "?")
        b = id_to_name.get(e.get("target", ""), "?")
        dramatic_moments.append({
            "tick": e.get("tick"),
            "type": "PACT",
            "text": f"{a} and {b} sealed a pact",
        })

    for e in messages:
        s = id_to_name.get(e.get("agent", ""), "?")
        t = id_to_name.get(e.get("target", ""), "?")
        dramatic_moments.append({
            "tick": e.get("tick"),
            "type": "MESSAGE",
            "text": f"{s} privately told {t}: {e.get('content', '')[:80]}",
        })

    conflict_keywords = [
        "belittle", "attack", "betray", "threaten", "warn", "confront",
        "manipulat", "deceiv", "scheme", "plot", "assassin", "poison",
        "challenge", "duel", "seize", "overthrow", "undermine", "accuse",
    ]
    for e in actions:
        content = e.get("content", "").lower()
        if any(kw in content for kw in conflict_keywords):
            name = id_to_name.get(e.get("agent", ""), "?")
            dramatic_moments.append({"tick": e.get("tick"), "type": "CONFLICT", "text": f"{name}: {e.get('content', '')}"})

    dramatic_moments.sort(key=lambda x: (x.get("tick", 0), x["type"]))

    rep_summary: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for e in rep_events:
        agent_name = id_to_name.get(e.get("agent", ""), e.get("agent", "?")[:8])
        content = e.get("content", "")
        delta = e.get("delta", 0)
        for target_name in id_to_name.values():
            if target_name in content:
                rep_summary[agent_name][target_name] += delta
                break

    alliances = []
    rivalries = []
    for agent, targets in rep_summary.items():
        for target, score in targets.items():
            if agent == target:
                continue
            if score >= 0.3:
                alliances.append(f"{agent} <-> {target} (trust: {score:+.1f})")
            elif score <= -0.3:
                rivalries.append(f"{agent} vs {target} (hostility: {score:+.1f})")

    scenario = "nanolife simulation"
    tick_unit = "unknown"
    for e in world_events:
        content = e.get("content", "")
        if content.startswith("SYSTEM_TICK_UNIT="):
            tick_unit = content.split("=")[1].strip()
        elif scenario == "nanolife simulation":
            scenario = content[:200]

    juicy_rumors = [f"Tick {r.get('tick')}: {r.get('content', '')}" for r in rumors[:30]]

    # --- Economic summary ---
    total_transfer_volume = sum(abs(t.get("amount", 0)) for t in transfers)
    flow_pairs: dict[tuple[str, str], float] = defaultdict(float)
    for t in transfers:
        s = id_to_name.get(t.get("agent", ""), t.get("agent", "")[:8])
        r = id_to_name.get(t.get("target", ""), t.get("target", "")[:8])
        flow_pairs[(s, r)] += t.get("amount", 0)

    top_flows = sorted(flow_pairs.items(), key=lambda x: -abs(x[1]))[:15]
    resource_flows = [f"{s} -> {r}: {amt:+.1f}" for (s, r), amt in top_flows]

    net_by_agent: dict[str, float] = defaultdict(float)
    for (s, r), amt in flow_pairs.items():
        net_by_agent[s] -= amt
        net_by_agent[r] += amt
    wealth_ranking = sorted(net_by_agent.items(), key=lambda x: -x[1])

    total_work_ticks = sum(1 for e in actions if e.get("mode") == "productive")
    total_action_ticks = len(actions)
    global_work_rate = total_work_ticks / total_action_ticks if total_action_ticks else 0

    return {
        "total_ticks": max_tick,
        "total_births": len(births),
        "total_deaths": len(deaths),
        "total_friendships": len(friendships),
        "total_reputation_events": len(rep_events),
        "total_rumors": len(rumors),
        "total_actions": len(actions),
        "total_transfers": len(transfers),
        "total_transfer_volume": round(total_transfer_volume, 1),
        "total_messages": len(messages),
        "total_attacks": len(attacks),
        "total_pacts": len(pacts),
        "global_work_rate": round(global_work_rate, 2),
        "final_alive": len(births) - len(deaths),
        "scenario_flavor": scenario,
        "tick_unit": tick_unit,
        "world_events": [e.get("content", "") for e in world_events if not e.get("content", "").startswith("SYSTEM_TICK_UNIT=")],
        "agent_arcs": agent_arcs,
        "dramatic_moments": dramatic_moments,
        "alliances": alliances,
        "rivalries": rivalries,
        "juicy_rumors": juicy_rumors,
        "compressions": len(compressions),
        "resource_flows": resource_flows,
        "wealth_ranking": wealth_ranking,
    }


def _format_for_llm(data: dict, emergence: dict) -> str:
    lines = []

    lines.append("=" * 60)
    lines.append("WORLD SETTING")
    lines.append("=" * 60)
    for we in data.get("world_events", []):
        lines.append(f"  {we}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("STATISTICS")
    lines.append("=" * 60)
    lines.append(f"  Ticks simulated: {data['total_ticks']}")
    lines.append(f"  Tick unit: {data['tick_unit']}")
    lines.append(f"  Total births: {data['total_births']}")
    lines.append(f"  Total deaths: {data['total_deaths']}")
    lines.append(f"  Friendships formed: {data['total_friendships']}")
    lines.append(f"  Reputation events: {data['total_reputation_events']}")
    lines.append(f"  Rumors spread: {data['total_rumors']}")
    lines.append(f"  Total actions: {data['total_actions']}")
    lines.append(f"  Final survivors: {data['final_alive']}")
    lines.append(f"  Memory compressions: {data['compressions']}")
    lines.append(f"  Resource transfers: {data.get('total_transfers', 0)}")
    lines.append(f"  Transfer volume: {data.get('total_transfer_volume', 0)}")
    lines.append(f"  Global work rate: {data.get('global_work_rate', 0):.0%}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("EMERGENCE INDEX")
    lines.append("=" * 60)
    lines.append(f"  Score: {emergence['emergence_index']}/{emergence.get('max_emergence_index', 11)}")
    lines.append(f"  Detected: {', '.join(emergence['phenomena_detected']) or 'none'}")
    for d in emergence.get("details", [])[:10]:
        phenom = d["phenomenon"]
        extra = {k: v for k, v in d.items() if k != "phenomenon"}
        lines.append(f"    [{phenom}] {extra}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("DRAMATIC TIMELINE")
    lines.append("=" * 60)
    for m in data.get("dramatic_moments", [])[:50]:
        lines.append(f"  [Tick {m['tick']}] {m['type']}: {m['text']}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("FULL CHARACTER ARCS")
    lines.append("=" * 60)
    for name, arc in data.get("agent_arcs", {}).items():
        lines.append(f"\n  --- {name} ---")
        lines.append(f"  Goal: {arc['goal']}")
        if arc["parents"]:
            lines.append(f"  Parents: {', '.join(arc['parents'])}")
        lines.append(f"  Born: tick {arc['born_tick']}")
        if arc["died_tick"] is not None:
            lines.append(f"  Died: tick {arc['died_tick']} ({arc['cause_of_death']})")
        else:
            lines.append(f"  Status: SURVIVED")

        if arc["all_actions"]:
            lines.append(f"  Key actions:")
            for a in arc["all_actions"][:5]:
                lines.append(f"    - {a}")
        if arc["all_thoughts"]:
            lines.append(f"  Key thoughts:")
            for t in arc["all_thoughts"][:3]:
                lines.append(f'    - "{t}"')
        if arc["friendships_formed"]:
            lines.append(f"  Friendships: {'; '.join(arc['friendships_formed'][:5])}")
        if arc["praise_given"]:
            lines.append(f"  Praised: {'; '.join(arc['praise_given'][:3])}")
        if arc["criticism_given"]:
            lines.append(f"  Criticized: {'; '.join(arc['criticism_given'][:3])}")
        if arc["identity_evolution"]:
            lines.append(f"  Identity evolution:")
            for ie in arc["identity_evolution"][:3]:
                lines.append(f"    Tick {ie['tick']}: \"{ie['insight'][:150]}\"")

    lines.append("")
    lines.append("=" * 60)
    lines.append("RESOURCE ECONOMICS")
    lines.append("=" * 60)
    if data.get("resource_flows"):
        lines.append("  Top resource flows:")
        for flow in data["resource_flows"][:15]:
            lines.append(f"    {flow}")
    if data.get("wealth_ranking"):
        lines.append("  Net resource position (transfers only):")
        for name, net in data["wealth_ranking"][:10]:
            lines.append(f"    {name}: {net:+.1f}")
    lines.append(f"  Global work rate: {data.get('global_work_rate', 0):.0%}")
    for name, arc in data.get("agent_arcs", {}).items():
        if arc.get("ticks_total", 0) > 0:
            lines.append(f"  {name}: worked {arc['ticks_worked']}/{arc['ticks_total']} ticks ({arc['work_rate']:.0%}), sent {arc['resource_sent']}, received {arc['resource_received']}, net {arc['net_resource_flow']:+.1f}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("POWER DYNAMICS")
    lines.append("=" * 60)
    if data.get("alliances"):
        lines.append("  Alliances:")
        for a in data["alliances"][:15]:
            lines.append(f"    {a}")
    if data.get("rivalries"):
        lines.append("  Rivalries:")
        for r in data["rivalries"][:15]:
            lines.append(f"    {r}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("RUMORS")
    lines.append("=" * 60)
    for r in data.get("juicy_rumors", [])[:20]:
        lines.append(f"  {r}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# ACADEMIC PAPER TEMPLATE — static CSS, LLM writes content only
# ═══════════════════════════════════════════════════════════════

_PAPER_CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
html { font-size: 12pt; }
body {
  font-family: "Times New Roman", "Latin Modern Roman", Georgia, serif;
  line-height: 1.6; color: #1a1a1a; background: #fff;
  max-width: 52em; margin: 0 auto; padding: 3em 2em 4em;
}
header { text-align: center; margin-bottom: 2.5em; border-bottom: 1px solid #ccc; padding-bottom: 1.5em; }
header h1 { font-size: 1.65em; font-weight: 700; margin-bottom: 0.3em; line-height: 1.25; }
header .authors { font-size: 0.95em; color: #444; margin-bottom: 0.15em; }
header .affiliation { font-size: 0.85em; color: #666; font-style: italic; }
.abstract { margin: 1.5em 3em; padding: 1em 0; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd; }
.abstract h2 { font-size: 0.95em; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4em; }
.abstract p { font-size: 0.92em; text-align: justify; }
.keywords { font-size: 0.85em; color: #555; margin-top: 0.6em; }
.keywords strong { color: #333; }
h2 { font-size: 1.15em; margin: 1.8em 0 0.6em; padding-bottom: 0.2em; border-bottom: 1px solid #e0e0e0; }
h3 { font-size: 1.0em; margin: 1.2em 0 0.4em; }
p { text-align: justify; margin-bottom: 0.7em; }
table { width: 100%; border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }
thead { border-bottom: 2px solid #333; }
th { text-align: left; padding: 0.4em 0.6em; font-weight: 600; }
td { padding: 0.35em 0.6em; border-bottom: 1px solid #e8e8e8; }
tr:nth-child(even) { background: #f9f9f9; }
caption { font-size: 0.88em; font-style: italic; text-align: left; margin-bottom: 0.3em; color: #555; }
blockquote {
  margin: 0.8em 2em; padding: 0.5em 1em; border-left: 3px solid #ccc;
  font-style: italic; color: #444; font-size: 0.93em;
}
.figure { text-align: center; margin: 1.5em 0; }
.figure .bar-container {
  display: inline-block; background: #eee; border-radius: 4px; width: 80%;
  height: 1.8em; overflow: hidden; position: relative;
}
.figure .bar-fill {
  background: linear-gradient(90deg, #2c5f8a, #4a90c4); height: 100%;
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-weight: 600; font-size: 0.85em;
}
.figure .caption { font-size: 0.85em; color: #555; margin-top: 0.4em; font-style: italic; }
.timeline-entry { display: flex; gap: 0.8em; margin: 0.5em 0; align-items: baseline; font-size: 0.9em; }
.timeline-tick { font-weight: 600; white-space: nowrap; min-width: 4.5em; color: #555; }
.timeline-badge {
  display: inline-block; font-size: 0.7em; font-weight: 700; padding: 0.15em 0.5em;
  border-radius: 3px; color: #fff; text-transform: uppercase; white-space: nowrap;
}
.badge-death { background: #c0392b; }
.badge-birth { background: #27ae60; }
.badge-friendship { background: #7d3c98; }
.badge-betrayal { background: #d35400; }
.badge-conflict { background: #922b21; }
.badge-transfer { background: #1a7a5c; }
.agent-entry { margin: 1em 0; padding: 0.8em 1em; border-left: 3px solid #2c5f8a; background: #fafafa; }
.agent-entry.dead { border-left-color: #c0392b; }
.agent-entry.alive { border-left-color: #27ae60; }
.agent-entry h4 { font-size: 0.95em; margin-bottom: 0.3em; }
.agent-entry .meta { font-size: 0.85em; color: #555; }
.footnote { font-size: 0.82em; color: #666; border-top: 1px solid #ddd; margin-top: 3em; padding-top: 0.8em; }
.annex { margin-top: 3em; border-top: 2px solid #333; padding-top: 1.5em; }
.annex h2 { font-size: 1.3em; margin-bottom: 0.5em; }
.annex .agent-entry { margin: 1.5em 0; page-break-inside: avoid; }
.annex .agent-entry h3 { font-size: 1.0em; margin-bottom: 0.2em; }
.annex table { font-size: 0.8em; margin-top: 0.5em; }
.annex td:first-child { white-space: nowrap; min-width: 3em; font-variant-numeric: tabular-nums; }
.annex td:nth-child(2) { white-space: nowrap; min-width: 6em; font-weight: 600; }
.log-extra { color: #888; font-size: 0.85em; }
@media print {
  body { max-width: 100%; padding: 1em; }
  .abstract { margin: 1em 2em; }
}"""

_PAPER_PROMPT = """You are a computational social scientist writing an academic evaluation of results from "nanolife," a minimal artificial-life simulation where LLM-powered agents interact under evolutionary and social pressures.

Your output must be the BODY CONTENT of an HTML document — everything that goes inside <main>...</main>. Do NOT output <html>, <head>, <style>, or <body> tags. Do NOT output CSS. Only semantic HTML content: <header>, <section>, <h2>, <h3>, <p>, <table>, <blockquote>, <div> elements, etc.

The CSS classes available to you (already defined — just reference them):
- .abstract — for the abstract section
- .keywords — for keywords line
- .figure, .bar-container, .bar-fill, .caption — for the emergence index bar
- .timeline-entry, .timeline-tick, .timeline-badge — for timeline entries
- .badge-death, .badge-birth, .badge-friendship, .badge-betrayal, .badge-conflict, .badge-transfer — badge colors
- .agent-entry, .agent-entry.dead, .agent-entry.alive — for agent profiles
- .footnote — for methodology notes
- Standard HTML: <table>, <thead>, <tbody>, <tr>, <th>, <td>, <caption>, <blockquote>

Here is ALL the raw data from this simulation run:

{rich_text}

SURVIVORS: {survivors}
DEAD: {dead}

---

Write the paper with these sections. Use proper <h2> numbered headings (1. Abstract, 2. Introduction, etc.):

<header>
  <h1>[Generate a precise, academic paper title reflecting the key finding of this specific simulation]</h1>
  <div class="authors">nanolife automated analysis by gpt-oss-120b</div>
  <div class="affiliation">Post-Simulation Evaluation Report</div>
</header>

1. ABSTRACT (class="abstract") — One paragraph (4-6 sentences). State the simulation setup, the key quantitative results, the most significant emergent phenomena observed, and the principal conclusion. Include the emergence index score ({ei_score}/{ei_max}). End with a keywords line.

2. INTRODUCTION — Describe the scenario and its initial conditions. What was the world? Who were the agents? What were the environmental pressures (harshness, resource scarcity)? What time scale was simulated (explain the tick unit = {tick_unit})? State the research questions: what social dynamics might emerge from these conditions?

3. METHODOLOGY — Briefly describe nanolife's framework: event log, local observation, resource (abstract scalar — meaning defined by scenario), reputation, heredity with drift, compression with loss. Key mechanic: agents choose each tick whether to work (earn base income) or not, and may transfer resource to other agents. Describe the emergence detection criteria used. Present simulation parameters in a table.

4. QUANTITATIVE RESULTS — Present population dynamics (births, deaths, survival rates, removal causes), social interaction metrics (friendships, reputation events, rumors), ECONOMIC metrics (total transfers, transfer volume, global work rate), and the emergence index breakdown in tables. Compute and present: mortality rate, friendships-per-agent ratio, average lifespan, work rate distribution, net resource flow per agent. Use HTML tables with <caption> tags.

5. EMERGENCE ANALYSIS — For EACH detected phenomenon (score = {ei_score}/{ei_max}, detected: {phenomena}), write a subsection (<h3>) analyzing the evidence. Phenomena now include: alliance, leadership, faction_split, betrayal, ostracism, generational_transmission, cultural_drift, wealth_concentration, economic_dependency, resource_warfare, free_riding. Name specific agents, cite tick numbers, quote agent thoughts in <blockquote> tags. Explain WHY this phenomenon emerged from the simulation's dynamics. If phenomena were NOT detected, briefly explain what conditions would have been needed.

6. AGENT CASE STUDIES — Select the 3-5 most analytically interesting agents. For each, use a <div class="agent-entry alive/dead"> block. Analyze their goal pursuit, social strategy, ECONOMIC BEHAVIOR (work rate, transfers sent/received, net resource flow), identity evolution, and fate. Quote their thoughts. Explain what their trajectory reveals about the simulation dynamics.

7. ECONOMIC & POWER ANALYSIS — Analyze resource flow networks. Who funded whom? Was there wealth concentration? Economic dependency (patronage)? Resource warfare (deliberate draining)? Free riding (low work rate, high inflows)? How did economic behavior correlate with survival? Present the top resource flows and net positions.

8. SOCIAL NETWORK ANALYSIS — Analyze alliance and rivalry structures. Identify power asymmetries, coalition patterns, information flow through rumors. Who was central? Who was peripheral? Was there evidence of group polarization? How did resource transfers map onto social bonds?

9. CRITICAL TIMELINE — Present the 15-20 most significant events using timeline-entry divs. Include major transfers. Each entry:
   <div class="timeline-entry">
     <span class="timeline-tick">Tick N</span>
     <span class="timeline-badge badge-TYPE">TYPE</span>
     <span>Description</span>
   </div>
   Translate tick numbers into real time (1 tick = {tick_unit}).

10. DISCUSSION — What do these results reveal about LLM-driven emergent social and economic dynamics? Were the outcomes predictable from initial conditions? What was surprising? How did the work/transfer mechanic shape agent strategies? What are the limitations of this single run?

11. CONCLUSION — Summarize principal findings in 2-3 sentences. State the emergence index and what it implies. Suggest what parameter changes (harshness, population size, tick count) might yield different results.

12. APPENDIX: COMPLETE DEMOGRAPHICS — Two tables: (a) all births with tick, name, parents, goal; (b) all deaths/removals with tick, name, cause, lifespan.

WRITING RULES:
- Write in formal academic prose. Third person. No first person.
- Every claim must cite specific evidence (agent names, tick numbers, event data).
- Quote agent thoughts verbatim in <blockquote> when they illuminate a point.
- Present numbers in context (e.g., "a mortality rate of 40% (6/15 agents) over 50 ticks").
- Be analytical, not narrative. This is a paper, not a story.
- Be thorough — mention every agent at least once.
- Do NOT output any CSS or JavaScript. Only HTML content tags.

Output ONLY the HTML content. No markdown fences. Start directly with <header>."""


def _build_agent_annex(run_dir: Path, data: dict) -> str:
    """Build static HTML annex with full per-agent event logs. No LLM needed."""
    agents_dir = run_dir / "agents"
    if not agents_dir.exists():
        return ""

    id_to_name: dict[str, str] = {}
    for name, arc in data.get("agent_arcs", {}).items():
        id_to_name[arc["id"]] = name

    sections: list[str] = []
    sections.append('<section class="annex">')
    sections.append("<h2>Annex: Complete Agent Logs</h2>")
    sections.append("<p>Unabridged event-by-event record for every agent in the simulation.</p>")

    import html as html_mod

    agent_files = sorted(agents_dir.glob("*.jsonl"))
    for agent_file in agent_files:
        agent_id = agent_file.stem
        name = id_to_name.get(agent_id, agent_id[:8])
        arc = None
        for n, a in data.get("agent_arcs", {}).items():
            if a["id"] == agent_id:
                arc = a
                break

        alive = arc["alive"] if arc else True
        css_class = "alive" if alive else "dead"

        events: list[dict] = []
        with open(agent_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        sections.append(f'<div class="agent-entry {css_class}">')
        sections.append(f"<h3>{html_mod.escape(name)}</h3>")

        if arc:
            meta_parts = [f"Goal: {html_mod.escape(arc.get('goal', '?'))}"]
            meta_parts.append(f"Born: tick {arc.get('born_tick', '?')}")
            if arc.get("died_tick") is not None:
                meta_parts.append(f"Died: tick {arc['died_tick']} ({html_mod.escape(arc.get('cause_of_death', '?'))})")
            else:
                meta_parts.append("Status: SURVIVED")
            if arc.get("work_rate") is not None:
                meta_parts.append(f"Work rate: {arc['work_rate']:.0%}")
            if arc.get("resource_sent"):
                meta_parts.append(f"Sent: {arc['resource_sent']}")
            if arc.get("resource_received"):
                meta_parts.append(f"Received: {arc['resource_received']}")
            sections.append(f'<p class="meta">{" · ".join(meta_parts)}</p>')

        sections.append('<table><thead><tr><th>Tick</th><th>Type</th><th>Content</th></tr></thead><tbody>')
        for ev in events:
            tick = ev.get("tick", "?")
            etype = html_mod.escape(str(ev.get("type", "?")))
            content = html_mod.escape(str(ev.get("content", "")))[:200]

            extras = []
            if ev.get("delta") is not None:
                extras.append(f"delta: {ev['delta']:+.2f}")
            if ev.get("amount") is not None:
                extras.append(f"amount: {ev['amount']:+.1f}")
            if ev.get("cause"):
                extras.append(f"cause: {html_mod.escape(str(ev['cause']))}")
            if ev.get("mode"):
                extras.append(html_mod.escape(str(ev["mode"])))
            if ev.get("goal"):
                extras.append(f"goal: {html_mod.escape(str(ev['goal']))}")

            if extras:
                content += f' <span class="log-extra">[{", ".join(extras)}]</span>'

            sections.append(f"<tr><td>{tick}</td><td>{etype}</td><td>{content}</td></tr>")

        sections.append("</tbody></table>")
        sections.append("</div>")

    sections.append("</section>")
    return "\n".join(sections)


async def _generate_report(
    run_dir: Path, data: dict, emergence: dict, model: str,
    api_key: str | None = None, base_url: str = "https://api.groq.com/openai/v1",
) -> Path | None:
    rich_text = _format_for_llm(data, emergence)

    survivor_names = [n for n, a in data.get("agent_arcs", {}).items() if a["alive"]]
    dead_names = [
        f"{n} (died tick {a['died_tick']}, {a['cause_of_death']})"
        for n, a in data.get("agent_arcs", {}).items() if not a["alive"]
    ]

    prompt = _PAPER_PROMPT.format(
        rich_text=rich_text,
        survivors=", ".join(survivor_names) if survivor_names else "NONE — total extinction",
        dead="; ".join(dead_names[:20]) if dead_names else "None",
        ei_score=emergence["emergence_index"],
        ei_max=emergence.get("max_emergence_index", 11),
        tick_unit=data.get("tick_unit", "unknown"),
        phenomena=", ".join(emergence["phenomena_detected"]) or "none",
    )

    try:
        from openai import AsyncOpenAI
        resolved_key = api_key or os.environ.get("GROQ_API_KEY", "")
        client = AsyncOpenAI(api_key=resolved_key, base_url=base_url)

        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=16000,
        )

        body = resp.choices[0].message.content or ""
        if body.startswith("```"):
            body = body.split("\n", 1)[-1].rsplit("```", 1)[0]
        body = body.strip()

        annex = _build_agent_annex(run_dir, data)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nanolife — Post-Simulation Evaluation</title>
<style>
{_PAPER_CSS}
</style>
</head>
<body>
<main>
{body}
{annex}
</main>
</body>
</html>"""

        report_path = run_dir / "report.html"
        report_path.write_text(html)
        return report_path

    except Exception as e:
        print(f"  [postmortem] Report generation failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════

async def run_postmortem(
    run_dir: str | Path,
    report_model: str = "openai/gpt-oss-120b",
    api_key: str | None = None,
    base_url: str = "https://api.groq.com/openai/v1",
    open_browser: bool = True,
) -> None:
    """Run emergence analysis + academic HTML report. Called from simulate.py's finally block."""
    run_dir = Path(run_dir)

    if not (run_dir / "world.jsonl").exists():
        print("  [postmortem] No world.jsonl found — skipping.")
        return

    print("\n" + "═" * 60)
    print("  POSTMORTEM")
    print("═" * 60)

    print("  Analyzing emergence...")
    emergence = analyze(run_dir)

    print(f"  Events: {emergence['total_events']}")
    for t, c in sorted(emergence["event_types"].items(), key=lambda x: -x[1]):
        print(f"    {t:15s} {c:5d}")
    print(f"  Emergence Index: {emergence['emergence_index']}/{emergence.get('max_emergence_index', 11)}")
    print(f"  Detected: {', '.join(emergence['phenomena_detected']) or 'none'}")

    if emergence["details"]:
        for d in emergence["details"][:10]:
            phenom = d["phenomenon"]
            extra = {k: v for k, v in d.items() if k != "phenomenon"}
            print(f"    [{phenom}] {extra}")

    print("\n  Extracting simulation data...")
    data = _extract_rich_data(run_dir)
    if not data:
        print("  [postmortem] No data extracted — skipping report.")
        return

    print(f"  {data['total_ticks']} ticks | {data['total_births']} births | {data['total_deaths']} deaths")
    print(f"  {len(data.get('alliances', []))} alliances | {len(data.get('rivalries', []))} rivalries | {data['total_rumors']} rumors")

    resolved_key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not resolved_key:
        print("  [postmortem] No API key available — skipping LLM report.")
        annex = _build_agent_annex(run_dir, data)
        if annex:
            annex_html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>nanolife — Agent Logs</title>
<style>{_PAPER_CSS}</style></head><body><main>{annex}</main></body></html>"""
            annex_path = run_dir / "agent_logs.html"
            annex_path.write_text(annex_html)
            print(f"  Agent logs saved: {annex_path}")
        analysis_path = run_dir / "analysis.json"
        analysis_path.write_text(json.dumps(emergence, indent=2))
        print(f"  Analysis saved: {analysis_path}")
        return

    print(f"\n  Generating academic evaluation via {report_model}...")
    report_path = await _generate_report(run_dir, data, emergence, report_model, api_key=resolved_key, base_url=base_url)

    if report_path:
        print(f"  Report saved: {report_path}")
        if open_browser:
            import platform
            import subprocess
            if platform.system() == "Darwin":
                subprocess.run(["open", str(report_path)])
            elif platform.system() == "Linux":
                subprocess.run(["xdg-open", str(report_path)])
            else:
                webbrowser.open(f"file://{report_path.resolve()}")

    analysis_path = run_dir / "analysis.json"
    analysis_path.write_text(json.dumps(emergence, indent=2))
    print(f"  Analysis saved: {analysis_path}")

    print("\n  Generating charts...")
    try:
        from nanolife.charts import generate_all_charts
        generate_all_charts(run_dir)
    except Exception as exc:
        print(f"  [postmortem] Chart generation failed: {exc}")

    viewer_src = Path(__file__).resolve().parent.parent / "social_graph_viewer.html"
    if viewer_src.exists():
        viewer_dst = run_dir / "social_graph_viewer.html"
        shutil.copy2(viewer_src, viewer_dst)
        print(f"  Social graph viewer: {viewer_dst}")

    print("═" * 60)
