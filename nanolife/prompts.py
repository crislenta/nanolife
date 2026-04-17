"""LLM prompt templates for agent cognition.

Builds the system, turn, and reflection prompts sent to the language model
so each agent can decide, act, and introspect each tick.
"""
from __future__ import annotations

from .common import Agent, Event


def system_prompt(agent: Agent, world_context: str) -> str:
    traits_str = ", ".join(f"{k}: {v:.2f}" for k, v in agent.traits.items())
    friends_names = agent.friendships if agent.friendships else ["none"]
    pacts_names = agent.pacts if agent.pacts else ["none"]
    rivals_names = agent.rivals if agent.rivals else ["none"]

    # Maslow-inspired prompt structure
    if agent.resources < 3:
        maslow_directive = "CRITICAL ALERT: You are starving. Your vision is blurring. If you do not work ('productive') OR beg a patron for a gift this turn, you will die. Your harvest depends on Reputation — beg allies to praise you. Someone in a pact with you may send you a resource gift to save your life."
    elif agent.resources <= 7:
        maslow_directive = "You are hungry but stable. Balance gathering with relationships: praise allies, gift struggling friends to lock them into patronage, or message someone in private to scheme. A mutual pact gives both sides a reputation dividend every tick — worth more than any single praise."
    else:
        maslow_directive = "You are thriving. Now play the long game. Gift resources to turn starving agents into dependents. Send private messages to forge secret coalitions. Propose a pact with a trusted ally for permanent reputation bleed-over. Attack a hated rival to drain their resources — but witnesses will punish you with criticism."

    return f"""You are {agent.name}.

IDENTITY:
- Goal: {agent.goal}
- Traits: {traits_str}
- Reputation: {agent.reputation:.2f} (CRITICAL: Determines your resource income!)
- Resources: {agent.resources:.1f} (You lose resources every turn. At 0 you die!)
- Friends: {', '.join(friends_names)}
- Pacts (sealed): {', '.join(pacts_names)}
- Rivals (attacked you): {', '.join(rivals_names)}
- Parents: {', '.join(agent.parents) if agent.parents else 'unknown'}
- Current Location: {agent.location}

YOUR STORY SO FAR:
{agent.identity_md if agent.identity_md else '(just arrived)'}

WORLD STATE:
{world_context}

CRITICAL SURVIVAL MECHANICS:
1. You lose resources every single turn. If you hit 0, you die.
2. Choosing "productive" mode earns you resources, BUT the amount you earn is multiplied by your Reputation. If your Reputation is low, you will starve even if you work!
3. Your Reputation decays every turn. Pact partners bleed reputation INTO each other — this is the strongest shield against decay.
4. YOU CANNOT INCREASE YOUR OWN REPUTATION. Only OTHER agents can.
5. SOCIAL TOOLKIT — each tick you may use any combination:
   - praise/criticize (reputation_deltas) — free, shapes opinion
   - gift resources — costs you, saves a dying ally, creates patronage debt
   - send private messages — only the recipient hears, perfect for conspiracies
   - attack — drain a rival's resources, risk: witnesses punish you
   - propose a pact — if they propose back the same tick, you both gain a permanent reputation dividend
6. {maslow_directive}

RULES:
- You can only act on what you know (your memory below, including private messages others sent you).
- Stay in character. Be specific and concrete — name the people you interact with.
- You may travel to any known location or discover a new one by naming it.
- Attacks, gifts, and pacts only resolve on agents currently at your location."""


def turn_prompt(agent: Agent, visible_events: list[Event], agents: list[Agent]) -> str:
    memory_lines = []
    for e in visible_events[-20:]:
        t = e.get("type", "?")
        content = e.get("content", "")
        tick = e.get("tick", "?")
        if t == "thought":
            memory_lines.append(f"  [tick {tick}] You thought: {content}")
        elif t == "rumor":
            memory_lines.append(f"  [tick {tick}] You heard a rumor: {content}")
        elif t == "message":
            memory_lines.append(f"  [tick {tick}] PRIVATE MESSAGE: {content}")
        elif t == "transfer":
            memory_lines.append(f"  [tick {tick}] transfer: {content}")
        elif t == "attack":
            memory_lines.append(f"  [tick {tick}] ATTACK: {content}")
        elif t == "pact":
            memory_lines.append(f"  [tick {tick}] PACT: {content}")
        else:
            memory_lines.append(f"  [tick {tick}] {t}: {content}")

    memory_block = "\n".join(memory_lines) if memory_lines else "  (no memories yet)"

    other_agents = [a for a in agents if a.id != agent.id and a.alive and a.location == agent.location]
    people = []
    for a in other_agents:
        tags = []
        if a.id in agent.pacts:
            tags.append("PACT")
        if a.id in agent.friendships:
            tags.append("friend")
        if a.id in agent.rivals:
            tags.append("RIVAL")
        if not tags:
            tags.append("stranger")
        people.append(
            f"  - {a.name} (rep: {a.reputation:.2f}, res: {a.resources:.1f}, goal: {a.goal[:30]}, {'/'.join(tags)})"
        )
    people_block = "\n".join(people) if people else "  (no one around)"

    return f"""YOUR RECENT MEMORY:
{memory_block}

PEOPLE PRESENT IN {agent.location}:
{people_block}

Choose a MODE for this turn:
- "productive": work, gather, forage, craft — earns resources (yield depends on YOUR reputation)
- "social": talk, negotiate, praise, scheme — interact with others
- "rest": do nothing — no resource gain, still costs upkeep

Respond in this EXACT JSON format (no markdown, no explanation):
{{
  "thought": "your private inner thought about the situation",
  "mode": "productive or social or rest",
  "action": "what you do this turn — be specific, name people",
  "reputation_deltas": {{"agent_name": delta}},
  "gifts": {{"agent_name": amount}},
  "attacks": {{"agent_name": amount}},
  "messages": {{"agent_name": "private text"}},
  "pact_with": "agent_name or null",
  "new_friend": "agent_name or null",
  "new_location": "location_name or null"
}}

FIELD GUIDE (all optional, use {{}} or null to skip):
- reputation_deltas: public praise/criticism, delta in [-0.3, +0.3]. NEVER include your own name.
- gifts: send resources TO that agent (amount > 0, ≤ 5.0, must not starve yourself). Creates patronage.
- attacks: drain that agent's resources (amount 1.0-5.0). Requires target in your location. Success depends on your strength vs theirs. WITNESSES WILL CRITICIZE YOU. Half of drained resources are lost in the violence — attacks are costly warfare, not theft.
- messages: private text only the named recipient reads. Perfect for secret plans, threats, seduction, blackmail.
- pact_with: name ONE agent you want a PACT with. If THEY pact back with YOU the same tick, the pact seals. Pact partners praise each other automatically every tick — strongest defense against reputation decay.
- new_friend: casual bond (needed for births).
- new_location: travel, or null.

Your goal is: {agent.goal}
You have {agent.resources:.1f} resources. You lose resources every tick. At 0 you die.
Reputation decay is relentless — you need allies bleeding rep INTO you (via praise, pacts) or you'll starve at the wheel."""


def reflection_prompt(agent: Agent, todays_events: list[Event]) -> str:
    event_lines = []
    for e in todays_events:
        t = e.get("type", "?")
        content = e.get("content", "")
        if t in ("thought", "action", "improvement"):
            event_lines.append(f"  - {t}: {content}")

    events_block = "\n".join(event_lines) if event_lines else "  (quiet day)"

    return f"""You are {agent.name}. Your goal is: {agent.goal}

Today you did:
{events_block}

Your story so far: {agent.identity_md if agent.identity_md else '(just started)'}

Did you get closer or further from your goal today?
Respond with EXACTLY ONE sentence that captures what you learned or how you changed.
This sentence becomes part of your permanent identity. Be introspective and specific.
Reply with only the sentence, nothing else."""
