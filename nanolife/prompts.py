"""LLM prompt templates for agent cognition.

Builds the system, turn, and reflection prompts sent to the language model
so each agent can decide, act, and introspect each tick.
"""
from __future__ import annotations

from .common import Agent, Event


def system_prompt(agent: Agent, world_context: str) -> str:
    traits_str = ", ".join(f"{k}: {v:.2f}" for k, v in agent.traits.items())
    friends_names = agent.friendships if agent.friendships else ["none"]

    # Maslow-inspired prompt structure
    if agent.resources < 3:
        maslow_directive = "CRITICAL ALERT: You are starving. Your vision is blurring. If you do not choose the 'productive' mode this turn to gather food, you will die. BUT REMEMBER: your harvest depends on your Reputation! Beg others to praise you!"
    elif agent.resources <= 7:
        maslow_directive = "You are hungry but stable. You must balance gathering resources ('productive') with building alliances ('social'). You need others to praise you to keep your Reputation high, or your work will yield nothing."
    else:
        maslow_directive = "Your belly is full and your survival is secured. Now is the time to strike. Focus entirely on your grand goal. Scheme, betray, or build strong mutual-praise alliances to keep your Reputation and income high."

    return f"""You are {agent.name}.

IDENTITY:
- Goal: {agent.goal}
- Traits: {traits_str}
- Reputation: {agent.reputation:.2f} (CRITICAL: Determines your resource income!)
- Resources: {agent.resources:.1f} (You lose resources every turn. At 0 you die!)
- Friends: {', '.join(friends_names)}
- Parents: {', '.join(agent.parents) if agent.parents else 'unknown'}
- Current Location: {agent.location}

YOUR STORY SO FAR:
{agent.identity_md if agent.identity_md else '(just arrived)'}

WORLD STATE:
{world_context}

CRITICAL SURVIVAL MECHANICS:
1. You lose resources every single turn. If you hit 0, you die.
2. Choosing "productive" mode earns you resources, BUT the amount you earn is multiplied by your Reputation. If your Reputation is low, you will starve even if you work!
3. Your Reputation decays every turn.
4. YOU CANNOT INCREASE YOUR OWN REPUTATION. Only OTHER agents can increase your reputation by praising you.
5. Therefore, to survive, you MUST form alliances and mutually praise each other! Use 'reputation_deltas' to praise allies so they survive, and convince them to praise you back!
6. {maslow_directive}

RULES:
- You can only act on what you know (your memory below).
- Stay in character. Be specific and concrete — name the people you interact with.
- You may travel to any known location or discover a new one by naming it."""


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
        else:
            memory_lines.append(f"  [tick {tick}] {t}: {content}")

    memory_block = "\n".join(memory_lines) if memory_lines else "  (no memories yet)"

    other_agents = [a for a in agents if a.id != agent.id and a.alive and a.location == agent.location]
    people = []
    for a in other_agents:
        rel = "friend" if a.id in agent.friendships else "acquaintance"
        people.append(f"  - {a.name} (rep: {a.reputation:.2f}, {rel})")
    people_block = "\n".join(people) if people else "  (no one around)"

    return f"""YOUR RECENT MEMORY:
{memory_block}

PEOPLE PRESENT IN {agent.location}:
{people_block}

Choose a MODE for this turn:
- "productive": work, gather, forage, craft — earns resources (yield depends on YOUR reputation)
- "social": talk, negotiate, praise, scheme — interact with others (use reputation_deltas to boost allies or crush enemies)
- "rest": do nothing — no resource gain, still costs upkeep

Respond in this EXACT JSON format (no markdown, no explanation):
{{
  "thought": "your private inner thought about the situation",
  "mode": "productive or social or rest",
  "action": "what you do this turn — be specific, name people",
  "reputation_deltas": {{"agent_name": delta}},
  "new_friend": "agent_name or null",
  "new_location": "location_name or null"
}}

reputation_deltas: dict of agent names -> float delta (-0.3 to +0.3). Use positive values to praise allies (helping them survive) and negative to criticize enemies. You CANNOT include your own name.
new_friend: name of ONE agent you want to befriend, or null.
new_location: a location to travel to (existing or new), or null to stay.

Your goal is: {agent.goal}
You have {agent.resources:.1f} resources. You lose resources every tick. At 0 you die.
Remember: You need others to praise you to keep your reputation high, or your productive work will yield nothing!"""


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
