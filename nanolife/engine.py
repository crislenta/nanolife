"""Simulation engine — runs the main tick loop.

Coordinates agent decisions, births, deaths, reputation, rumor spread,
memory compression, and resource economics each tick.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any

from .common import Agent, Event, TickResult, make_id
from .interfaces import CognitiveFunction, CompressionFunction, SpreadFunction
from .world import WorldState


class Engine:
    def __init__(
        self,
        world: WorldState,
        cognitive: CognitiveFunction,
        compression: CompressionFunction | None = None,
        spread: SpreadFunction | None = None,
    ):
        self.world = world
        self.cognitive = cognitive
        self.compression = compression
        self.spread = spread
        self._cost_usd: float = 0.0

    @property
    def cost(self) -> float:
        return self._cost_usd

    def spawn_agent(
        self,
        name: str,
        traits: dict[str, float] | None = None,
        goal: str = "survive and thrive",
        parents: list[str] | None = None,
        identity_md: str = "",
        lifespan: int = 365,
        location: str | None = None,
        resources: float | None = None,
        inherited_traits: dict[str, Any] | None = None,
        inherited_goal: str = "",
    ) -> tuple[Agent, Event]:
        agent = Agent(
            id=make_id(),
            name=name,
            alive=True,
            traits=traits or {"strength": random.random(), "charisma": random.random(), "intelligence": random.random()},
            memory=[],
            friendships=[],
            parents=parents or [],
            reputation=0.5,
            goal=goal,
            identity_md=identity_md,
            birth_tick=self.world.clock.tick,
            death_tick=None,
            resources=resources if resources is not None else self.world.starting_resources,
            lifespan=lifespan,
            location=location or random.choice(self.world.locations),
        )
        self.world.agents.append(agent)

        birth_event: Event = {
            "tick": self.world.clock.tick,
            "type": "birth",
            "agent": agent.id,
            "content": f"{agent.name} enters the world. Goal: {agent.goal}",
            "parents": agent.parents,
            "witnesses": [a.id for a in self.world.alive_agents],
        }
        if identity_md:
            birth_event["inherited_identity"] = identity_md
        if inherited_traits:
            birth_event["inherited_traits"] = inherited_traits
        if inherited_goal:
            birth_event["inherited_goal"] = inherited_goal
        self.world.event_log.append(birth_event)
        agent.memory.append(birth_event)
        return agent, birth_event

    async def tick(self) -> TickResult:
        tick_num = self.world.clock.advance()
        alive = self.world.alive_agents
        result = TickResult(tick=tick_num, population=len(alive))

        world_ctx = (
            f"Tick {tick_num} | {self.world.clock.label()} | "
            f"Population: {len(alive)} | "
            f"Harshness: {self.world.harshness} | "
            f"Locations: {', '.join(self.world.locations)}"
        )

        # --- All agents observe and act in PARALLEL ---
        agent_contexts = []
        for agent in alive:
            visible = self.world.event_log.for_agent(agent.id)
            recent = visible[-self.world.memory_depth:]
            agent_contexts.append((agent, recent))

        decisions = await asyncio.gather(*(
            self.cognitive.decide(agent, recent, world_ctx, alive)
            for agent, recent in agent_contexts
        ))

        for (agent, _), decision in zip(agent_contexts, decisions):
            # Restrict witnesses to agents in the same location
            local_witnesses = [a.id for a in alive if a.location == agent.location]

            new_location = decision.get("new_location")
            if new_location and new_location != agent.location:
                if new_location not in self.world.locations:
                    self.world.locations.append(new_location)
                    disc_event: Event = {
                        "tick": tick_num,
                        "type": "world",
                        "content": f"{agent.name} discovered a new location: {new_location}",
                        "witnesses": [a.id for a in alive],
                    }
                    self.world.event_log.append(disc_event)
                    result.events.append(disc_event)
                    
                move_event: Event = {
                    "tick": tick_num,
                    "type": "move",
                    "agent": agent.id,
                    "content": f"{agent.name} moved from {agent.location} to {new_location}",
                    "witnesses": list(set(local_witnesses + [a.id for a in alive if a.location == new_location])),
                }
                agent.location = new_location
                self.world.event_log.append(move_event)
                result.events.append(move_event)
                agent.memory.append(move_event)

            thought_event: Event = {
                "tick": tick_num,
                "type": "thought",
                "agent": agent.id,
                "content": decision["thought"],
                "witnesses": [agent.id],
            }

            mode = decision.get("mode", "productive")
            action_event: Event = {
                "tick": tick_num,
                "type": "action",
                "agent": agent.id,
                "content": decision["action"],
                "mode": mode,
                "witnesses": local_witnesses,
            }

            self.world.event_log.append(thought_event)
            self.world.event_log.append(action_event)
            agent.memory.append(thought_event)
            agent.memory.append(action_event)
            result.events.extend([thought_event, action_event])

            if mode == "productive":
                gain = self.world.base_gain * (1 - self.world.harshness) * max(0.1, 0.5 + agent.reputation)
                agent.resources += gain

            for target_id, delta in decision.get("reputation_deltas", {}).items():
                target = next((a for a in self.world.agents if a.id == target_id), None)
                if target:
                    target.reputation = max(-1.0, min(1.0, target.reputation + delta))
                    rep_event: Event = {
                        "tick": tick_num,
                        "type": "reputation",
                        "agent": agent.id,
                        "content": f"{agent.name} {'praised' if delta > 0 else 'criticized'} {target.name}",
                        "delta": delta,
                        "source": agent.id,
                        "witnesses": local_witnesses,
                    }
                    self.world.event_log.append(rep_event)
                    result.events.append(rep_event)

            new_friend = decision.get("new_friend")
            if new_friend and new_friend not in agent.friendships:
                agent.friendships.append(new_friend)
                friend_target = next((a for a in self.world.agents if a.id == new_friend), None)
                if friend_target and agent.id not in friend_target.friendships:
                    friend_target.friendships.append(agent.id)
                friendship_event: Event = {
                    "tick": tick_num,
                    "type": "friendship",
                    "agent": agent.id,
                    "content": f"{agent.name} became friends with {friend_target.name if friend_target else new_friend}",
                    "witnesses": local_witnesses,
                }
                self.world.event_log.append(friendship_event)
                result.events.append(friendship_event)

        # --- Rumor spread ---
        if self.spread:
            rumors = self.spread.spread(alive, self.world.event_log.all(), degradation=0.2)
            for rumor in rumors:
                rumor["tick"] = tick_num
                self.world.event_log.append(rumor)
                result.events.append(rumor)
                if "agent" in rumor:
                    target_agent = next((a for a in alive if a.id == rumor["agent"]), None)
                    if target_agent:
                        target_agent.memory.append(rumor)

        # --- Self-improvement (reflection) — all agents in PARALLEL ---
        reflect_inputs = []
        for agent in alive:
            todays = [e for e in result.events if e.get("agent") == agent.id]
            reflect_inputs.append((agent, todays))

        sentences = await asyncio.gather(*(
            self.cognitive.reflect(agent, todays)
            for agent, todays in reflect_inputs
        ))

        for (agent, _), sentence in zip(reflect_inputs, sentences):
            if agent.identity_md:
                agent.identity_md += " " + sentence
            else:
                agent.identity_md = sentence

            imp_event: Event = {
                "tick": tick_num,
                "type": "improvement",
                "agent": agent.id,
                "content": sentence,
                "goal": agent.goal,
                "witnesses": [agent.id],
            }
            self.world.event_log.append(imp_event)
            agent.memory.append(imp_event)
            result.events.append(imp_event)

        # --- Per-agent resource drain + reputation decay ---
        for agent in alive:
            agent.resources -= self.world.base_drain + self.world.harshness
            agent.reputation = max(-1.0, agent.reputation - self.world.reputation_decay)

        # --- Death checks ---
        for agent in list(alive):
            cause = self._check_death(agent, tick_num)
            if cause:
                agent.alive = False
                agent.death_tick = tick_num
                self.world.total_deaths += 1
                result.deaths += 1
                death_event: Event = {
                    "tick": tick_num,
                    "type": "death",
                    "agent": agent.id,
                    "content": f"{agent.name} has died — cause: {cause}.",
                    "cause": cause,
                    "age_days": agent.age(tick_num),
                    "witnesses": [a.id for a in self.world.alive_agents] + [agent.id],
                }
                self.world.event_log.append(death_event)
                result.events.append(death_event)

        # --- Birth checks ---
        births = await self._check_births(tick_num)
        for child, birth_evt in births:
            result.births += 1
            self.world.total_births += 1
            result.events.append(birth_evt)

        # --- Compression ---
        if self.compression and self.world.event_log.token_estimate() > self.world.token_threshold:
            summary, remaining = await self.compression.compress(
                self.world.event_log.all(), self.world.token_threshold
            )
            self.world.event_log.replace(remaining)
            comp_event: Event = {
                "tick": tick_num,
                "type": "compression",
                "content": summary,
                "witnesses": [a.id for a in self.world.alive_agents],
            }
            self.world.event_log.append(comp_event)
            result.events.append(comp_event)

        # --- Bound agent memories ---
        depth = self.world.memory_depth
        for agent in self.world.alive_agents:
            if len(agent.memory) > depth:
                agent.memory = agent.memory[-depth:]

        result.population = len(self.world.alive_agents)
        return result

    def _check_death(self, agent: Agent, tick: int) -> str | None:
        if agent.age(tick) >= agent.lifespan:
            return "old age"
        if agent.resources <= 0:
            return "starvation"
        return None

    async def _check_births(self, tick: int) -> list[tuple[Agent, Event]]:
        alive = self.world.alive_agents
        births: list[tuple[Agent, Event]] = []
        friend_pairs: set[tuple[str, str]] = set()
        for a in alive:
            for friend_id in a.friendships:
                pair = tuple(sorted([a.id, friend_id]))
                friend_pairs.add(pair)  # type: ignore[arg-type]

        birth_cost = 5.0
        for a_id, b_id in friend_pairs:
            a = next((x for x in alive if x.id == a_id), None)
            b = next((x for x in alive if x.id == b_id), None)
            if not a or not b or a.location != b.location:
                continue
            if a.resources < birth_cost or b.resources < birth_cost:
                continue

            if random.random() < 0.05:
                cost_a = a.resources / 2
                cost_b = b.resources / 2
                a.resources -= cost_a
                b.resources -= cost_b
                child_resources = (cost_a + cost_b) / 2

                child_traits = {}
                all_keys = set(list(a.traits.keys()) + list(b.traits.keys()))
                drift = 0.1
                for k in all_keys:
                    avg = (a.traits.get(k, 0.5) + b.traits.get(k, 0.5)) / 2
                    child_traits[k] = max(0.0, min(1.0, avg + random.uniform(-drift, drift)))

                child_name = await self.cognitive.name_child(a, b)
                goal_parent = random.choice([a, b])
                child_goal = goal_parent.goal
                child_identity = f"Child of {a.name} and {b.name}."

                inherited_traits_log = {
                    "from_parent_a": {a.name: dict(a.traits)},
                    "from_parent_b": {b.name: dict(b.traits)},
                    "child_result": dict(child_traits),
                }
                inherited_goal_log = f"From {goal_parent.name}: {child_goal}"

                child, birth_evt = self.spawn_agent(
                    name=child_name,
                    traits=child_traits,
                    goal=child_goal,
                    parents=[a.id, b.id],
                    identity_md=child_identity,
                    location=a.location,
                    resources=child_resources,
                    inherited_traits=inherited_traits_log,
                    inherited_goal=inherited_goal_log,
                )
                births.append((child, birth_evt))
        return births

    async def run(self, ticks: int, on_tick: Any = None) -> list[TickResult]:
        results: list[TickResult] = []
        for _ in range(ticks):
            result = await self.tick()
            results.append(result)
            if on_tick:
                on_tick(result)
            if not self.world.alive_agents:
                break
        return results
