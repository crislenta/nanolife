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

    # -- tiny helpers --------------------------------------------------------

    @staticmethod
    def _resolve(key: str | int, pool: list[Agent]) -> Agent | None:
        """Resolve by id first, then case-insensitive name."""
        return next((a for a in pool if a.id == str(key)), None) or \
            next((a for a in pool if a.name.lower() == str(key).lower()), None)

    def _evt(self, result: TickResult, tick: int, typ: str, content: str,
             agent: Agent = None, **extra) -> dict:
        """Create, log, and return an event dict."""
        evt: Event = {"tick": tick, "type": typ, "content": content}
        if agent:
            evt["agent"] = agent.id
        evt.update(extra)
        self.world.event_log.append(evt)
        result.events.append(evt)
        return evt

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

        base_ctx = (
            f"Tick {tick_num} | {self.world.clock.label()} | "
            f"Population: {len(alive)} | "
            f"Harshness: {self.world.harshness} | "
            f"Locations: {', '.join(self.world.locations)}"
        )

        # --- All agents observe and act in PARALLEL ---
        agent_contexts: list[tuple[Agent, list[Event], str]] = []
        wmap = self.world.world_map
        for agent in alive:
            visible = self.world.event_log.for_agent(agent.id)
            recent = visible[-self.world.memory_depth:]
            ctx = base_ctx
            if wmap is not None and agent.position is not None:
                patch = wmap.local_view(agent.position, radius=3, agents=alive)
                ctx = (
                    f"{base_ctx} | Position: {agent.position}\n"
                    f"LOCAL VIEW (radius 3, @ = you):\n{patch}"
                )
            agent_contexts.append((agent, recent, ctx))

        decisions = await asyncio.gather(*(
            self.cognitive.decide(agent, recent, ctx, alive)
            for agent, recent, ctx in agent_contexts
        ))

        for (agent, _, _), decision in zip(agent_contexts, decisions):
            # Restrict witnesses to agents in the same location
            local_witnesses = [a.id for a in alive if a.location == agent.location]

            mode = decision.get("mode", "productive")

            # Surface cognitive-layer parse errors so they show up in logs.
            if perr := decision.get("parse_error"):
                self._evt(result, tick_num, "parse_error", perr, agent, witnesses=[agent.id])

            # Grid step — only on the `walk` verb. Non-walk actions ignore delta.
            if mode == "walk":
                step = self._try_step(agent, decision.get("delta"), wmap, alive,
                                      tick_num, local_witnesses)
                if step is not None:
                    self.world.event_log.append(step)
                    result.events.append(step)
                    agent.memory.append(step)

            new_location = decision.get("new_location")
            if new_location and new_location != agent.location:
                if new_location not in self.world.locations:
                    self.world.locations.append(new_location)
                    self._evt(result, tick_num, "world",
                              f"{agent.name} discovered a new location: {new_location}",
                              witnesses=[a.id for a in alive])
                old_loc = agent.location
                agent.location = new_location
                witnesses = list(set(local_witnesses + [a.id for a in alive if a.location == new_location]))
                move_ev = self._evt(result, tick_num, "move",
                          f"{agent.name} moved from {old_loc} to {new_location}",
                          agent, witnesses=witnesses)
                agent.memory.append(move_ev)

            thought_ev = self._evt(result, tick_num, "thought", decision["thought"],
                       agent, witnesses=[agent.id])
            agent.memory.append(thought_ev)
            action_ev = self._evt(result, tick_num, "action", decision["action"],
                      agent, mode=mode, witnesses=local_witnesses)
            agent.memory.append(action_ev)

            if mode == "productive":
                # Resource-site gating: if scenario declared resource_sites
                # {resource: [terrain,...]} AND the action text mentions a
                # gated resource AND the agent is not on/adjacent to a matching
                # terrain tile, the productive action yields nothing. Absent
                # resource_sites = no gating (legacy behavior unchanged).
                blocked = False
                sites = self.world.resource_sites
                if sites and wmap is not None and agent.position is not None:
                    low = decision["action"].lower()
                    for res, terrains in sites.items():
                        if res.lower() not in low:
                            continue
                        ax, ay = agent.position
                        ok = any(
                            wmap.in_bounds(ax + dx, ay + dy)
                            and wmap.tiles[ay + dy][ax + dx].terrain in terrains
                            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
                        )
                        blocked = not ok
                        break
                if not blocked:
                    gain = self.world.base_gain * (1 - self.world.harshness) * max(0.1, 0.5 + agent.reputation)
                    agent.resources += gain

            for target_id, delta in decision.get("reputation_deltas", {}).items():
                target = next((a for a in self.world.agents if a.id == target_id), None)
                if target:
                    target.reputation = max(-1.0, min(1.0, target.reputation + delta))
                    self._evt(result, tick_num, "reputation",
                              f"{agent.name} {'praised' if delta > 0 else 'criticized'} {target.name}",
                              agent, delta=delta, source=agent.id, witnesses=local_witnesses)

            new_friend = decision.get("new_friend")
            if new_friend and new_friend not in agent.friendships:
                agent.friendships.append(new_friend)
                friend_target = next((a for a in self.world.agents if a.id == new_friend), None)
                if friend_target and agent.id not in friend_target.friendships:
                    friend_target.friendships.append(agent.id)
                self._evt(result, tick_num, "friendship",
                          f"{agent.name} became friends with {friend_target.name if friend_target else new_friend}",
                          agent, witnesses=local_witnesses)

        # --- Transfer / Trade: collect unilateral transfers, then match reciprocal barters ---
        xfer_bucket: dict[tuple[str, str], tuple[Agent, Agent, float]] = {}
        trade_matched: set[tuple[str, str]] = set()
        for (agent, _, _), decision in zip(agent_contexts, decisions):
            # Unilateral transfer: agent gives resources to target
            xfer = decision.get("transfer", {})
            xfer_to = xfer.get("to") if xfer else None
            xfer_amount = xfer.get("amount")
            if xfer_to and xfer_amount and xfer_amount > 0:
                target = self._resolve(xfer_to, alive)
                if target and agent.resources >= xfer_amount:
                    agent.resources -= xfer_amount
                    xfer_bucket[(agent.id, target.id)] = (agent, target, xfer_amount)

        # Match reciprocal barters: A->B and B->A = instant trade
        for (g_id, r_id), (giver, receiver, amt) in list(xfer_bucket.items()):
            if (g_id, r_id) in trade_matched:
                continue
            if (r_id, g_id) in xfer_bucket:
                o_giver, o_recv, o_amt = xfer_bucket[(r_id, g_id)]
                # Both give to each other => trade
                receiver.resources += amt
                o_recv.resources += o_amt
                self._evt(result, tick_num, "trade",
                    f"{giver.name} and {receiver.name} traded ({o_amt:.1f} <-> {amt:.1f})",
                    giver, witnesses=[a.id for a in alive if a.location == giver.location],
                    offer_give=o_amt, offer_want=amt, trade_partner=receiver.id,
                    trade_accepted=True)
                trade_matched.add((g_id, r_id))
                trade_matched.add((r_id, g_id))
            else:
                # Unilateral gift — credit receiver
                receiver.resources += amt
                self._evt(result, tick_num, "transfer",
                    f"{giver.name} gave {amt:.1f} to {receiver.name}",
                    giver, witnesses=[a.id for a in alive if a.location == giver.location],
                    amount=amt)

        # Log unmatched barter offers
        for (agent, _, _), decision in zip(agent_contexts, decisions):
            offer = decision.get("barter_offer")
            if offer and isinstance(offer, dict):
                to, give, want = offer.get("to"), offer.get("give"), offer.get("want")
                if to and give is not None and want is not None:
                    target = self._resolve(to, alive)
                    fwd = (agent.id, target.id) if target else None
                    rev = (target.id, agent.id) if target else None
                    if target and target.location == agent.location and fwd not in trade_matched and rev not in trade_matched:
                        self._evt(result, tick_num, "trade_offer",
                            f"{agent.name} proposed trade to {target.name}: {give:.1f} for {want:.1f} (no match)",
                            agent.id, witnesses=[a.id for a in alive if a.location == agent.location],
                            offer_give=give, offer_want=want, trade_partner=target.id,
                            trade_accepted=False)

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
            agent.identity_md = (agent.identity_md + " " + sentence).strip()
            imp_ev = self._evt(result, tick_num, "improvement", sentence,
                      agent, goal=agent.goal, witnesses=[agent.id])
            agent.memory.append(imp_ev)

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
                self._evt(result, tick_num, "death",
                          f"{agent.name} has died — cause: {cause}.", agent,
                          cause=cause, age_days=agent.age(tick_num),
                          witnesses=[a.id for a in self.world.alive_agents] + [agent.id])

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
            self._evt(result, tick_num, "compression", summary,
                      witnesses=[a.id for a in self.world.alive_agents])

        # --- Bound agent memories ---
        depth = self.world.memory_depth
        for agent in self.world.alive_agents:
            if len(agent.memory) > depth:
                agent.memory = agent.memory[-depth:]

        result.population = len(self.world.alive_agents)
        return result

    def _try_step(self, agent: Agent, delta: Any, wmap: Any, alive: list[Agent],
                  tick: int, witnesses: list[str]) -> Event | None:
        """Apply a 1-tile move if valid, returning a step event or None."""
        if wmap is None or agent.position is None or not isinstance(delta, (list, tuple)) or len(delta) != 2:
            return None
        dx, dy = delta
        if (dx, dy) == (0, 0):
            return None
        nx, ny = agent.position[0] + dx, agent.position[1] + dy
        if not wmap.passable(nx, ny):
            return None
        if any(a.alive and a.id != agent.id and a.position == (nx, ny) for a in alive):
            return None
        old = agent.position
        agent.position = (nx, ny)
        return {
            "tick": tick,
            "type": "step",
            "agent": agent.id,
            "content": f"{agent.name} stepped {old} -> {agent.position}",
            "witnesses": witnesses,
        }

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
