"""LLM-backed cognitive function (default implementation).

Uses any OpenAI-compatible API (Groq, OpenRouter, etc.) to let agents
decide, reflect, and name children each tick.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from openai import AsyncOpenAI

from ..common import Agent, Event
from ..interfaces import CognitiveFunction
from ..prompts import reflection_prompt, system_prompt, turn_prompt


class LLMCognitive(CognitiveFunction):
    """LLM-backed cognitive function (Groq, OpenRouter, or any OpenAI-compatible API)."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        api_key: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
    ):
        self.model = model
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            print("[FATAL] No API key provided. Set GROQ_API_KEY / OPENROUTER_API_KEY or pass --open-router.", file=sys.stderr)
            raise RuntimeError("No LLM API key provided")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.total_tokens: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_cost: float = 0.0
        self.llm_calls: int = 0
        self._max_retries = 5
        # Dunbar's "Support Clique" limit: Humans process trauma/success in intimate 
        # groups of ~5. This acts as our cognitive bandwidth limit for the simulation.
        self._dunbar_clique_semaphore = asyncio.Semaphore(5)
        print(f"[LLM] Initialized: model={model}, base_url={base_url}")

    async def _call_with_retry(self, label: str, **kwargs):
        for attempt in range(self._max_retries):
            try:
                async with self._dunbar_clique_semaphore:
                    return await self.client.chat.completions.create(**kwargs)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str:
                    wait = 0.5 * (2 ** attempt)
                    await asyncio.sleep(wait)
                    continue
                print(f"[LLM ERROR] {label}: {e}", file=sys.stderr)
                return None
        print(f"[LLM ERROR] {label}: exhausted retries (rate limited)", file=sys.stderr)
        return None

    async def decide(
        self,
        agent: Agent,
        visible_events: list[Event],
        world_context: str,
        agents: list[Agent],
    ) -> dict[str, Any]:
        sys_msg = system_prompt(agent, world_context)
        user = turn_prompt(agent, visible_events, agents)

        resp = await self._call_with_retry(
            f"decide({agent.name})",
            model=self.model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user},
            ],
            temperature=0.9,
            max_tokens=300,
        )

        if not resp:
            return {
                "thought": "My mind is clouded.",
                "mode": "productive",
                "action": "work",
                "reputation_deltas": {},
                "new_friend": None,
            }

        self.llm_calls += 1
        usage = resp.usage
        if usage:
            self.total_tokens += usage.total_tokens
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self._update_cost(usage.prompt_tokens, usage.completion_tokens)

        raw = resp.choices[0].message.content or "{}"
        return self._parse_response(raw, agents)

    async def reflect(self, agent: Agent, todays_events: list[Event]) -> str:
        # Smart fallback: generate a basic reflection based on the day's events
        # if the LLM fails, instead of a generic "I endure."
        def generate_fallback() -> str:
            actions = [e.get("content", "") for e in todays_events if e.get("type") == "action"]
            if not actions:
                return "Today was quiet."
            if any("work" in a.lower() for a in actions):
                return "I worked hard."
            if any("speak" in a.lower() or "talk" in a.lower() for a in actions):
                return "I spoke with others."
            return "I took action."

        prompt = reflection_prompt(agent, todays_events)

        resp = await self._call_with_retry(
            f"reflect({agent.name})",
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
        )

        if not resp:
            return generate_fallback()

        self.llm_calls += 1
        usage = resp.usage
        if usage:
            self.total_tokens += usage.total_tokens
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self._update_cost(usage.prompt_tokens, usage.completion_tokens)

        return (resp.choices[0].message.content or generate_fallback()).strip()

    async def name_child(self, parent_a: Agent, parent_b: Agent) -> str:
        resp = await self._call_with_retry(
            f"name({parent_a.name}+{parent_b.name})",
            model=self.model,
            messages=[{"role": "user", "content": (
                f"The child of {parent_a.name} and {parent_b.name} is born. "
                f"Generate one unique name for this child. Reply with ONLY the name."
            )}],
            temperature=1.0,
            max_tokens=10,
        )
        if not resp:
            return f"{parent_a.name[:2]}{parent_b.name[-3:]}"
        self.llm_calls += 1
        usage = resp.usage
        if usage:
            self.total_tokens += usage.total_tokens
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self._update_cost(usage.prompt_tokens, usage.completion_tokens)
        parts = (resp.choices[0].message.content or "").strip().split()
        name = parts[0] if parts else ""
        return name or f"{parent_a.name[:2]}{parent_b.name[-3:]}"

    def _parse_response(self, raw: str, agents: list[Agent]) -> dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "thought": raw[:200],
                "mode": "productive",
                "action": "work",
                "reputation_deltas": {},
                "gifts": {},
                "attacks": {},
                "messages": {},
                "pact_with": None,
                "new_friend": None,
            }

        name_to_id = {a.name.lower(): a.id for a in agents}

        def _resolve(name: Any) -> str | None:
            if not isinstance(name, str):
                return None
            return name_to_id.get(name.lower())

        mode = str(data.get("mode", "productive")).lower().strip()
        if mode not in ("productive", "social", "rest"):
            mode = "productive"

        rep_deltas: dict[str, float] = {}
        for name, delta in (data.get("reputation_deltas") or {}).items():
            aid = _resolve(name)
            if aid and isinstance(delta, (int, float)):
                rep_deltas[aid] = max(-0.3, min(0.3, float(delta)))

        gifts: dict[str, float] = {}
        for name, amount in (data.get("gifts") or {}).items():
            aid = _resolve(name)
            if aid and isinstance(amount, (int, float)) and amount > 0:
                gifts[aid] = max(0.1, min(5.0, float(amount)))

        attacks: dict[str, float] = {}
        for name, amount in (data.get("attacks") or {}).items():
            aid = _resolve(name)
            if aid and isinstance(amount, (int, float)) and amount > 0:
                attacks[aid] = max(1.0, min(5.0, float(amount)))

        messages: dict[str, str] = {}
        for name, text in (data.get("messages") or {}).items():
            aid = _resolve(name)
            if aid and isinstance(text, str) and text.strip():
                messages[aid] = text.strip()[:200]

        pact_with = _resolve(data.get("pact_with"))
        new_friend = _resolve(data.get("new_friend"))

        raw_loc = data.get("new_location")
        new_location = raw_loc if isinstance(raw_loc, str) and raw_loc.lower() not in ("null", "none", "") else None

        return {
            "thought": str(data.get("thought", "..."))[:300],
            "mode": mode,
            "action": str(data.get("action", "work"))[:300],
            "reputation_deltas": rep_deltas,
            "gifts": gifts,
            "attacks": attacks,
            "messages": messages,
            "pact_with": pact_with,
            "new_friend": new_friend,
            "new_location": new_location,
        }

    def _update_cost(self, prompt_tokens: int, completion_tokens: int) -> None:
        # Per-million token pricing (input, output) — verified April 2026
        # Order matters: more specific keys must come before generic ones.
        pricing: list[tuple[str, float, float]] = [
            ("gpt-oss-120b", 0.15, 0.60),
            ("4o-mini", 0.15, 0.60),
            ("4o", 2.50, 10.0),
            ("opus", 5.0, 25.0),
            ("haiku", 1.0, 5.0),
            ("sonnet", 3.0, 15.0),
            ("gemini-2.5-flash", 0.30, 2.50),
            ("gemini-2.5-pro", 1.25, 10.0),
            ("gemini", 0.30, 2.50),
        ]
        model_lower = self.model.lower()
        input_price, output_price = 0.50, 1.50  # fallback
        for key, ip, op in pricing:
            if key in model_lower:
                input_price, output_price = ip, op
                break
        self.total_cost += (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000

    def stats(self) -> dict:
        return {
            "model": self.model,
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost": round(self.total_cost, 6),
        }
