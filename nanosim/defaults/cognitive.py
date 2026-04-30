"""LLM-backed cognitive function (default implementation).

Uses any OpenAI-compatible API (Groq, OpenRouter, etc.) to let agents
decide, reflect, and name children each tick.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
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
            max_tokens=500,
            reasoning_effort="low",
        )

        if not resp:
            # API failure — do NOT force "work". Rest so the fallback does not
            # collapse free-will into a monoculture when many agents fail at once.
            return {
                "thought": "My mind is clouded.",
                "mode": "rest",
                "action": "pause and gather my thoughts",
                "reputation_deltas": {},
                "new_friend": None,
                "new_location": None,
            }

        self.llm_calls += 1
        usage = resp.usage
        if usage:
            prompt_t, completion_t, total_t = self._normalize_usage(usage)
            self.total_tokens += total_t
            self.prompt_tokens += prompt_t
            self.completion_tokens += completion_t
            self._update_cost(prompt_t, completion_t)

        raw = resp.choices[0].message.content or "{}"
        return self._parse_response(raw, agents, agent_name=agent.name)

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
            prompt_t, completion_t, total_t = self._normalize_usage(usage)
            self.total_tokens += total_t
            self.prompt_tokens += prompt_t
            self.completion_tokens += completion_t
            self._update_cost(prompt_t, completion_t)

        msg = resp.choices[0].message if resp and resp.choices else None
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        return (content or generate_fallback()).strip()

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
            prompt_t, completion_t, total_t = self._normalize_usage(usage)
            self.total_tokens += total_t
            self.prompt_tokens += prompt_t
            self.completion_tokens += completion_t
            self._update_cost(prompt_t, completion_t)
        msg = resp.choices[0].message if resp and resp.choices else None
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        parts = (content or "").strip().split()
        name = parts[0] if parts else ""
        return name or f"{parent_a.name[:2]}{parent_b.name[-3:]}"

    @staticmethod
    def _normalize_usage(usage: Any) -> tuple[int, int, int]:
        """Normalize token accounting across providers.

        Some OpenAI-compatible backends (including Vertex in some responses)
        may return None for one or more usage fields.
        """
        prompt_t = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_t = int(getattr(usage, "completion_tokens", 0) or 0)
        total_raw = getattr(usage, "total_tokens", None)
        total_t = int(total_raw) if total_raw is not None else prompt_t + completion_t
        return prompt_t, completion_t, total_t

    def _parse_response(self, raw: str, agents: list[Agent], agent_name: str = "?") -> dict[str, Any]:
        original = raw
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self._log_parse_failure(agent_name, original)
            salvaged = self._salvage_partial_json(raw)
            if salvaged is not None:
                data = salvaged
            else:
                # Last resort: keep thought, but do NOT force "work" — use "pause"
                # so the fallback itself does not collapse free-will into a monoculture.
                return {
                    "thought": raw[:200],
                    "mode": "rest",
                    "action": "pause and gather my thoughts",
                    "reputation_deltas": {},
                    "new_friend": None,
                    "new_location": None,
                }

        name_to_id = {a.name.lower(): a.id for a in agents}

        # Spatial schema promotes `walk` to a first-class action verb and uses
        # `action` for the verb + `description` for the free-form text. Legacy
        # schema keeps `mode` (productive/social/rest) + free-form `action`.
        raw_action = str(data.get("action", "")).lower().strip()
        if raw_action == "walk":
            mode = "walk"
            action_text = str(data.get("description", data.get("action", "walk")))[:300]
        else:
            mode = str(data.get("mode", raw_action or "productive")).lower().strip()
            if mode not in ("productive", "social", "rest"):
                mode = "productive"
            action_text = str(data.get("description", data.get("action", "work")))[:300]

        rep_deltas: dict[str, float] = {}
        for name, delta in data.get("reputation_deltas", {}).items():
            aid = name_to_id.get(name.lower())
            if aid and isinstance(delta, (int, float)):
                rep_deltas[aid] = max(-0.3, min(0.3, float(delta)))

        new_friend = None
        friend_name = data.get("new_friend")
        if friend_name and isinstance(friend_name, str):
            new_friend = name_to_id.get(friend_name.lower())

        raw_loc = data.get("new_location")
        new_location = raw_loc if isinstance(raw_loc, str) and raw_loc.lower() not in ("null", "none", "") else None

        # Grid step. Only meaningful when mode=="walk"; the engine ignores
        # delta for other modes. Clamp components to {-1, 0, 1}.
        delta: list[int] | None = None
        parse_error: str | None = None
        raw_delta = data.get("delta")
        if isinstance(raw_delta, (list, tuple)) and len(raw_delta) == 2:
            try:
                dx, dy = int(raw_delta[0]), int(raw_delta[1])
                delta = [max(-1, min(1, dx)), max(-1, min(1, dy))]
            except (TypeError, ValueError):
                delta = None

        # walk REQUIRES a non-zero delta. Fall back to productive with a
        # parse-error marker so the engine can log it.
        if mode == "walk" and (delta is None or delta == [0, 0]):
            parse_error = f"walk chosen but delta invalid: {raw_delta!r}"
            mode = "productive"
            delta = None

        out: dict[str, Any] = {
            "thought": str(data.get("thought", "..."))[:300],
            "mode": mode,
            "action": action_text,
            "reputation_deltas": rep_deltas,
            "new_friend": new_friend,
            "new_location": new_location,
        }
        if delta is not None:
            out["delta"] = delta
        if parse_error:
            out["parse_error"] = parse_error
        return out

    _KEY_RE = re.compile(r'"(mode|action|thought|new_friend|new_location)"\s*:\s*"((?:[^"\\]|\\.)*)"')

    def _salvage_partial_json(self, raw: str) -> dict[str, Any] | None:
        """Extract top-level string fields from truncated JSON.

        When the model gets cut off mid-response (the #1 cause of parse
        failure here), the prefix is still well-formed JSON up to the
        truncation point. We can recover whatever complete "key": "value"
        pairs exist and surface them to the engine, so free-will survives
        even a cut-off response.
        """
        found: dict[str, Any] = {}
        for m in self._KEY_RE.finditer(raw):
            key, val = m.group(1), m.group(2)
            if key not in found:
                found[key] = val
        # Must have at least mode or action — otherwise nothing to salvage.
        if "mode" not in found and "action" not in found:
            return None
        return found

    def _log_parse_failure(self, agent_name: str, raw: str) -> None:
        """Append a parse-failure sample to NANOLIFE_DEBUG_COGNITIVE if set.

        The env var holds an absolute path. Appended lines are JSONL with
        {ts, agent, raw}. Silent no-op when unset, so production runs
        carry zero overhead beyond an env lookup.
        """
        path = os.environ.get("NANOLIFE_DEBUG_COGNITIVE")
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "agent": agent_name, "raw": raw}) + "\n")
        except Exception:
            pass

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
