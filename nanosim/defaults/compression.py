"""LLM-backed event log compression (default implementation).

Summarizes the oldest half of the event log into a historical record
when the token budget is exceeded.
"""
from __future__ import annotations

import json
import os
from typing import Any

from ..common import Event
from ..interfaces import CompressionFunction


class LLMCompression(CompressionFunction):
    """Summarize oldest events when log exceeds token budget via LLM call."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        api_key: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._base_url = base_url
        self.total_tokens: int = 0
        self.total_cost: float = 0.0

    async def compress(self, events: list[Event], token_budget: int) -> tuple[str, list[Event]]:
        # Split: compress oldest half, keep newest half
        split = len(events) // 2
        to_compress = events[:split]
        to_keep = events[split:]

        summary = await self._summarize(to_compress)
        return summary, to_keep

    async def _summarize(self, events: list[Event]) -> str:
        lines = []
        for e in events[:100]:
            t = e.get("type", "?")
            content = e.get("content", "")
            tick = e.get("tick", "?")
            lines.append(f"[tick {tick}] {t}: {content}")

        events_text = "\n".join(lines)
        prompt = f"""Summarize these world events into a concise historical record (3-5 sentences).
Preserve: key friendships, deaths, betrayals, leadership changes, and cultural developments.
Write as a historian — factual, compact, mentioning names.

EVENTS:
{events_text}

HISTORICAL RECORD:"""

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
            resp = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            usage = resp.usage
            if usage:
                self.total_tokens += usage.total_tokens
            return (resp.choices[0].message.content or "History was lost.").strip()
        except Exception:
            return self._stub_summarize(events)

    @staticmethod
    def _stub_summarize(events: list[Event]) -> str:
        """Fallback when no LLM available."""
        deaths = [e for e in events if e.get("type") == "death"]
        births = [e for e in events if e.get("type") == "birth"]
        friendships = [e for e in events if e.get("type") == "friendship"]

        parts = []
        if deaths:
            names = [e.get("content", "someone")[:30] for e in deaths[:3]]
            parts.append(f"{len(deaths)} died: {'; '.join(names)}")
        if births:
            parts.append(f"{len(births)} were born")
        if friendships:
            parts.append(f"{len(friendships)} friendships formed")
        parts.append(f"Covered ticks {events[0].get('tick', '?')}-{events[-1].get('tick', '?')}")

        return "Historical record: " + ". ".join(parts) + "."


