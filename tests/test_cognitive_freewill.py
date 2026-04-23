"""Contract tests for the free-will parser guarantees introduced in PR #4.

Background
----------
PR #4 (fix/cognitive-parser-freewill, merged to master 2026-04-23) fixed the
"97% of all agent actions were the literal string 'work'" bug. Three
invariants keep the fix in place:

  1. _salvage_partial_json recovers top-level string fields from truncated
     JSON whenever `mode` or `action` are intact. This is the #1 cause of
     parse failure with verbose reasoning models like gpt-oss-120b.

  2. When json.loads fails AND salvage returns None, the fallback must NOT
     collapse to action="work". It must use a rest/pause action so a
     wave of simultaneous API failures cannot recreate a free-will
     monoculture.

  3. When json.loads succeeds but the model emitted a concrete free-will
     verb ("forage", "gossip", "flee"...), the parser must preserve it
     verbatim — the engine trusts this string.

These are *unit* tests that never hit the LLM API. We construct an
LLMCognitive instance via __new__ to skip the GROQ_API_KEY check and
exercise the pure parsing methods directly.
"""
from __future__ import annotations

import json

import pytest

from nanolife.common import Agent
from nanolife.defaults.cognitive import LLMCognitive


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parser() -> LLMCognitive:
    """An LLMCognitive with the API client skipped — only parser methods are
    exercised, so no network dependency is needed."""
    inst = LLMCognitive.__new__(LLMCognitive)
    return inst


@pytest.fixture
def agents() -> list[Agent]:
    return [
        Agent(
            id="alice_id",
            name="Alice",
            alive=True,
            traits={},
            memory=[],
            friendships=[],
            parents=[],
            reputation=0.0,
            goal="",
            identity_md="",
            birth_tick=0,
            death_tick=None,
        ),
        Agent(
            id="bob_id",
            name="Bob",
            alive=True,
            traits={},
            memory=[],
            friendships=[],
            parents=[],
            reputation=0.0,
            goal="",
            identity_md="",
            birth_tick=0,
            death_tick=None,
        ),
    ]


# ---------------------------------------------------------------------------
# Invariant 1: _salvage_partial_json recovers mid-response truncations
# ---------------------------------------------------------------------------


def test_salvage_truncated_after_action(parser: LLMCognitive) -> None:
    """Model hit max_tokens right after emitting `action`. mode + action
    must both survive."""
    raw = '{"mode": "productive", "action": "forage for mushrooms", "thought": "I wonder'
    salvaged = parser._salvage_partial_json(raw)
    assert salvaged is not None
    assert salvaged["mode"] == "productive"
    assert salvaged["action"] == "forage for mushrooms"


def test_salvage_truncated_after_mode_only(parser: LLMCognitive) -> None:
    """Even if only `mode` made it out intact, salvage must succeed —
    `mode OR action` is the documented floor."""
    raw = '{"mode": "social", "action": "gossip with Al'
    salvaged = parser._salvage_partial_json(raw)
    assert salvaged is not None
    assert salvaged["mode"] == "social"
    # action truncated mid-string, should NOT be captured
    assert "action" not in salvaged


def test_salvage_no_complete_field_returns_none(parser: LLMCognitive) -> None:
    """If neither mode nor action completed, there is nothing to recover
    and salvage must return None so the caller hits the rest/pause
    fallback instead of inventing values."""
    raw = '{"thought": "I am thinking about'
    assert parser._salvage_partial_json(raw) is None


def test_salvage_handles_escaped_quotes(parser: LLMCognitive) -> None:
    """Escaped quotes inside the value must not terminate the regex early."""
    raw = r'{"mode": "productive", "action": "shout \"hello\" at Bob", "thought":'
    salvaged = parser._salvage_partial_json(raw)
    assert salvaged is not None
    assert salvaged["mode"] == "productive"
    assert "shout" in salvaged["action"]
    assert "Bob" in salvaged["action"]


def test_salvage_ignores_duplicate_keys(parser: LLMCognitive) -> None:
    """First occurrence wins; later repeats (rare but possible from
    stray model output) must not overwrite."""
    raw = '{"mode": "social", "action": "barter", "mode": "rest"}'
    salvaged = parser._salvage_partial_json(raw)
    assert salvaged is not None
    assert salvaged["mode"] == "social"


# ---------------------------------------------------------------------------
# Invariant 2: fallback must never collapse to action="work"
# ---------------------------------------------------------------------------


def test_total_parse_failure_does_not_return_work(
    parser: LLMCognitive, agents: list[Agent]
) -> None:
    """Garbage in -> rest/pause out. NEVER action="work", which was the
    root cause of the 97% monoculture."""
    raw = "the model had a stroke and returned pure prose with no json"
    result = parser._parse_response(raw, agents, agent_name="Alice")
    assert result["action"] != "work"
    assert result["mode"] == "rest"
    assert "pause" in result["action"].lower()


def test_empty_string_does_not_return_work(
    parser: LLMCognitive, agents: list[Agent]
) -> None:
    """API timeout / empty response must also resolve to pause, not work."""
    result = parser._parse_response("", agents, agent_name="Alice")
    assert result["action"] != "work"
    assert result["mode"] == "rest"


def test_only_thought_field_does_not_return_work(
    parser: LLMCognitive, agents: list[Agent]
) -> None:
    """Model produced thought but never got to mode/action. Salvage returns
    None, fallback fires, must NOT be work."""
    raw = '{"thought": "I am pondering the nature of the forest"'
    result = parser._parse_response(raw, agents, agent_name="Alice")
    assert result["action"] != "work"
    assert result["mode"] == "rest"


# ---------------------------------------------------------------------------
# Invariant 3: well-formed responses preserve free-will verbs
# ---------------------------------------------------------------------------


FREE_WILL_VERBS = [
    "forage for wild mushrooms",
    "hunt a boar with Bob",
    "weave a basket from reeds",
    "barter salt for flint",
    "gossip with Alice about the stranger",
    "flee toward the mountains",
    "teach the children a song",
    "bury the dead",
]


@pytest.mark.parametrize("action", FREE_WILL_VERBS)
def test_valid_json_preserves_free_will_action(
    parser: LLMCognitive, agents: list[Agent], action: str
) -> None:
    """Well-formed JSON with a concrete free-will action must survive
    verbatim — the engine trusts this string downstream."""
    raw = json.dumps(
        {
            "mode": "productive",
            "action": action,
            "thought": "being a person",
            "reputation_deltas": {},
        }
    )
    result = parser._parse_response(raw, agents)
    assert result["action"] == action
    assert result["mode"] == "productive"


def test_valid_json_with_code_fence_is_stripped(
    parser: LLMCognitive, agents: list[Agent]
) -> None:
    """Many models wrap JSON in ```json fences — must be stripped cleanly."""
    raw = '```json\n{"mode": "social", "action": "dance at the fire"}\n```'
    result = parser._parse_response(raw, agents)
    assert result["action"] == "dance at the fire"
    assert result["mode"] == "social"


def test_mode_normalized_and_bounded(
    parser: LLMCognitive, agents: list[Agent]
) -> None:
    """Unknown modes fall back to 'productive', not 'work'. Case is lowered."""
    raw = '{"mode": "SOCIAL", "action": "hunt"}'
    result = parser._parse_response(raw, agents)
    assert result["mode"] == "social"

    raw2 = '{"mode": "existential_crisis", "action": "hunt"}'
    result2 = parser._parse_response(raw2, agents)
    assert result2["mode"] == "productive"


def test_reputation_delta_clamped(
    parser: LLMCognitive, agents: list[Agent]
) -> None:
    """Reputation deltas must be clamped to [-0.3, +0.3] so a runaway model
    cannot explode the reputation graph in a single tick."""
    raw = json.dumps(
        {
            "mode": "social",
            "action": "praise Bob",
            "reputation_deltas": {"Bob": 99.0, "Alice": -99.0},
        }
    )
    result = parser._parse_response(raw, agents)
    assert result["reputation_deltas"]["bob_id"] == pytest.approx(0.3)
    assert result["reputation_deltas"]["alice_id"] == pytest.approx(-0.3)


def test_unknown_friend_name_ignored(
    parser: LLMCognitive, agents: list[Agent]
) -> None:
    """Hallucinated agent names in new_friend must be rejected (-> None),
    not leak into the social graph."""
    raw = json.dumps(
        {"mode": "social", "action": "greet the stranger", "new_friend": "Zephyr"}
    )
    result = parser._parse_response(raw, agents)
    assert result["new_friend"] is None


def test_known_friend_name_resolved_to_id(
    parser: LLMCognitive, agents: list[Agent]
) -> None:
    """A real agent name in new_friend is resolved to that agent's id."""
    raw = json.dumps(
        {"mode": "social", "action": "befriend Bob", "new_friend": "Bob"}
    )
    result = parser._parse_response(raw, agents)
    assert result["new_friend"] == "bob_id"
