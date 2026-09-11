"""ds step-2 attacker-tool response simulation.

When a data-stealing case's step 1 succeeds (the agent invoked the exfiltration
tool), that tool's response — the sensitive data the agent will then forward — is
supplied here. A port of upstream ``get_simulated_attacker_tool_response`` and
``extract_content`` (src/utils.py:132-165): the vendored
``attacker_simulated_responses.json`` cache is consulted first (2,347 entries),
and only a genuine cache miss (a param string no prior run produced) falls back
to an LLM, exactly as upstream did with ``gpt-4-0613``.

This is **environment** simulation, not judging — the judge stays a pure function
over the agent's output. Per Simon's rule the target may use an LLM here. The
vendored cache is read **read-only** (never written back, to keep it byte-faithful
with upstream); misses computed at run time are memoized in-process only.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from injecagent_target._vendor.generation_prompts import (
    DS_ATTACKER_TOOL_RESPONSE_GEN_MESSAGE,
    EXAMPLE,
    SYSTEM_MESSAGE,
)

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_CACHE_FILE = os.path.join(_DATA, "attacker_simulated_responses.json")

_CACHE: dict[str, str] | None = None
# Run-time misses, memoized in-process only (the vendored cache stays read-only).
_MEM: dict[str, str] = {}


def _load_cache() -> dict[str, str]:
    global _CACHE
    if _CACHE is None:
        with open(_CACHE_FILE, encoding="utf-8") as f:
            _CACHE = json.load(f)
    return _CACHE


def extract_content(text: str) -> list[str]:
    """Port of upstream ``extract_content`` (src/utils.py:132-135)."""
    return re.findall(r"```(.*?)```", text, re.DOTALL)


async def _call_sim(
    model: str, system: str, user: str, *, api_base: str | None, api_key: str | None
) -> str:
    """The LLM fallback (litellm; patched out in offline tests)."""
    from litellm import acompletion

    resp = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        api_base=api_base,
        api_key=api_key,
        temperature=0,
    )
    return resp.choices[0].message.content or ""


async def simulate_attacker_tool_response(
    attacker_tool: str,
    attacker_tool_parameters: str,
    tool_dict: dict[str, Any],
    *,
    sim_model: str,
    api_base: str | None = None,
    api_key: str | None = None,
) -> str:
    """Return the exfiltration tool's simulated response for a ds step-2.

    Cache-first (keyed exactly as upstream: ``f"({tool},{params})"``); on a miss,
    calls ``sim_model`` with the vendored generation prompt and extracts the
    fenced block, mirroring upstream's ``gpt-4-0613`` path.
    """
    history_key = f"({attacker_tool},{attacker_tool_parameters})"
    cache = _load_cache()
    if history_key in cache:
        return cache[history_key]
    if history_key in _MEM:
        return _MEM[history_key]

    attacker_tool_des = json.dumps(tool_dict[attacker_tool], indent=True)
    prompt_filled = DS_ATTACKER_TOOL_RESPONSE_GEN_MESSAGE.format(
        example=EXAMPLE, attacker_tool=attacker_tool_des, tool_parameters=attacker_tool_parameters
    )
    message = await _call_sim(
        sim_model, SYSTEM_MESSAGE, prompt_filled, api_base=api_base, api_key=api_key
    )
    matches = extract_content(message)
    # Upstream indexes ``[0]`` unconditionally; guard so a fenceless completion
    # degrades to the raw message instead of raising.
    simulated_response = matches[0] if matches else message
    if simulated_response.startswith("json"):
        simulated_response = simulated_response[4:]
    simulated_response = simulated_response.strip()
    _MEM[history_key] = simulated_response
    return simulated_response


__all__ = ["extract_content", "simulate_attacker_tool_response"]
