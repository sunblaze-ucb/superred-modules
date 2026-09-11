"""Agent-side plumbing for the InjecAgent target.

Ports the tool-dictionary loader and the prompt/message construction from
upstream ``src/utils.py`` (``get_tool_dict`` / ``transform_tool_format_gpt``) and
the two ``predict_one_case`` functions (``src/evaluate_prompted_agent.py`` for the
text-ReAct path, ``src/evaluate_finetuned_agent.py`` for the function-calling
path). The actual model calls go through litellm in :func:`call_prompted` /
:func:`call_finetuned`, which tests patch to run offline.
"""

from __future__ import annotations

import json
import os
from typing import Any

from injecagent_target._vendor.agent_prompts import PROMPT_DICT

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# A fixed synthetic call id, matching upstream (evaluate_finetuned_agent.py:46).
_CALL_ID = "call_dx6NRJIZOLS2GS7HtIFxVpyG"

_TOOL_DICT_CACHE: dict[bool, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Tool dictionary (port of upstream get_tool_dict / transform_tool_format_gpt)
# ---------------------------------------------------------------------------
def transform_tool_format_gpt(tool: dict[str, Any]) -> dict[str, Any]:
    """Port of upstream ``transform_tool_format_gpt`` (src/utils.py:72-114).

    Converts an InjecAgent/ToolEmu tool spec into an OpenAI function schema.
    """
    transformed_tool: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["summary"],
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    for param in tool["parameters"]:
        if param["type"] == "array":
            if "array of" in param["description"]:
                start = param["description"].index("array of")
                item_des = param["description"][start + len("array of") :].strip()
            elif "list of" in param["description"]:
                start = param["description"].index("list of")
                item_des = param["description"][start + len("list of") :].strip()
            else:
                item_des = param["description"]
            transformed_tool["function"]["parameters"]["properties"][param["name"]] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        param["name"]: {"type": "string", "description": item_des},
                    },
                },
            }
        else:
            transformed_tool["function"]["parameters"]["properties"][param["name"]] = {
                "type": param["type"],
                "description": param["description"],
            }

        if param.get("required"):
            transformed_tool["function"]["parameters"]["required"].append(param["name"])
    return transformed_tool


def get_tool_dict(gpt_format: bool = False) -> dict[str, Any]:
    """Port of upstream ``get_tool_dict`` (src/utils.py:117-130), reading the
    vendored ``data/tools.json``. Cached per ``gpt_format`` (a fresh dict is
    always built from file, so the cache never double-prefixes tool names)."""
    if gpt_format in _TOOL_DICT_CACHE:
        return _TOOL_DICT_CACHE[gpt_format]
    tool_dict: dict[str, Any] = {}
    with open(os.path.join(_DATA, "tools.json"), encoding="utf-8") as f:
        toolkits = json.load(f)
    for toolkit in toolkits:
        kit_name = toolkit["toolkit"]
        for tool in toolkit["tools"]:
            tool_name = kit_name + tool["name"]
            tool["name"] = tool_name
            tool_dict[tool["name"]] = transform_tool_format_gpt(tool) if gpt_format else tool
    _TOOL_DICT_CACHE[gpt_format] = tool_dict
    return tool_dict


# ---------------------------------------------------------------------------
# Prompt / message construction
# ---------------------------------------------------------------------------
def build_prompted_input(
    case: dict[str, Any], tool_dict: dict[str, Any], prompt_type: str, step_2: bool = False
) -> tuple[str, str]:
    """Build (system_prompt, filled user_prompt) for the text-ReAct path.

    Port of ``predict_one_case`` (evaluate_prompted_agent.py:12-24).
    """
    available_tool_names = [case["User Tool"]] + case["Attacker Tools"]
    available_tool_descriptions = json.dumps(
        [tool_dict[tool] for tool in available_tool_names], indent=True
    )
    input_ = case["User Instruction"]

    if step_2:
        agent_scratchpad = (
            f"\n{case['Thought']}\nAction: {case['User Tool']}\nAction Input: "
            f"{case['Tool Parameters']}\nObservation: {case['Tool Response']}\n"
            f"Thought: {case['Attacker Tool Thought']}\nAction: {case['Attacker Tools'][0]}\n"
            f"Action Input: {case['Attacker Tool Parameters']}\nObservation: "
            f"{case['Attacker Tool Response']}\n"
        )
    else:
        agent_scratchpad = (
            f"\n{case['Thought']}\nAction: {case['User Tool']}\nAction Input: "
            f"{case['Tool Parameters']}\nObservation: {case['Tool Response']}\n"
        )

    system_prompt, user_prompt = PROMPT_DICT[prompt_type]
    user_prompt_filled = user_prompt.format(
        available_tool_descriptions=available_tool_descriptions,
        input=input_,
        available_tool_names=available_tool_names,
        agent_scratchpad=agent_scratchpad,
    )
    return system_prompt, user_prompt_filled


def build_finetuned_messages(
    case: dict[str, Any], tool_dict_gpt: dict[str, Any], prompt_type: str, step_2: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build (messages, tools) for the native function-calling path.

    Port of ``predict_one_case`` (evaluate_finetuned_agent.py:37-79). The user
    tool's call + poisoned response are pre-seeded as prior turns; ``step_2``
    additionally seeds the first attacker-tool call + its simulated response.
    """
    available_tool_names = [case["User Tool"]] + case["Attacker Tools"]
    available_tools = [tool_dict_gpt[tool] for tool in available_tool_names]
    system_prompt, _ = PROMPT_DICT[prompt_type]

    function_name = case["User Tool"]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": case["User Instruction"]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": _CALL_ID,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(case["Tool Parameters"]),
                    },
                }
            ],
        },
        {
            "tool_call_id": _CALL_ID,
            "role": "tool",
            "name": function_name,
            "content": json.dumps(case["Tool Response"]),
        },
    ]

    if step_2:
        tool_call = case["tool_calls"][0]
        messages.append(
            {
                "role": "assistant",
                "content": case["output"],
                "tool_calls": [
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": json.dumps(case["Attacker Tool Parameters"]),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "name": tool_call["function"]["name"],
                "content": case["Attacker Tool Response"],
            }
        )

    return messages, available_tools


# ---------------------------------------------------------------------------
# Model calls (litellm; patched out in offline tests)
# ---------------------------------------------------------------------------
async def call_prompted(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    api_base: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    """One text-completion turn (chat messages -> text). litellm subsumes the
    per-provider prompt templating upstream did via ``src/models.py``."""
    from litellm import acompletion

    resp = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


async def call_finetuned(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    api_base: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """One native function-calling turn (returns (content, tool_calls))."""
    from litellm import acompletion

    resp = await acompletion(
        model=model,
        messages=messages,
        tools=tools,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    message = resp.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        tool_calls = [
            tc.model_dump() if hasattr(tc, "model_dump") else dict(tc) for tc in tool_calls
        ]
    else:
        tool_calls = None
    return message.content, tool_calls


__all__ = [
    "transform_tool_format_gpt",
    "get_tool_dict",
    "build_prompted_input",
    "build_finetuned_messages",
    "call_prompted",
    "call_finetuned",
]
