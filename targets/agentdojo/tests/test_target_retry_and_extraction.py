"""Pin :class:`AgentDojoTarget`'s retry loop and the
``_model_output_from_messages`` extraction shape.

Coverage gaps surfaced in the multi-agent code review:
- The 3-retry loop in :meth:`AgentDojoTarget.run` retries when
  ``model_output`` is ``None`` (thinking-only or tool-calling
  assistant) and catches ``AbortAgentError`` (defense abort) like
  upstream.  All other exceptions propagate.
- ``_model_output_from_messages`` should mirror upstream's
  ``model_output_from_messages`` which returns
  ``list[MessageContentBlock] | None`` rather than a flat string.
- ``_content_blocks_to_text`` renders content (block list, plain
  string, or None) into the flat string stored on ``self._last_response``
  and exposed via ``target.query("last_response")``.

These tests build a stand-in pipeline that runs in :func:`asyncio.to_thread`
exactly like the real one but lets us script its behaviour per attempt.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import pytest
from agentdojo.types import ChatMessage

from agentdojo_target.target import (
    AgentDojoTarget,
    _content_blocks_to_text,
    _model_output_from_messages,
)


# ---------------------------------------------------------------------------
# _model_output_from_messages: shape contract matching upstream
# ---------------------------------------------------------------------------


def test_model_output_empty_messages_returns_none() -> None:
    assert _model_output_from_messages([]) is None


def test_model_output_last_is_user_returns_none() -> None:
    # Upstream raises ValueError when last is not assistant; we mirror by
    # returning None so the outer retry loop can decide.
    messages: list[ChatMessage] = [
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "hello"},
    ]
    assert _model_output_from_messages(messages) is None


def test_model_output_last_is_assistant_returns_content_blocks() -> None:
    blocks = [{"type": "text", "content": "hello"}]
    messages: list[ChatMessage] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": blocks, "tool_calls": None},
    ]
    out = _model_output_from_messages(messages)
    assert out == blocks


def test_model_output_last_assistant_with_none_content() -> None:
    # Upstream returns the raw content (which is None); the retry loop in
    # target.run treats this as "no output yet" and tries again.
    messages: list[ChatMessage] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": None},
    ]
    out = _model_output_from_messages(messages)
    assert out is None


def test_model_output_thinking_only_block_is_not_dropped() -> None:
    # Distinct from the old port behaviour which joined only text blocks
    # and returned None for thinking-only responses; upstream returns the
    # thinking-only block list as-is so the outer loop can decide.
    blocks = [{"type": "thinking", "content": "deliberating..."}]
    messages: list[ChatMessage] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": blocks, "tool_calls": None},
    ]
    out = _model_output_from_messages(messages)
    assert out == blocks


# ---------------------------------------------------------------------------
# _content_blocks_to_text: rendering for last_response query slot
# ---------------------------------------------------------------------------


def test_content_blocks_to_text_none() -> None:
    assert _content_blocks_to_text(None) == ""


def test_content_blocks_to_text_plain_string() -> None:
    assert _content_blocks_to_text("ready") == "ready"


def test_content_blocks_to_text_joins_text_blocks() -> None:
    blocks = [
        {"type": "text", "content": "Hello"},
        {"type": "text", "content": " world"},
    ]
    assert _content_blocks_to_text(blocks) == "Hello world"


def test_content_blocks_to_text_skips_thinking_blocks() -> None:
    blocks = [
        {"type": "thinking", "content": "let me consider..."},
        {"type": "text", "content": "Answer: 42"},
    ]
    assert _content_blocks_to_text(blocks) == "Answer: 42"


def test_content_blocks_to_text_skips_non_dict_entries() -> None:
    # Defensive: malformed entries should not blow up the renderer.
    blocks: list[Any] = [
        {"type": "text", "content": "ok"},
        "stray-string",
        None,
    ]
    assert _content_blocks_to_text(blocks) == "ok"


def test_content_blocks_to_text_handles_missing_content_key() -> None:
    blocks = [{"type": "text"}, {"type": "text", "content": "x"}]
    assert _content_blocks_to_text(blocks) == "x"


# ---------------------------------------------------------------------------
# 3-retry loop in AgentDojoTarget.run: behaviour on pipeline.query failure
# ---------------------------------------------------------------------------


class _ScriptedPipeline:
    """Stand-in for the AgentDojo AgentPipeline returned by
    :func:`build_pipeline`.  Records every call and returns
    pre-scripted outcomes."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[Any, ...]] = []

    @property
    def name(self) -> str:
        return "scripted-pipeline"

    @name.setter
    def name(self, _value: str) -> None:
        # build_pipeline assigns; tolerate.
        pass

    def query(
        self,
        query: str,
        runtime: Any,
        env: Any,
    ) -> tuple[Any, Any, Any, list[ChatMessage], dict]:
        self.calls.append((query, env))
        if not self._outcomes:
            raise RuntimeError("scripted pipeline exhausted")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def loop() -> asyncio.AbstractEventLoop:
    import threading

    new_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=new_loop.run_forever, daemon=True)
    thread.start()
    yield new_loop
    new_loop.call_soon_threadsafe(new_loop.stop)
    thread.join(timeout=2)


def _assistant_message(text: str) -> ChatMessage:
    return {
        "role": "assistant",
        "content": [{"type": "text", "content": text}],
        "tool_calls": None,
    }


async def _drive_run(target: AgentDojoTarget, pipeline: _ScriptedPipeline) -> None:
    """Run :meth:`AgentDojoTarget.run` with the scripted pipeline patched in.

    Provides no-op event handlers since the run never needs the real
    optimizer channel for these tests.
    """

    async def send_event(_event: Any) -> Any:
        from superred.core.types.events import ControllableNoInjection

        return ControllableNoInjection(
            event=_event,
            controllable=_event.controllable,
        )

    def emit(_event: Any) -> None:
        pass

    # build_pipeline now returns (pipeline, close); the scripted pipeline
    # has no real client, so the closer is a no-op.
    with patch(
        "agentdojo_target.target.build_pipeline",
        return_value=(pipeline, lambda: None),
    ):
        await target.run(emit=emit, send_event=send_event)


@pytest.mark.asyncio
async def test_retry_loop_succeeds_on_first_attempt() -> None:
    target = AgentDojoTarget(pipeline_model="openai/gpt-4o-2024-05-13")
    target.set_config("user_prompt", "do a thing")
    pre_env = target._build_seed_env_with_overrides()
    pipeline = _ScriptedPipeline(
        [
            (
                "q",
                None,
                pre_env,
                [
                    {"role": "user", "content": "do a thing"},
                    _assistant_message("done"),
                ],
                {},
            ),
        ]
    )
    await _drive_run(target, pipeline)
    assert len(pipeline.calls) == 1
    assert target.query("last_response") == "done"


@pytest.mark.asyncio
async def test_non_abort_exception_propagates() -> None:
    """Non-AbortAgentError exceptions propagate out of the retry loop,
    matching upstream where only AbortAgentError is caught."""
    target = AgentDojoTarget(pipeline_model="openai/gpt-4o-2024-05-13")
    target.set_config("user_prompt", "trigger failure")
    pipeline = _ScriptedPipeline([RuntimeError("rate limit; please retry")])
    with pytest.raises(RuntimeError, match="rate limit"):
        await _drive_run(target, pipeline)
    assert len(pipeline.calls) == 1


@pytest.mark.asyncio
async def test_abort_agent_error_caught_not_propagated() -> None:
    """AbortAgentError (raised by defense pipeline elements) is caught
    and does not propagate, matching upstream.  The error's constructor
    appends an assistant message with the abort reason, so model_output
    is non-None and the retry loop exits on the first attempt."""
    from agentdojo.agent_pipeline.errors import AbortAgentError

    target = AgentDojoTarget(pipeline_model="openai/gpt-4o-2024-05-13")
    target.set_config("user_prompt", "defense test")
    pre_env = target._build_seed_env_with_overrides()
    abort = AbortAgentError(
        "prompt injection detected",
        messages=[{"role": "user", "content": "defense test"}],
        task_environment=pre_env,
    )
    pipeline = _ScriptedPipeline([abort])
    await _drive_run(target, pipeline)
    assert len(pipeline.calls) == 1
    assert target.query("last_response") == "prompt injection detected"


@pytest.mark.asyncio
async def test_retry_loop_retries_until_model_output_present() -> None:
    """The loop retries when the assistant message exists but its
    content is None (still tool-calling, no final text yet)."""
    target = AgentDojoTarget(pipeline_model="openai/gpt-4o-2024-05-13")
    target.set_config("user_prompt", "multi-turn task")
    pre_env = target._build_seed_env_with_overrides()
    pipeline = _ScriptedPipeline(
        [
            # Attempt 1: assistant has no content yet.
            (
                "q",
                None,
                pre_env,
                [
                    {"role": "user", "content": "multi-turn task"},
                    {"role": "assistant", "content": None, "tool_calls": None},
                ],
                {},
            ),
            # Attempt 2: assistant gives a real answer.
            (
                "q",
                None,
                pre_env,
                [
                    {"role": "user", "content": "multi-turn task"},
                    _assistant_message("final answer"),
                ],
                {},
            ),
        ]
    )
    await _drive_run(target, pipeline)
    assert len(pipeline.calls) == 2
    assert target.query("last_response") == "final answer"


@pytest.mark.asyncio
async def test_retry_loop_stops_after_three_attempts() -> None:
    target = AgentDojoTarget(pipeline_model="openai/gpt-4o-2024-05-13")
    target.set_config("user_prompt", "never settles")
    pre_env = target._build_seed_env_with_overrides()
    pipeline = _ScriptedPipeline(
        [
            (
                "q",
                None,
                pre_env,
                [
                    {"role": "user", "content": "never settles"},
                    {"role": "assistant", "content": None, "tool_calls": None},
                ],
                {},
            ),
        ]
        * 5
    )  # Plenty of outcomes; loop should stop at 3.
    await _drive_run(target, pipeline)
    assert len(pipeline.calls) == 3, (
        f"expected 3-attempt cap, got {len(pipeline.calls)}"
    )


# ---------------------------------------------------------------------------
# Reset: per-run state is wiped so a fresh target run starts clean
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_ephemeral_state_resets_per_run_state() -> None:
    target = AgentDojoTarget(pipeline_model="openai/gpt-4o-2024-05-13")
    target.set_config("user_prompt", "first")
    pre_env = target._build_seed_env_with_overrides()
    pipeline = _ScriptedPipeline(
        [
            (
                "q",
                None,
                pre_env,
                [
                    {"role": "user", "content": "first"},
                    _assistant_message("first-answer"),
                ],
                {},
            ),
        ]
    )
    await _drive_run(target, pipeline)
    assert target.query("last_response") == "first-answer"

    await target.reset_ephemeral_state()
    # After reset, last_response is wiped and the conversation history
    # is empty so a fresh evaluation cannot accidentally reuse it.
    assert target.query("last_response") == ""
    assert json.loads(target.query("conversation_history")) == []
    assert json.loads(target.query("function_call_trace")) == []
