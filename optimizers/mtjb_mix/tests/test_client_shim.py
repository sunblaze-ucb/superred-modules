"""Tests for the stdlib client shim (no heavy deps needed).

Covers the Conversation API and the synchronous facade's bridge to an async
LLMClient via ``run_coroutine_threadsafe`` -- the mechanism the Mix engine
bridge relies on.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from mtjb_mix_optimizer._client_shim import (
    Conversation,
    UnifiedLLMClient,
    bind_llm,
)


def test_conversation_roundtrip() -> None:
    conv = Conversation(system_prompt="SYS")
    conv.add_user_message("u1")
    conv.add_assistant_message("a1")
    assert conv.serialize() == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    assert conv.serialize(include_system=False)[0]["role"] == "user"


def test_conversation_from_list() -> None:
    conv = Conversation.from_list(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    )
    assert conv.system_prompt == "S"
    assert conv.history == [{"role": "user", "content": "u"}]


def _resp(content: str) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    return r


def test_facade_bridges_sync_generate_to_async_llm() -> None:
    async def main():
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=_resp("REPLY"))
        loop = asyncio.get_running_loop()
        with bind_llm(llm, loop):
            # The vendored engine calls .generate synchronously; here we run it in
            # a worker thread as the bridge does, so run_coroutine_threadsafe can
            # schedule back onto this loop.
            text, conv = await asyncio.to_thread(
                lambda: UnifiedLLMClient("model").generate("hi", system_prompt="sys")
            )
        return text, conv, llm

    text, conv, llm = asyncio.run(main())
    assert text == "REPLY"
    assert conv.history[-1] == {"role": "assistant", "content": "REPLY"}
    # system + user message were sent; no temperature forwarded
    sent = llm.complete.await_args.args[0]
    assert sent[0] == {"role": "system", "content": "sys"}
    assert "temperature" not in llm.complete.await_args.kwargs


def test_facade_forwards_response_format() -> None:
    async def main():
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=_resp("{}"))
        loop = asyncio.get_running_loop()
        with bind_llm(llm, loop):
            await asyncio.to_thread(
                lambda: UnifiedLLMClient().generate("hi", response_format={"type": "json_object"})
            )
        return llm

    llm = asyncio.run(main())
    assert llm.complete.await_args.kwargs["response_format"] == {"type": "json_object"}
