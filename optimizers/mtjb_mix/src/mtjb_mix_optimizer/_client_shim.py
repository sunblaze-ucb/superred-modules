"""Stdlib-only ``client.unified_llm_client`` shim for the vendored MT-JailBench engine.

The vendored engine files import ``from client.unified_llm_client import
UnifiedLLMClient, Conversation``. Upstream's real module drags the OpenAI /
Anthropic / Bedrock / Gemini SDKs; we install this shim under that exact name in
``sys.modules`` so the vendored code imports with no SDKs.

* ``Conversation`` is a faithful stdlib-only re-implementation of upstream's
  message-history holder (API: ``history``, ``system_prompt``,
  ``add_user_message``, ``add_assistant_message``, ``serialize``, ``from_list``).
* ``UnifiedLLMClient`` is a *facade*: the vendored code constructs it from config
  strings, but ``generate`` ignores them and routes to the superred
  ``LLMClient`` bound (per task, via a ``ContextVar``) by the engine bridge. The
  call is synchronous (the vendored code is synchronous); it bridges to the async
  ``LLMClient`` on the running loop via ``run_coroutine_threadsafe`` (the bridge
  runs the vendored code in a worker thread). ``temperature`` is never forwarded
  (superred house rule).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import types
from contextvars import ContextVar
from typing import Any

#: Bound per task by the bridge: (LLMClient, running event loop).
_LLM_CTX: ContextVar[tuple[Any, asyncio.AbstractEventLoop]] = ContextVar("mtjb_mix_llm")


@contextlib.contextmanager
def bind_llm(llm: Any, loop: asyncio.AbstractEventLoop):
    """Bind the LLMClient + loop for the duration of a vendored-engine call.

    ``asyncio.to_thread`` copies the current context into the worker thread, so a
    facade constructed there reads the same binding. Concurrency-safe: each task
    sets its own ContextVar value.
    """
    token = _LLM_CTX.set((llm, loop))
    try:
        yield
    finally:
        _LLM_CTX.reset(token)


class Conversation:
    """Stdlib-only message-history holder (upstream API, no SDKs)."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self.history: list[dict[str, str]] = []
        self.system_prompt = system_prompt

    def add_user_message(self, message: str) -> None:
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str) -> None:
        self.history.append({"role": "assistant", "content": message})

    def serialize(self, include_system: bool = True) -> list[dict]:
        copy = self.history.copy()
        if include_system and self.system_prompt:
            copy.insert(0, {"role": "system", "content": self.system_prompt})
        return copy

    def __str__(self, include_system: bool = True) -> str:
        if include_system and self.system_prompt:
            all_history = [{"role": "system", "content": self.system_prompt}, *self.history]
            return json.dumps(all_history, indent=2)
        return json.dumps(self.history, indent=2)

    @classmethod
    def from_list(cls, messages: list) -> Conversation:
        system_prompt = None
        history: list[dict[str, str]] = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system" and system_prompt is None:
                system_prompt = content
            else:
                history.append({"role": role, "content": content})
        conv = cls(system_prompt)
        conv.history = history
        return conv


class UnifiedLLMClient:
    """Facade over superred's async ``LLMClient`` (config ignored; routed via ctx)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Vendored code constructs this from model/provider/base_url/etc.; the
        # experiment, not the attack config, decides the model, so all of it is
        # ignored and the call is routed to the bound LLMClient.
        self._args = args
        self._kwargs = kwargs

    def generate(
        self,
        user_input: str,
        conversation: Conversation | None = None,
        system_prompt: str | None = None,
        response_format: Any | None = None,
        **_kw: Any,
    ) -> tuple[str, Conversation]:
        # Any other kwargs the vendored code might pass (e.g. a sampling
        # parameter) are absorbed by ``_kw`` and never forwarded: the experiment,
        # not the attack, controls the model call (superred house rule).
        llm, loop = _LLM_CTX.get()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if conversation is not None:
            messages.extend(conversation.serialize(include_system=False))
        messages.append({"role": "user", "content": user_input})

        kwargs: dict[str, Any] = {}
        if response_format is not None:
            kwargs["response_format"] = response_format

        future = asyncio.run_coroutine_threadsafe(llm.complete(messages, **kwargs), loop)
        response = future.result()
        text = response.choices[0].message.content or ""

        conv = conversation if conversation is not None else Conversation(system_prompt)
        conv.add_user_message(user_input)
        conv.add_assistant_message(text)
        return text, conv


def install() -> None:
    """Register this shim as ``client`` / ``client.unified_llm_client``.

    Idempotent. Uses ``setdefault`` so a pre-existing (identical) registration by
    another vendored module in the same process is preserved.
    """
    if "client.unified_llm_client" in sys.modules:
        return
    pkg = types.ModuleType("client")
    pkg.__path__ = []  # mark as a package
    sys.modules.setdefault("client", pkg)
    sys.modules.setdefault("client.unified_llm_client", sys.modules[__name__])
