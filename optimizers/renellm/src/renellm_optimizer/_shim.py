"""Stdlib drop-in for ReNeLLM's ``utils/llm_completion_utils`` helper.

Upstream's ``utils/llm_completion_utils.py`` builds ``openai``/``anthropic``
SDK clients. That file is the *only* upstream util we do not vendor: it is
replaced by this shim so the byte-identical vendored rewrite/judge helpers
(which do ``from utils.llm_completion_utils import chatCompletion``) route every
model call through the constrained superred ``LLMClient`` instead.

The shim keeps upstream's ``chatCompletion`` signature verbatim so the vendored
callers bind to it unchanged, but ignores every routing / sampling / credential
knob: the superred ``LLMClient`` owns the model and credentials, and no sampling
parameter is ever set by this package (house rule; see
``tests/test_no_temperature.py``). Only ``messages`` is forwarded.

The forwarding target is a per-call *bridge* stored in :data:`BRIDGE`. The
optimizer sets it (in its async context) before invoking the synchronous
vendored helpers; ``asyncio.to_thread`` copies the context into the worker
thread, where the vendored code runs and calls back through the bridge to the
event loop. ``claudeCompletion`` (the upstream *attacked-model* helper) is
deliberately absent: the model under attack is reached via superred injection,
never through this shim.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable

#: Per-call synchronous bridge to ``self.llm``. Set by the optimizer before the
#: vendored helpers run; read here from inside the worker thread. Raises
#: ``LookupError`` if the vendored code is ever called without a bridge armed.
BRIDGE: contextvars.ContextVar[Callable[[list[dict[str, str]]], str]] = contextvars.ContextVar(
    "renellm_llm_bridge"
)


def chatCompletion(  # noqa: N802 - upstream name kept so vendored imports bind
    model: object,
    messages: list[dict[str, str]],
    sampling: object,
    retry_times: object,
    round_sleep: object,
    fail_sleep: object,
    api_key: object,
    base_url: object = None,
) -> str:
    """Route one chat completion to ``self.llm`` via the armed bridge.

    All arguments except ``messages`` are accepted for signature compatibility
    with the vendored callers and then dropped: the constrained client fixes the
    model and credentials, this package never sets a sampling parameter, and the
    real retry/backoff policy lives in the client, not here.
    """
    del model, sampling, retry_times, round_sleep, fail_sleep, api_key, base_url
    bridge = BRIDGE.get()
    return bridge(messages)
